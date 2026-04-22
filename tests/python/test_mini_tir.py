# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Mini-TIR exhaustive test suite — combinatorial grammar coverage.

Mini-TIR is its own language. Its grammar is defined entirely by the IR
nodes in :mod:`tvm_ffi.testing.mini.tir` and their associated traits.
**Any IR we can construct from those nodes should print, and anything
that prints should roundtrip-parse.

Organization:

* **T0 — Atoms**: single nodes, no nesting (each dtype, each literal kind).
* **T1 — Simple expressions**: one or two children (binops with Vars, Not).
* **T2 — Composed expressions**: nested operators, indexed loads.
* **T3 — Single statements**: Bind, BufferStore, AssertStmt, Evaluate.
* **T4 — Control flow (single level)**: If, While, For, Block, SeqStmt.
* **T5 — Nested control flow**: if-in-for, for-in-for, etc.
* **T6 — Functions**: PrimFunc with various shapes.
* **T7 — Modules**: IRModule with one or more functions.
* **T8 — End-to-end programs**: realistic mini-TIR kernels.
"""

from __future__ import annotations

from typing import Any

import pytest

from tvm_ffi import dtype as ffi_dtype
from tvm_ffi import pyast
from tvm_ffi.testing.mini import tir as mt
from tvm_ffi.testing.roundtrip import assert_roundtrip


# ============================================================================
# New language-module protocol checks
# (parser hooks — ``load`` / ``bind`` / ``buffer_store`` / ... — are
# now fully wired in ``mini/tir.py``. Nothing to patch here.)
# ============================================================================


def test_protocol_prim_ty_is_attribute_not_factory():
    """``T.int32`` must be a :class:`PrimTy` instance itself."""
    assert isinstance(mt.TLang.int32, mt.PrimTy)
    assert mt.TLang.int32.dtype == "int32"
    assert isinstance(mt.TLang.float64, mt.PrimTy)
    assert mt.TLang.bool.dtype == "bool"


def test_protocol_prim_ty_zero_arg_call_builds_var():
    """``T.int32()`` (zero-arg) builds a default-named :class:`mt.Var` via
    ``__ffi_make_var__``."""
    v = mt.TLang.int32()
    assert isinstance(v, mt.Var)
    assert v.name == "_"
    assert v.ty.dtype == "int32"

    v = mt.TLang.float32()
    assert isinstance(v, mt.Var)
    assert v.name == "_"
    assert v.ty.dtype == "float32"


def test_protocol_prim_ty_var_name_builds_named_var():
    """``T.int32(var_name="x")`` builds a Var with that name."""
    v = mt.TLang.int32(var_name="x")
    assert isinstance(v, mt.Var)
    assert v.name == "x"
    assert v.ty.dtype == "int32"


def test_protocol_prim_ty_value_and_var_name_conflict():
    """Passing both ``value`` and ``var_name`` raises :class:`TypeError`."""
    import pytest  # noqa: PLC0415

    with pytest.raises(TypeError, match="cannot pass both"):
        mt.TLang.int32(42, var_name="x")
    with pytest.raises(TypeError, match="cannot pass both"):
        mt.TLang.int32(value=42, var_name="x")


# ============================================================================
# Literal dispatch protocol checks.
# ============================================================================


def test_protocol_default_ty_hooks_wired():
    """``TLang.__ffi_default_{int,float,bool}_ty__`` must all point at
    callable type-handles (mini-TIR's :class:`_DtypeHandle` instances).
    """
    for hook, expected_dtype in [
        ("__ffi_default_int_ty__", "int32"),
        ("__ffi_default_float_ty__", "float32"),
        ("__ffi_default_bool_ty__", "bool"),
    ]:
        handle = getattr(mt.TLang, hook, None)
        assert callable(handle), f"TLang.{hook} must be a callable type-handle"
        assert isinstance(handle, mt.PrimTy), (
            f"TLang.{hook} must be a PrimTy-compatible handle"
        )
        assert handle.dtype == expected_dtype, (
            f"TLang.{hook}.dtype must be {expected_dtype!r}"
        )


def test_protocol_parse_literal_python_types_dispatch():
    """:func:`parse_literal` routes each Python primitive to the right Imm class."""
    from tvm_ffi.pyast_trait_parse import parse_literal  # noqa: PLC0415

    parser = pyast.IRParser(lang_modules=mt.LANG_MODULES)

    v = parse_literal(parser, pyast.Literal(value=1))
    assert isinstance(v, mt.IntImm)
    assert v.value == 1
    assert str(v.dtype) == "int32"

    v = parse_literal(parser, pyast.Literal(value=3.14))
    assert isinstance(v, mt.FloatImm)
    assert v.value == 3.14
    assert str(v.dtype) == "float32"

    v = parse_literal(parser, pyast.Literal(value=True))
    assert isinstance(v, mt.IntImm)
    assert v.value == 1
    assert str(v.dtype) == "bool"

    v = parse_literal(parser, pyast.Literal(value=False))
    assert isinstance(v, mt.IntImm)
    assert v.value == 0
    assert str(v.dtype) == "bool"


def test_protocol_binop_literal_wrapping_end_to_end():
    """``a + 1`` (sugar) round-trips to ``Add(Var("a"), IntImm(1, int32))``."""
    parser = pyast.IRParser(
        lang_modules=mt.LANG_MODULES, var_factory=mt.make_var_factory,
    )
    parser.push_scope()
    parser.define("a", mt.Var(name="a", ty=mt.PrimTy(dtype="int32")))

    node = pyast.Operation(
        op=pyast.OperationKind.Add,
        operands=[pyast.Id(name="a"), pyast.Literal(value=1)],
    )
    result = parser.visit_operation(node)

    assert isinstance(result, mt.Add)
    assert isinstance(result.lhs, mt.Var)
    assert result.lhs.name == "a"
    assert isinstance(result.rhs, mt.IntImm)
    assert result.rhs.value == 1
    assert str(result.rhs.dtype) == "int32"


def test_protocol_prim_ty_one_arg_call_builds_imm():
    """``T.int32(42) → IntImm`` and ``T.float32(3.5) → FloatImm``."""
    i = mt.TLang.int64(42)
    assert isinstance(i, mt.IntImm)
    assert i.value == 42
    f = mt.TLang.float32(3.5)
    assert isinstance(f, mt.FloatImm)
    assert f.value == 3.5


def test_protocol_ffi_make_var_hook_wired_on_tlang():
    """``TLang.__ffi_make_var__`` must be a callable on the module."""
    hook = getattr(mt.TLang, "__ffi_make_var__", None)
    assert callable(hook), "TLang.__ffi_make_var__ must be wired"


def test_protocol_ffi_make_var_from_prim_ty():
    """Hook accepts a PrimTy and builds a Var carrying that type."""
    parser = pyast.IRParser(lang_modules=mt.LANG_MODULES)
    var = mt.TLang.__ffi_make_var__(parser, "x", mt.TLang.int32)
    assert isinstance(var, mt.Var)
    assert var.name == "x"
    assert var.ty.dtype == "int32"


def test_protocol_parse_value_def_with_default_ty():
    """End-to-end: ``parse_value_def`` with ``annotation=None`` uses default_ty,
    calls the ``__ffi_make_var__`` hook, and registers the Var in scope."""
    from tvm_ffi.pyast_trait_parse import parse_value_def  # noqa: PLC0415

    parser = pyast.IRParser(lang_modules=mt.LANG_MODULES)
    var = parse_value_def(
        parser,
        "i",
        annotation=None,
        make_var=mt.TLang.__ffi_make_var__,
        default_ty=mt.TLang.int32,
    )
    assert isinstance(var, mt.Var)
    assert var.name == "i"
    assert var.ty.dtype == "int32"
    # Side effect: registered in the innermost scope.
    assert parser.lookup("i") is var


def test_protocol_ffi_assign_hook_wired_on_tlang():
    """``TLang.__ffi_assign__`` must be a callable on the module."""
    hook = getattr(mt.TLang, "__ffi_assign__", None)
    assert callable(hook), "TLang.__ffi_assign__ must be wired"


# ============================================================================
# Operation dispatch protocol — ``__ffi_op_classes__`` map +
# auto-derived ``T.<Name>`` factories.
# ============================================================================


def test_protocol_op_classes_map_wired():
    """``TLang.__ffi_op_classes__`` is a dict {OperationKind: dotted-path str}."""
    op_classes = getattr(mt.TLang, "__ffi_op_classes__", None)
    assert isinstance(op_classes, dict), "TLang.__ffi_op_classes__ must be a dict"
    for kind, ref in op_classes.items():
        assert isinstance(kind, int), f"key must be OperationKind int, got {type(kind).__name__}"
        assert isinstance(ref, str), f"value must be a dotted path str, got {type(ref).__name__}"
        assert "." in ref, f"value must be a dotted path like 'T.Add', got {ref!r}"


def test_protocol_t_factories_auto_registered():
    """Each ``T.<Name>`` mentioned in ``__ffi_op_classes__`` must be a callable
    on the lang module (``partial(parse_binop, ir_class=...)`` or unary equivalent).
    """
    for ref in mt.TLang.__ffi_op_classes__.values():
        prefix, _, name = ref.partition(".")
        assert prefix == "T", f"only T.* paths supported in this dialect, got {ref!r}"
        factory = getattr(mt.TLang, name, None)
        assert callable(factory), f"{ref} must be auto-registered from __ffi_op_classes__"


def test_protocol_op_classes_unique_op_symbols():
    """Each ``BinOpTraits.op`` / ``UnaryOpTraits.op`` symbol used in the
    dialect must appear in at most one IR class — otherwise the printer's
    sugar form is ambiguous to parse.
    """
    seen: dict[tuple[str, int], type] = {}  # (op_sym, arity) -> class
    for cls in mt._OP_KIND_TO_IR_CLASS.values():
        trait = cls.__ffi_ir_traits__
        op_sym = trait.op
        if hasattr(trait, "lhs"):
            arity = 2
        elif hasattr(trait, "operand"):
            arity = 1
        else:
            raise AssertionError(f"unexpected trait shape on {cls.__name__}")
        key = (op_sym, arity)
        assert key not in seen, (
            f"op symbol collision on ({op_sym!r}, arity={arity}): "
            f"both {seen[key].__name__} and {cls.__name__} claim it — "
            f"sugar parsing would be ambiguous"
        )
        seen[key] = cls


def test_protocol_sugar_dispatch_to_correct_ir_class():
    """``a + b`` (sugar) routes through ``__ffi_op_classes__['Add']`` =
    ``"T.Add"`` → :class:`partial`-wrapped :func:`parse_binop` → :class:`mt.Add`.
    """
    parser = pyast.IRParser(
        lang_modules=mt.LANG_MODULES,
        var_factory=mt.make_var_factory,
    )
    parser.push_scope()
    parser.define("a", mt.Var(name="a", ty=mt.PrimTy(dtype="int32")))
    parser.define("b", mt.Var(name="b", ty=mt.PrimTy(dtype="int32")))

    sugar_node = pyast.Operation(
        op=pyast.OperationKind.Add,
        operands=[pyast.Id(name="a"), pyast.Id(name="b")],
    )
    result = parser.visit_operation(sugar_node)
    assert isinstance(result, mt.Add)
    assert result.lhs.name == "a"
    assert result.rhs.name == "b"


def test_protocol_unary_sugar_dispatch_to_correct_ir_class():
    """``not a`` (sugar) routes through ``__ffi_op_classes__['Not']`` =
    ``"T.Not"`` → :class:`partial`-wrapped :func:`parse_unaryop` → :class:`mt.Not`.
    """
    parser = pyast.IRParser(
        lang_modules=mt.LANG_MODULES,
        var_factory=mt.make_var_factory,
    )
    parser.push_scope()
    parser.define("a", mt.Var(name="a", ty=mt.PrimTy(dtype="bool")))

    sugar_node = pyast.Operation(
        op=pyast.OperationKind.Not,
        operands=[pyast.Id(name="a")],
    )
    result = parser.visit_operation(sugar_node)
    assert isinstance(result, mt.Not)
    assert result.a.name == "a"

# ============================================================================
# Constants
# ============================================================================


PRIM_INT_DTYPES = [
    "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
    "bool",
]
PRIM_FLOAT_DTYPES = ["float16", "float32", "float64"]
PRIM_DTYPES = PRIM_INT_DTYPES + PRIM_FLOAT_DTYPES

BINOPS_ARITH = [(mt.Add, "+"), (mt.Sub, "-"), (mt.Mul, "*")]
BINOPS_CMP = [(mt.Lt, "<"), (mt.Eq, "==")]
BINOPS_LOGIC = [(mt.And, "and"), (mt.Or, "or")]
BINOPS_ALL = BINOPS_ARITH + BINOPS_CMP + BINOPS_LOGIC

FOR_KINDS = ["serial", "parallel", "unroll", "vectorized"]


# ============================================================================
# Helpers
# ============================================================================


def _v(name: str, dtype: str = "int32") -> mt.Var:
    return mt.Var(name=name, ty=mt.PrimTy(dtype=dtype))


def _int(value: int, dtype: str = "int32") -> mt.IntImm:
    """Build an ``IntImm`` with dtype always wrapped via :func:`tvm_ffi.dtype`."""
    return mt.IntImm(value=value, dtype=ffi_dtype(dtype))


def _float(value: float, dtype: str = "float32") -> mt.FloatImm:
    """Same dtype-wrapping contract as :func:`_int`."""
    return mt.FloatImm(value=value, dtype=ffi_dtype(dtype))


def _wrap(body: list, params: list | None = None) -> mt.PrimFunc:
    """Wrap stmts in a minimal PrimFunc named ``main`` for roundtrip context."""
    return mt.PrimFunc(
        name="test_func",
        params=list(params) if params else [],
        body=list(body),
    )


def _rt(ir: Any) -> None:
    """Roundtrip helper using mini-TIR's lang modules and var factory."""
    assert_roundtrip(
        ir,
        lang_modules=mt.LANG_MODULES,
        var_factory=mt.make_var_factory,
        verbose=True,
    )


# ============================================================================
# Tier 0 — Atoms
# ============================================================================


@pytest.mark.parametrize("dtype", PRIM_DTYPES)
def test_t0_prim_ty_prints(dtype):
    """Every PrimTy dtype prints as ``T.<dtype>``."""
    assert pyast.to_python(mt.PrimTy(dtype=dtype)) == f"T.{dtype}"


@pytest.mark.parametrize("shape", [[1], [128], [128, 64], [2, 3, 4]])
@pytest.mark.parametrize("dtype", ["int32", "float32", "bool"])
def test_t0_buffer_ty_prints(shape, dtype):
    """BufferTy with various shapes/dtypes; defaults (strides/offset/scope) elided."""
    bt = mt.BufferTy(shape=shape, dtype=dtype)
    text = pyast.to_python(bt)
    assert text.startswith("T.Buffer(")
    assert f'"{dtype}"' in text


def test_t0_buffer_ty_with_scope():
    """BufferTy with non-default scope is rendered."""
    bt = mt.BufferTy(shape=[64], dtype="float32", scope="shared")
    text = pyast.to_python(bt)
    assert "shared" in text


@pytest.mark.parametrize("dtype", PRIM_INT_DTYPES)
@pytest.mark.parametrize("value", [0, 1, 42, -1])
def test_t0_int_imm_prints(dtype, value):
    """IntImm prints across all int dtypes and representative values."""
    text = pyast.to_python(_int(value, dtype))
    # int32 prints bare; other dtypes wrap as T.<dtype>(v) (or T.bool(True/False)).
    if dtype == "int32":
        assert text == str(value)
    elif dtype == "bool":
        assert text in (f"T.bool({bool(value)})", f"T.bool({value})")
    else:
        # T.int64(42), T.uint8(0), etc.
        assert text.startswith(f"T.{dtype}(")


@pytest.mark.parametrize("dtype", PRIM_FLOAT_DTYPES)
@pytest.mark.parametrize("value", [0.0, 1.5, -3.14])
def test_t0_float_imm_prints(dtype, value):
    """FloatImm prints across all float dtypes."""
    text = pyast.to_python(_float(value, dtype))
    assert text.startswith(f"T.{dtype}(")


def test_t0_var_use_site_prints_name():
    """Standalone Var prints just the name (use-site)."""
    assert pyast.to_python(_v("x", "int32")) == "x"


def test_t0_cast_level_zero_print():
    """Cast is a Level-0 node — prints as ``mini.tir.Cast(target=..., value=...)``."""
    text = pyast.to_python(mt.Cast(target=mt.PrimTy(dtype="float32"), value=_int(1)))
    assert "Cast" in text


# ============================================================================
# Tier 1 — Simple expressions (one or two children)
# ============================================================================


@pytest.mark.parametrize(("op_cls", "op_str"), BINOPS_ALL)
def test_t1_binop_two_vars_sugar(op_cls, op_str):
    """Each binop with two Vars renders with infix sugar: ``a OP b``."""
    a, b = _v("a"), _v("b")
    expr = op_cls(lhs=a, rhs=b)
    result_dtype = "bool" if op_str in ("<", "==", "and", "or") else "int32"
    func = _wrap(
        [mt.Bind(var=_v("c", result_dtype), value=expr)],
        params=[a, b],
    )
    text = pyast.to_python(func)
    assert f"a {op_str} b" in text
    _rt(func)


@pytest.mark.parametrize(("op_cls", "op_str"), BINOPS_ARITH + BINOPS_CMP)
def test_t1_binop_var_and_int_literal(op_cls, op_str):
    """Each binop with Var + IntImm: ``a OP 1`` (sugar still applies)."""
    a = _v("a")
    expr = op_cls(lhs=a, rhs=_int(1))
    result_dtype = "bool" if op_str in ("<", "==") else "int32"
    func = _wrap(
        [mt.Bind(var=_v("c", result_dtype), value=expr)],
        params=[a],
    )
    text = pyast.to_python(func)
    assert f"a {op_str} 1" in text
    _rt(func)


@pytest.mark.parametrize(("op_cls", "func_name"), [
    (mt.Add, "Add"), (mt.Sub, "Sub"), (mt.Mul, "Mul"),
    (mt.Lt, "Lt"), (mt.Eq, "Eq"),
    (mt.And, "And"), (mt.Or, "Or"),
])
def test_t1_binop_two_literals_no_sugar(op_cls, func_name):
    """Each binop with two literals refuses sugar: ``T.<Op>(1, 2)``."""
    expr = op_cls(lhs=_int(1), rhs=_int(2))
    result_dtype = "bool" if func_name in ("Lt", "Eq", "And", "Or") else "int32"
    func = _wrap(
        [mt.Bind(var=_v("c", result_dtype), value=expr)],
    )
    text = pyast.to_python(func)
    assert f"T.{func_name}(1, 2)" in text
    _rt(func)


def test_t1_not_on_var():
    """Unary ``not a`` on a Var."""
    a = _v("a", "bool")
    func = _wrap(
        [mt.Bind(var=_v("c", "bool"), value=mt.Not(a=a))],
        params=[a],
    )
    text = pyast.to_python(func)
    assert "not a" in text
    _rt(func)


def test_t1_call_no_args():
    """``T.foo()`` — call with literal callee, no args."""
    func = _wrap(
        [mt.Bind(var=_v("c"), value=mt.Call(op_name="T.foo", args=[]))],
    )
    text = pyast.to_python(func)
    assert "T.foo()" in text
    _rt(func)


def test_t1_call_with_args():
    """``T.foo(a, b)`` — call with literal callee, two var args."""
    a, b = _v("a"), _v("b")
    func = _wrap(
        [mt.Bind(var=_v("c"), value=mt.Call(op_name="T.foo", args=[a, b]))],
        params=[a, b],
    )
    text = pyast.to_python(func)
    assert "T.foo(a, b)" in text
    _rt(func)


@pytest.mark.parametrize("indices", [
    [0],            # single literal index
    ["i"],          # single var index (replaced with _v)
    [0, 1],         # multi literal
    ["i", "j"],     # multi var
    ["i", 0],       # mixed
])
def test_t1_buffer_load_index_shapes(indices):
    """BufferLoad across single/multi/mixed index shapes."""
    A = _v("A", "int32")
    params = [A]
    seen = {}
    real_indices = []
    for idx in indices:
        if isinstance(idx, str):
            v = seen.setdefault(idx, _v(idx))
            if v not in params:
                params.append(v)
            real_indices.append(v)
        else:
            # Match the parser's canonical wrapping: a literal index
            # roundtrips as an ``IntImm(v, int32)``.
            real_indices.append(_int(idx))
    expr = mt.BufferLoad(source=A, indices=real_indices)
    func = _wrap(
        [mt.Bind(var=_v("c", "int32"), value=expr)],
        params=params,
    )
    pyast.to_python(func)  # just verify it prints
    _rt(func)


# ============================================================================
# Tier 2 — Composed expressions (multi-level)
# ============================================================================


def test_t2_nested_arith():
    """``(a + b) * c`` — multi-level arithmetic."""
    a, b, c = _v("a"), _v("b"), _v("c")
    expr = mt.Mul(lhs=mt.Add(lhs=a, rhs=b), rhs=c)
    func = _wrap(
        [mt.Bind(var=_v("r", "int32"), value=expr)],
        params=[a, b, c],
    )
    _rt(func)


def test_t2_nested_logic():
    """``a < b and c < d`` — boolean conjunction of comparisons."""
    a, b, c, d = (_v(n) for n in "abcd")
    expr = mt.And(
        lhs=mt.Lt(lhs=a, rhs=b),
        rhs=mt.Lt(lhs=c, rhs=d),
    )
    func = _wrap(
        [mt.Bind(var=_v("r", "bool"), value=expr)],
        params=[a, b, c, d],
    )
    text = pyast.to_python(func)
    assert "a < b" in text
    assert "c < d" in text
    _rt(func)


def test_t2_or_chain():
    """``i == 0 or j == 10 or k == 20`` — three-way disjunction."""
    i, j, k = (_v(n) for n in "ijk")
    expr = mt.Or(
        lhs=mt.Eq(lhs=i, rhs=_int(0)),
        rhs=mt.Or(
            lhs=mt.Eq(lhs=j, rhs=_int(10)),
            rhs=mt.Eq(lhs=k, rhs=_int(20)),
        ),
    )
    func = _wrap(
        [mt.Bind(var=_v("r", "bool"), value=expr)],
        params=[i, j, k],
    )
    _rt(func)


def test_t2_not_on_binop():
    """``not (a < b)`` — unary negation of a comparison."""
    a, b = _v("a"), _v("b")
    expr = mt.Not(a=mt.Lt(lhs=a, rhs=b))
    func = _wrap(
        [mt.Bind(var=_v("r", "bool"), value=expr)],
        params=[a, b],
    )
    _rt(func)


def test_t2_buffer_load_with_arith_index():
    """``A[i + 1]`` — index expression contains arithmetic."""
    A = _v("A", "int32")
    i = _v("i")
    expr = mt.BufferLoad(source=A, indices=[mt.Add(lhs=i, rhs=_int(1))])
    func = _wrap(
        [mt.Bind(var=_v("c", "int32"), value=expr)],
        params=[A, i],
    )
    _rt(func)


def test_t2_call_with_arith_args():
    """``T.foo(a + b, c * 2)`` — call args contain arithmetic."""
    a, b, c = _v("a"), _v("b"), _v("c")
    expr = mt.Call(
        op_name="T.foo",
        args=[mt.Add(lhs=a, rhs=b), mt.Mul(lhs=c, rhs=_int(2))],
    )
    func = _wrap(
        [mt.Bind(var=_v("r"), value=expr)],
        params=[a, b, c],
    )
    _rt(func)


# ============================================================================
# Tier 3 — Single statements
# ============================================================================


@pytest.mark.parametrize("dtype", PRIM_INT_DTYPES + PRIM_FLOAT_DTYPES)
def test_t3_bind_each_dtype(dtype):
    """Bind a fresh Var of each dtype from a value of that dtype."""
    if dtype in PRIM_INT_DTYPES:
        value: Any = _int(0, dtype) if dtype != "bool" else _int(0, "bool")
    else:
        value = _float(0.0, dtype)
    func = _wrap([mt.Bind(var=_v("x", dtype), value=value)])
    _rt(func)


def test_t3_bind_from_var():
    """``b = a`` — bind from another var."""
    a = _v("a")
    func = _wrap(
        [mt.Bind(var=_v("b", "int32"), value=a)],
        params=[a],
    )
    _rt(func)


def test_t3_bind_from_load():
    """``b = A[0]`` — bind from a BufferLoad."""
    A = _v("A", "float32")
    func = _wrap(
        [mt.Bind(var=_v("b", "float32"),
                 value=mt.BufferLoad(source=A, indices=[_int(0)]))],
        params=[A],
    )
    _rt(func)


def test_t3_bind_from_binop_chain():
    """``r = (a + b) * (c - d)`` — bind from nested arith."""
    a, b, c, d = (_v(n) for n in "abcd")
    func = _wrap(
        [mt.Bind(
            var=_v("r", "int32"),
            value=mt.Mul(
                lhs=mt.Add(lhs=a, rhs=b),
                rhs=mt.Sub(lhs=c, rhs=d),
            ),
        )],
        params=[a, b, c, d],
    )
    _rt(func)


def test_t3_buffer_store_literal():
    """``A[0] = 1`` — store a literal at a literal index."""
    A = _v("A", "int32")
    func = _wrap(
        [mt.BufferStore(buffer=A, value=_int(1), indices=[_int(0)])],
        params=[A],
    )
    _rt(func)


def test_t3_buffer_store_var_index_var_value():
    """``A[i] = x`` — store at a var index, var value."""
    A = _v("A", "int32")
    i = _v("i")
    x = _v("x")
    func = _wrap(
        [mt.BufferStore(buffer=A, value=x, indices=[i])],
        params=[A, i, x],
    )
    _rt(func)


def test_t3_buffer_store_from_load():
    """``A[0] = B[0]`` — copy one buffer slot to another."""
    A, B = _v("A", "int32"), _v("B", "int32")
    func = _wrap(
        [mt.BufferStore(
            buffer=A,
            value=mt.BufferLoad(source=B, indices=[_int(0)]),
            indices=[_int(0)],
        )],
        params=[A, B],
    )
    _rt(func)


def test_t3_buffer_store_with_arith_value():
    """``A[i] = a + b`` — store a binop result."""
    A = _v("A", "int32")
    a, b, i = _v("a"), _v("b"), _v("i")
    func = _wrap(
        [mt.BufferStore(buffer=A, value=mt.Add(lhs=a, rhs=b), indices=[i])],
        params=[A, a, b, i],
    )
    _rt(func)


def test_t3_assert_no_message():
    """``assert a`` — AssertStmt without message."""
    a = _v("a", "bool")
    func = _wrap([mt.AssertStmt(cond=a, message=None)], params=[a])
    text = pyast.to_python(func)
    assert "assert a" in text
    _rt(func)


def test_t3_assert_with_message():
    """``assert a, "msg"`` — AssertStmt with literal message."""
    a = _v("a", "bool")
    func = _wrap(
        [mt.AssertStmt(cond=a, message="must hold")],
        params=[a],
    )
    text = pyast.to_python(func)
    assert 'assert a, "must hold"' in text
    _rt(func)


def test_t3_evaluate_var():
    """``T.evaluate(a)`` — Evaluate of a Var (non-call value)."""
    a = _v("a")
    func = _wrap([mt.Evaluate(value=a)], params=[a])
    text = pyast.to_python(func)
    assert "T.evaluate(a)" in text
    _rt(func)


def test_t3_evaluate_literal():
    """``T.evaluate(0)`` — Evaluate of a literal (Ex-style)."""
    func = _wrap([mt.Evaluate(value=_int(0))])
    text = pyast.to_python(func)
    assert "T.evaluate(0)" in text
    _rt(func)


def test_t3_evaluate_non_ret_call():
    """``T.evaluate(T.mma(a, b))`` — Evaluate wraps non-ret calls."""
    a, b = _v("a"), _v("b")
    func = _wrap(
        [mt.Evaluate(value=mt.Call(op_name="T.mma", args=[a, b]))],
        params=[a, b],
    )
    text = pyast.to_python(func)
    assert "T.evaluate(T.mma(a, b))" in text
    _rt(func)


def test_t3_evaluate_ret_call_inverts_to_return():
    """``Evaluate(Call(ret, [x]))`` → ``return x`` via return_check."""
    x = _v("x")
    func = _wrap(
        [mt.Evaluate(value=mt.Call(op_name="ret", args=[x]))],
        params=[x],
    )
    text = pyast.to_python(func)
    assert "return x" in text
    _rt(func)


def test_t3_evaluate_multi_arg_ret_does_not_invert():
    """``Evaluate(Call(ret, [a, b]))`` keeps the wrap (only single-arg ret inverts)."""
    a, b = _v("a"), _v("b")
    func = _wrap(
        [mt.Evaluate(value=mt.Call(op_name="ret", args=[a, b]))],
        params=[a, b],
    )
    text = pyast.to_python(func)
    assert "T.evaluate(" in text
    assert "return" not in text
    _rt(func)


# ============================================================================
# Tier 4 — Control flow (single level)
# ============================================================================


def test_t4_if_both_branches():
    """``if a: ... else: ...`` — IfThenElse with both regions populated."""
    a = _v("a", "bool")
    A = _v("A", "int32")
    func = _wrap(
        [mt.IfThenElse(
            cond=a,
            then_body=[mt.BufferStore(buffer=A, value=_int(1), indices=[_int(0)])],
            else_body=[mt.BufferStore(buffer=A, value=_int(2), indices=[_int(0)])],
        )],
        params=[a, A],
    )
    text = pyast.to_python(func)
    assert "if a:" in text
    assert "else:" in text
    _rt(func)


def test_t4_if_then_only():
    """``if a: ...`` — IfThenElse with empty else body."""
    a = _v("a", "bool")
    A = _v("A", "int32")
    func = _wrap(
        [mt.IfThenElse(
            cond=a,
            then_body=[mt.BufferStore(buffer=A, value=_int(1), indices=[_int(0)])],
            else_body=[],
        )],
        params=[a, A],
    )
    text = pyast.to_python(func)
    assert "if a:" in text
    _rt(func)


def test_t4_if_with_compound_cond():
    """``if (a < b) and (c < d):`` — cond is a compound boolean."""
    a, b, c, d = (_v(n) for n in "abcd")
    A = _v("A", "int32")
    cond = mt.And(lhs=mt.Lt(lhs=a, rhs=b), rhs=mt.Lt(lhs=c, rhs=d))
    func = _wrap(
        [mt.IfThenElse(
            cond=cond,
            then_body=[mt.BufferStore(buffer=A, value=_int(1), indices=[_int(0)])],
            else_body=[],
        )],
        params=[a, b, c, d, A],
    )
    _rt(func)


def test_t4_while_simple():
    """``while a: body``."""
    a = _v("a", "bool")
    A = _v("A", "int32")
    func = _wrap(
        [mt.While(
            cond=a,
            body=[mt.BufferStore(buffer=A, value=_int(0), indices=[_int(0)])],
        )],
        params=[a, A],
    )
    text = pyast.to_python(func)
    assert "while a:" in text
    _rt(func)


@pytest.mark.parametrize("kind", FOR_KINDS)
def test_t4_for_each_kind(kind):
    """``for i in T.<kind>(0, 10): A[i] = 1`` — each loop kind."""
    A = _v("A", "int32")
    i = _v("i")
    func = _wrap(
        [mt.For(
            loop_var=i, start=0, end=10, step=1,
            body=[mt.BufferStore(buffer=A, value=_int(1), indices=[i])],
            kind=kind,
        )],
        params=[A],
    )
    text = pyast.to_python(func)
    assert f"T.{kind}(" in text or "range(" in text
    _rt(func)


def test_t4_for_with_non_default_step():
    """``for i in T.serial(0, 10, step=2): ...`` — non-default step."""
    A = _v("A", "int32")
    i = _v("i")
    func = _wrap(
        [mt.For(
            loop_var=i, start=0, end=10, step=2,
            body=[mt.BufferStore(buffer=A, value=_int(1), indices=[i])],
            kind="serial",
        )],
        params=[A],
    )
    _rt(func)


def test_t4_for_with_annotations():
    """For with non-empty annotations dict."""
    A = _v("A", "int32")
    i = _v("i")
    func = _wrap(
        [mt.For(
            loop_var=i, start=0, end=10, step=1,
            body=[mt.BufferStore(buffer=A, value=_int(1), indices=[i])],
            kind="parallel",
            annotations={"unroll_factor": 4},
        )],
        params=[A],
    )
    text = pyast.to_python(func)
    assert "annotations=" in text
    _rt(func)


def test_t4_block_simple():
    """``with T.block(): body``."""
    func = _wrap(
        [mt.Block(body=[mt.Bind(var=_v("x", "int32"), value=_int(1))])],
    )
    text = pyast.to_python(func)
    assert "with T.block():" in text
    _rt(func)


@pytest.mark.skip(
    reason=(
        "SeqStmt uses WithTraits(text_printer_no_frame=True) which "
        "deliberately emits body stmts transparently — no enclosing "
        "syntax survives in the printed text. This makes SeqStmt-vs-"
        "bare-body fundamentally indistinguishable on parse, i.e. the "
        "roundtrip is inherently lossy for this one IR shape. "
        "Document-and-skip rather than force a degenerate inverse."
    ),
)
def test_t4_seq_stmt_transparent():
    """SeqStmt — transparent container, body stmts share parent scope."""
    func = _wrap(
        [mt.SeqStmt(stmts=[
            mt.Bind(var=_v("a", "int32"), value=_int(1)),
            mt.Bind(var=_v("b", "int32"), value=_int(2)),
        ])],
    )
    _rt(func)


# ============================================================================
# Tier 5 — Nested control flow
# ============================================================================


def test_t5_if_in_if():
    """``if a: if b: ...`` — depth-2 nested if."""
    a, b = _v("a", "bool"), _v("b", "bool")
    A = _v("A", "int32")
    func = _wrap(
        [mt.IfThenElse(
            cond=a,
            then_body=[mt.IfThenElse(
                cond=b,
                then_body=[mt.BufferStore(buffer=A, value=_int(1), indices=[_int(0)])],
                else_body=[],
            )],
            else_body=[],
        )],
        params=[a, b, A],
    )
    _rt(func)


def test_t5_if_in_if_in_if():
    """``if a: if b: if c: ...`` — depth-3 nested if."""
    a, b, c = _v("a", "bool"), _v("b", "bool"), _v("c", "bool")
    A = _v("A", "int32")
    inner = mt.IfThenElse(
        cond=c,
        then_body=[mt.BufferStore(buffer=A, value=_int(1), indices=[_int(0)])],
        else_body=[],
    )
    middle = mt.IfThenElse(cond=b, then_body=[inner], else_body=[])
    outer = mt.IfThenElse(cond=a, then_body=[middle], else_body=[])
    func = _wrap([outer], params=[a, b, c, A])
    _rt(func)


def test_t5_for_in_for():
    """Nested for-loops — ``for i: for j: A[i] = j``."""
    A = _v("A", "int32")
    i, j = _v("i"), _v("j")
    inner = mt.For(
        loop_var=j, start=0, end=10, step=1,
        body=[mt.BufferStore(buffer=A, value=j, indices=[i])],
        kind="serial",
    )
    outer = mt.For(
        loop_var=i, start=0, end=10, step=1,
        body=[inner], kind="serial",
    )
    func = _wrap([outer], params=[A])
    _rt(func)


def test_t5_if_in_for():
    """``for i in ...: if A[i] == 0: A[i] = 1`` — guarded store inside loop."""
    A = _v("A", "int32")
    i = _v("i")
    cond = mt.Eq(lhs=mt.BufferLoad(source=A, indices=[i]), rhs=_int(0))
    if_stmt = mt.IfThenElse(
        cond=cond,
        then_body=[mt.BufferStore(buffer=A, value=_int(1), indices=[i])],
        else_body=[],
    )
    func = _wrap(
        [mt.For(
            loop_var=i, start=0, end=10, step=1,
            body=[if_stmt], kind="serial",
        )],
        params=[A],
    )
    _rt(func)


def test_t5_while_in_if():
    """``if a: while cond: body`` — while inside if."""
    a, cond = _v("a", "bool"), _v("cond", "bool")
    A = _v("A", "int32")
    func = _wrap(
        [mt.IfThenElse(
            cond=a,
            then_body=[mt.While(
                cond=cond,
                body=[mt.BufferStore(buffer=A, value=_int(1), indices=[_int(0)])],
            )],
            else_body=[],
        )],
        params=[a, cond, A],
    )
    _rt(func)


def test_t5_block_in_for():
    """``for i: with T.block(): A[i] = 1`` — block inside loop."""
    A = _v("A", "int32")
    i = _v("i")
    func = _wrap(
        [mt.For(
            loop_var=i, start=0, end=10, step=1,
            body=[mt.Block(body=[
                mt.BufferStore(buffer=A, value=_int(1), indices=[i]),
            ])],
            kind="serial",
        )],
        params=[A],
    )
    _rt(func)


def test_t5_for_in_while_in_if():
    """3-level nesting: ``if a: while cond: for i: ...``."""
    a, cond = _v("a", "bool"), _v("cond", "bool")
    A = _v("A", "int32")
    i = _v("i")
    func = _wrap(
        [mt.IfThenElse(
            cond=a,
            then_body=[mt.While(
                cond=cond,
                body=[mt.For(
                    loop_var=i, start=0, end=10, step=1,
                    body=[mt.BufferStore(buffer=A, value=_int(1), indices=[i])],
                    kind="serial",
                )],
            )],
            else_body=[],
        )],
        params=[a, cond, A],
    )
    _rt(func)


def test_t5_deep_block_nesting():
    """``with T.block(): with T.block(): with T.block(): body`` — 3-level blocks."""
    func = _wrap(
        [mt.Block(body=[
            mt.Block(body=[
                mt.Block(body=[
                    mt.Bind(var=_v("x", "int32"), value=_int(1)),
                ]),
            ]),
        ])],
    )
    _rt(func)


# ============================================================================
# Tier 6 — Functions
# ============================================================================


def test_t6_primfunc_no_params_single_stmt():
    """Smallest possible PrimFunc."""
    func = _wrap([mt.Bind(var=_v("x", "int32"), value=_int(0))])
    text = pyast.to_python(func)
    assert "@T.prim_func" in text
    assert "def test_func():" in text
    _rt(func)


def test_t6_primfunc_no_params_no_body():
    """PrimFunc with empty body."""
    func = _wrap([])
    text = pyast.to_python(func)
    assert "def test_func():" in text
    _rt(func)


@pytest.mark.parametrize("dtype", PRIM_DTYPES)
def test_t6_primfunc_single_param_each_dtype(dtype):
    """PrimFunc with one param of each dtype, used in body."""
    a = _v("a", dtype)
    func = _wrap(
        [mt.Bind(var=_v("x", dtype), value=a)],
        params=[a],
    )
    text = pyast.to_python(func)
    assert f"a: T.{dtype}" in text
    _rt(func)


def test_t6_primfunc_many_params():
    """PrimFunc with 6 mixed-dtype params, all used in body."""
    a = _v("a", "int32")
    b = _v("b", "int64")
    c = _v("c", "float32")
    d = _v("d", "float64")
    e = _v("e", "bool")
    f = _v("f", "uint8")
    func = _wrap(
        [
            mt.Bind(var=_v("x1", "int32"), value=a),
            mt.Bind(var=_v("x2", "int64"), value=b),
            mt.Bind(var=_v("x3", "float32"), value=c),
            mt.Bind(var=_v("x4", "float64"), value=d),
            mt.Bind(var=_v("x5", "bool"), value=e),
            mt.Bind(var=_v("x6", "uint8"), value=f),
        ],
        params=[a, b, c, d, e, f],
    )
    _rt(func)


def test_t6_primfunc_long_body():
    """PrimFunc with 10 sequential stmts (tests stmt-ordering & scoping)."""
    A = _v("A", "int32")
    body = [
        mt.BufferStore(buffer=A, value=_int(i), indices=[_int(i)])
        for i in range(10)
    ]
    func = _wrap(body, params=[A])
    _rt(func)


def test_t6_primfunc_deep_nesting():
    """PrimFunc whose body is deeply nested (4 levels)."""
    a, b = _v("a", "bool"), _v("b", "bool")
    A = _v("A", "int32")
    i, j = _v("i"), _v("j")
    deep = mt.IfThenElse(
        cond=a,
        then_body=[mt.IfThenElse(
            cond=b,
            then_body=[mt.For(
                loop_var=i, start=0, end=10, step=1,
                body=[mt.For(
                    loop_var=j, start=0, end=10, step=1,
                    body=[mt.BufferStore(buffer=A, value=_int(1), indices=[i])],
                    kind="serial",
                )],
                kind="serial",
            )],
            else_body=[],
        )],
        else_body=[],
    )
    func = _wrap([deep], params=[a, b, A])
    _rt(func)


def test_t6_primfunc_with_assert_then_compute():
    """``def f(...): assert cond, msg; ...`` — common entry-checking pattern."""
    cond = _v("cond", "bool")
    A = _v("A", "int32")
    func = _wrap(
        [
            mt.AssertStmt(cond=cond, message="precondition"),
            mt.BufferStore(buffer=A, value=_int(0), indices=[_int(0)]),
            # Reuse the same ``A`` Var reference — a fresh ``_v("A")``
            # here would print as a distinct symbol (``A_1``) that the
            # parser can't resolve, shadowing the param.
            mt.Evaluate(value=mt.Call(op_name="ret", args=[A])),
        ],
        params=[cond, A],
    )
    _rt(func)


# ============================================================================
# Tier 7 — Modules
# ============================================================================


def test_t7_irmodule_one_func():
    """IRModule containing a single PrimFunc."""
    func = mt.PrimFunc(
        name="f",
        params=[],
        body=[mt.Bind(var=_v("x", "int32"), value=_int(0))],
    )
    mod = mt.IRModule(name="Module", funcs=[func])
    text = pyast.to_python(mod)
    assert "@I.ir_module" in text
    assert "class Module:" in text
    assert "def f():" in text
    _rt(mod)


def test_t7_irmodule_multiple_funcs():
    """IRModule containing two PrimFuncs."""
    f1 = mt.PrimFunc(
        name="f1", params=[],
        body=[mt.Bind(var=_v("x", "int32"), value=_int(1))],
    )
    f2 = mt.PrimFunc(
        name="f2", params=[],
        body=[mt.Bind(var=_v("y", "int32"), value=_int(2))],
    )
    mod = mt.IRModule(name="Module", funcs=[f1, f2])
    text = pyast.to_python(mod)
    assert "def f1():" in text
    assert "def f2():" in text
    _rt(mod)


def test_t7_irmodule_func_with_params():
    """IRModule whose function has parameters."""
    a = _v("a", "int32")
    inner = mt.PrimFunc(
        name="add_one", params=[a],
        body=[mt.Bind(var=_v("r", "int32"), value=mt.Add(lhs=a, rhs=_int(1)))],
    )
    mod = mt.IRModule(name="Module", funcs=[inner])
    _rt(mod)


# ============================================================================
# Tier 8 — Realistic mini-TIR programs (end-to-end)
# ============================================================================


def test_t8_buffer_init_loop():
    """Initialize a buffer with a constant: ``for i: A[i] = 0``."""
    A = _v("A", "int32")
    i = _v("i")
    func = _wrap(
        [mt.For(
            loop_var=i, start=0, end=128, step=1,
            body=[mt.BufferStore(buffer=A, value=_int(0), indices=[i])],
            kind="serial",
        )],
        params=[A],
    )
    _rt(func)


def test_t8_vector_add_pattern():
    """``for i: C[i] = A[i] + B[i]`` — element-wise add."""
    A, B, C = _v("A", "float32"), _v("B", "float32"), _v("C", "float32")
    i = _v("i")
    func = _wrap(
        [mt.For(
            loop_var=i, start=0, end=64, step=1,
            body=[mt.BufferStore(
                buffer=C,
                value=mt.Add(
                    lhs=mt.BufferLoad(source=A, indices=[i]),
                    rhs=mt.BufferLoad(source=B, indices=[i]),
                ),
                indices=[i],
            )],
            kind="parallel",
        )],
        params=[A, B, C],
    )
    _rt(func)


def test_t8_predicated_compute():
    """``for i: if cond: A[i] = compute else: A[i] = 0`` — branch in loop."""
    cond = _v("cond", "bool")
    A = _v("A", "int32")
    i = _v("i")
    func = _wrap(
        [mt.For(
            loop_var=i, start=0, end=32, step=1,
            body=[mt.IfThenElse(
                cond=cond,
                then_body=[mt.BufferStore(
                    buffer=A,
                    value=mt.Mul(lhs=i, rhs=_int(2)),
                    indices=[i],
                )],
                else_body=[mt.BufferStore(buffer=A, value=_int(0), indices=[i])],
            )],
            kind="serial",
        )],
        params=[cond, A],
    )
    _rt(func)


def test_t8_nested_loop_2d_init():
    """2D buffer init: ``for i: for j: A[i, j] = i + j``."""
    A = _v("A", "int32")
    i, j = _v("i"), _v("j")
    func = _wrap(
        [mt.For(
            loop_var=i, start=0, end=8, step=1,
            body=[mt.For(
                loop_var=j, start=0, end=8, step=1,
                body=[mt.BufferStore(
                    buffer=A,
                    value=mt.Add(lhs=i, rhs=j),
                    indices=[i, j],
                )],
                kind="serial",
            )],
            kind="serial",
        )],
        params=[A],
    )
    _rt(func)


def test_t8_assert_guarded_loop():
    """``assert cond, msg; for i: A[i] = i`` — assert before loop."""
    cond = _v("cond", "bool")
    A = _v("A", "int32")
    i = _v("i")
    func = _wrap(
        [
            mt.AssertStmt(cond=cond, message="bounds check"),
            mt.For(
                loop_var=i, start=0, end=16, step=1,
                body=[mt.BufferStore(buffer=A, value=i, indices=[i])],
                kind="serial",
            ),
        ],
        params=[cond, A],
    )
    _rt(func)


def test_t8_module_with_two_kernels():
    """IRModule with two PrimFuncs (init + compute).

    Each PrimFunc owns its own ``A`` / ``i`` Var identities (don't share
    across functions — each ``def f(A):`` introduces a fresh def-site,
    and sharing Python objects across funcs creates a cross-function
    alias the parser can't faithfully reconstruct).
    """
    A_init = _v("A", "int32")
    i_init = _v("i")
    init = mt.PrimFunc(
        name="init",
        params=[A_init],
        body=[mt.For(
            loop_var=i_init, start=0, end=16, step=1,
            body=[mt.BufferStore(buffer=A_init, value=_int(0), indices=[i_init])],
            kind="serial",
        )],
    )
    A_compute = _v("A", "int32")
    i_compute = _v("i")
    compute = mt.PrimFunc(
        name="compute",
        params=[A_compute],
        body=[mt.For(
            loop_var=i_compute, start=0, end=16, step=1,
            body=[mt.BufferStore(
                buffer=A_compute,
                value=mt.Add(
                    lhs=mt.BufferLoad(source=A_compute, indices=[i_compute]),
                    rhs=_int(1),
                ),
                indices=[i_compute],
            )],
            kind="parallel",
        )],
    )
    mod = mt.IRModule(name="Kernel", funcs=[init, compute])
    _rt(mod)


def test_t8_complex_boolean_guard():
    """``if (i < n) and (j < m) and not(k == 0): ...`` — multi-clause guard."""
    i, j, k, n, m = (_v(x) for x in ["i", "j", "k", "n", "m"])
    A = _v("A", "int32")
    cond = mt.And(
        lhs=mt.And(
            lhs=mt.Lt(lhs=i, rhs=n),
            rhs=mt.Lt(lhs=j, rhs=m),
        ),
        rhs=mt.Not(a=mt.Eq(lhs=k, rhs=_int(0))),
    )
    func = _wrap(
        [mt.IfThenElse(
            cond=cond,
            then_body=[mt.BufferStore(buffer=A, value=_int(1), indices=[i, j])],
            else_body=[],
        )],
        params=[i, j, k, n, m, A],
    )
    _rt(func)
