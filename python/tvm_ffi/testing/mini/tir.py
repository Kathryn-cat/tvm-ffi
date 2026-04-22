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
"""Mini-TIR — TIR-flavored dialect, auto-registered.

Refactored to use :func:`~tvm_ffi.dialect_autogen.finalize_module`
(see ``design_docs/parser_auto_registration.md``). The Python module
itself IS the ``T`` dialect — ``import tvm_ffi.testing.mini.tir as T``
then ``T.Add`` / ``T.prim_func`` / ``T.int32`` all resolve via
attribute lookup, with every factory / hook auto-injected at module
load time by ``finalize_module``.

The file splits cleanly into three buckets (per §3 of the design doc):

* **Bucket A — IR class declarations** (most of this file). Each IR
  class carries ``__ffi_ir_traits__`` describing its printer/parser
  behavior. Unchanged from the pre-refactor layout.
* **Bucket C — per-IR semantics** that can't be derived from traits
  alone: the ``_DtypeHandle`` dual-mode callable, the ``_IterHolder``
  dataclass shape, the ``$global:`` sugar-check / return-check /
  for-kind-prefix resolvers, and the explicit ``ret`` hook that builds
  the ``Evaluate(Call("ret", [v]))`` TIR-specific return shape. Kept
  as explicit user code; ``finalize_module`` respects the ``hasattr``
  check and won't overwrite anything the user defines.
* **Bucket B — mechanical wiring** (factories, hooks, op-class maps,
  dtype handle mounts, default-ty attrs). Eliminated — produced by
  one ``finalize_module(__name__, ...)`` call at the bottom of the
  file.
"""

# ruff: noqa: A003, D102, N802, UP006, UP045

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Union  # noqa: UP035

from tvm_ffi import Object, pyast, register_global_func
from tvm_ffi import dtype as ffi_dtype
from tvm_ffi import ir_traits as tr
from tvm_ffi.dataclasses import py_class
from tvm_ffi.dataclasses import field as dc_field
from tvm_ffi.dialect_autogen import finalize_module


# ============================================================================
# Types
# ============================================================================


@py_class("mini.tir.PrimTy", structural_eq="dag")
class PrimTy(Object):
    """Scalar primitive type — prints as ``T.<dtype>``."""

    __ffi_ir_traits__ = tr.PrimTyTraits("$field:dtype")
    dtype: str


@py_class("mini.tir.BufferTy", structural_eq="dag")
class BufferTy(Object):
    """Buffer type — ``T.Buffer((shape...), dtype, ...)`` with default elision."""

    __ffi_ir_traits__ = tr.BufferTyTraits(
        "$field:shape",
        "$field:dtype",
        "$field:strides",
        "$field:offset",
        "$field:scope",
    )
    shape: Any
    dtype: str
    strides: Optional[List[int]] = None
    offset: Optional[int] = None
    scope: Optional[str] = None


# ============================================================================
# Expressions
# ============================================================================


@py_class("mini.tir.Var", structural_eq="var")
class Var(Object):
    """Typed scalar — ``ValueTraits`` (use site = name; def site = name: ty)."""

    __ffi_ir_traits__ = tr.ValueTraits("$field:name", "$field:ty", None)
    name: str = dc_field(structural_eq="ignore")
    ty: PrimTy


@py_class("mini.tir.IntImm", structural_eq="dag")
class IntImm(Object):
    """Integer literal — ``LiteralTraits(format="int")``."""

    __ffi_ir_traits__ = tr.LiteralTraits("$field:value", "int")
    value: int
    dtype: Any


@py_class("mini.tir.FloatImm", structural_eq="dag")
class FloatImm(Object):
    """Float literal — ``LiteralTraits(format="float")``."""

    __ffi_ir_traits__ = tr.LiteralTraits("$field:value", "float")
    value: float
    dtype: Any


# ----------------------------------------------------------------------------
# Bucket C — sugar-check / return-check / kind-prefix policies.
# Encoded as registered globals so the trait can reference them by name.
# ----------------------------------------------------------------------------


def _no_const_fold(lhs: Any, rhs: Any) -> bool:
    """Sugar gate: refuse infix if both operands are literals."""
    return not (
        isinstance(lhs, (IntImm, FloatImm)) and isinstance(rhs, (IntImm, FloatImm))
    )


@register_global_func("mini.tir._binop_sugar_check")
def _binop_sugar_check_global(_printer: Any, obj: Any) -> bool:
    return _no_const_fold(obj.lhs, obj.rhs)


def _binop_traits(op: str, func_name: str) -> tr.BinOpTraits:
    """Trait-factory helper for the standard mini-TIR BinOps."""
    return tr.BinOpTraits(
        "$field:lhs", "$field:rhs", op,
        "$global:mini.tir._binop_sugar_check", func_name,
    )


@py_class("mini.tir.Add", structural_eq="dag")
class Add(Object):
    __ffi_ir_traits__ = _binop_traits("+", "T.Add")
    lhs: Any
    rhs: Any


@py_class("mini.tir.Sub", structural_eq="dag")
class Sub(Object):
    __ffi_ir_traits__ = _binop_traits("-", "T.Sub")
    lhs: Any
    rhs: Any


@py_class("mini.tir.Mul", structural_eq="dag")
class Mul(Object):
    __ffi_ir_traits__ = _binop_traits("*", "T.Mul")
    lhs: Any
    rhs: Any


@py_class("mini.tir.Lt", structural_eq="dag")
class Lt(Object):
    __ffi_ir_traits__ = _binop_traits("<", "T.Lt")
    lhs: Any
    rhs: Any


@py_class("mini.tir.Eq", structural_eq="dag")
class Eq(Object):
    __ffi_ir_traits__ = _binop_traits("==", "T.Eq")
    lhs: Any
    rhs: Any


@py_class("mini.tir.And", structural_eq="dag")
class And(Object):
    __ffi_ir_traits__ = _binop_traits("and", "T.And")
    lhs: Any
    rhs: Any


@py_class("mini.tir.Or", structural_eq="dag")
class Or(Object):
    __ffi_ir_traits__ = _binop_traits("or", "T.Or")
    lhs: Any
    rhs: Any


@py_class("mini.tir.Not", structural_eq="dag")
class Not(Object):
    __ffi_ir_traits__ = tr.UnaryOpTraits("$field:a", "not")
    a: Any


@py_class("mini.tir.Call", structural_eq="dag")
class Call(Object):
    """Call with literal callee: ``T.<op_name>(args...)``."""

    __ffi_ir_traits__ = tr.CallTraits(
        "$field:op_name", "$field:args",
        None, None, None, None,
    )
    op_name: str
    args: List[Any]


@py_class("mini.tir.BufferLoad", structural_eq="dag")
class BufferLoad(Object):
    """``buffer[indices]`` — ``LoadTraits``."""

    __ffi_ir_traits__ = tr.LoadTraits("$field:source", "$field:indices", None)
    source: Var
    indices: List[Any]


@py_class("mini.tir.Cast", structural_eq="dag")
class Cast(Object):
    """Level 0 fixture — no trait, prints as ``mini.tir.Cast(...)``."""

    target: PrimTy
    value: Any


@py_class("mini.tir.Flag", structural_eq="dag")
class Flag(Object):
    """Pure Tier-3 leaf fixture — no trait, no ``__ffi_text_parse__``.

    Tests validate that the default-parse path round-trips a bare
    ``@py_class`` leaf via reflection only (no wiring in the dialect
    module beyond the registration that :func:`finalize_module` adds
    for every IR class).
    """

    kind: str
    count: int


@py_class("mini.tir.FlagV2", structural_eq="dag")
class FlagV2(Object):
    """Tier-1 fixture — opts into ``__ffi_text_parse__`` at class body.

    The method is declared explicitly as the design-doc's canonical
    Tier-1 escape hatch: it builds ``FlagV2`` from the printed call's
    first positional arg (``FlagV2("X")`` form) OR from its ``kind``
    keyword (``FlagV2(kind="X")`` — the default-printer shape). Tests
    assert the custom parser fires in preference to Tier-3 default.
    A sentinel attribute (``_FLAG_V2_CUSTOM_FIRED``) on the class lets
    tests observe that this method ran.
    """

    kind: str

    @classmethod
    def __ffi_text_parse__(cls, parser: Any, node: Any) -> "FlagV2":
        cls._FLAG_V2_CUSTOM_FIRED = True
        if node.args:
            return cls(kind=parser.eval_expr(node.args[0]))
        kwargs = {
            k: parser.eval_expr(v)
            for k, v in zip(node.kwargs_keys, node.kwargs_values)
        }
        return cls(**kwargs)


# ============================================================================
# Statements
# ============================================================================


@py_class("mini.tir.Bind", structural_eq="tree")
class Bind(Object):
    """Local binding ``var: ty = value`` — ``AssignTraits``."""

    __ffi_ir_traits__ = tr.AssignTraits(
        "$field:var", "$field:value", None, None, None, None,
    )
    value: Any
    var: Var = dc_field(structural_eq="def")


@py_class("mini.tir.BufferStore", structural_eq="tree")
class BufferStore(Object):
    """``buffer[indices] = value`` — ``StoreTraits``."""

    __ffi_ir_traits__ = tr.StoreTraits(
        "$field:buffer", "$field:value", "$field:indices", None,
    )
    buffer: Var
    value: Any
    indices: List[Any]


# ----------------------------------------------------------------------------
# Evaluate — expr-stmt AssignTraits with dynamic kind + return-check.
# The three ``$global:`` refs encode mini-TIR's convention that a
# ``Call(op_name="ret", args=[x])`` wrapped in Evaluate prints as
# ``return x`` (via text_printer_return_check).
# ----------------------------------------------------------------------------


def _evaluate_is_ret_call(obj: Any) -> bool:
    return isinstance(obj.value, Call) and obj.value.op_name == "ret" and len(obj.value.args) == 1


@register_global_func("mini.tir._evaluate_expr")
def _evaluate_expr_global(_printer: Any, obj: Any) -> Any:
    if _evaluate_is_ret_call(obj):
        return obj.value.args[0]
    return obj.value


@register_global_func("mini.tir._evaluate_kind")
def _evaluate_kind_global(_printer: Any, obj: Any) -> Any:
    if _evaluate_is_ret_call(obj):
        return None
    return "T.evaluate"


@register_global_func("mini.tir._evaluate_is_return")
def _evaluate_is_return_global(_printer: Any, obj: Any) -> bool:
    return _evaluate_is_ret_call(obj)


@py_class("mini.tir.Evaluate", structural_eq="tree")
class Evaluate(Object):
    """Expression-statement — ``AssignTraits`` with dynamic kind + return check."""

    __ffi_ir_traits__ = tr.AssignTraits(
        None,
        "$global:mini.tir._evaluate_expr",
        None, None,
        "$global:mini.tir._evaluate_kind",
        "$global:mini.tir._evaluate_is_return",
    )
    value: Any


@py_class("mini.tir.IfThenElse", structural_eq="tree")
class IfThenElse(Object):
    __ffi_ir_traits__ = tr.IfTraits(
        "$field:cond",
        tr.RegionTraits("$field:then_body", None, None, None),
        tr.RegionTraits("$field:else_body", None, None, None),
    )
    cond: Any
    then_body: List[Any]
    else_body: List[Any] = dc_field(default_factory=list)


@py_class("mini.tir.While", structural_eq="tree")
class While(Object):
    __ffi_ir_traits__ = tr.WhileTraits(
        "$field:cond",
        tr.RegionTraits("$field:body", None, None, None),
    )
    cond: Any
    body: List[Any]


@py_class("mini.tir.AssertStmt", structural_eq="tree")
class AssertStmt(Object):
    __ffi_ir_traits__ = tr.AssertTraits("$field:cond", "$field:message")
    cond: Any
    message: Optional[str] = None


# NOTE: no dedicated ReturnStmt — ``return x`` is encoded as
# ``Evaluate(Call(op_name="ret", args=[x]))``. See the ``ret`` override
# below: auto-wiring produces a default ``ret`` hook, but mini-TIR
# needs the special Evaluate-wrapping shape, so the user-defined
# ``ret`` takes precedence.


# ----------------------------------------------------------------------------
# For — kind resolved via a global. Kinds themselves live in a
# class-level dtype and are looked up at print time.
# ----------------------------------------------------------------------------


@register_global_func("mini.tir._for_kind_prefix")
def _for_kind_prefix_global(_printer: Any, obj: Any) -> str:
    return f"T.{obj.kind}"


@py_class("mini.tir.For", structural_eq="tree")
class For(Object):
    __ffi_ir_traits__ = tr.ForTraits(
        tr.RegionTraits("$field:body", "$field:loop_var", None, None),
        "$field:start", "$field:end", "$field:step",
        None, None,
        "$field:annotations",
        "$global:mini.tir._for_kind_prefix",
    )
    loop_var: Var = dc_field(structural_eq="def")
    start: Any
    end: Any
    step: Any
    body: List[Any]
    kind: str = "serial"
    annotations: Optional[Any] = None


@py_class("mini.tir.Block", structural_eq="tree")
class Block(Object):
    """``with T.block(): body`` — ``WithTraits`` with literal kind."""

    __ffi_ir_traits__ = tr.WithTraits(
        tr.RegionTraits("$field:body", None, None, None),
        None, None,
        "T.block",
        None, None, None,
    )
    body: List[Any]


@py_class("mini.tir.SeqStmt", structural_eq="tree")
class SeqStmt(Object):
    """Inline sequence — ``WithTraits`` with ``text_printer_no_frame=True``.

    ``no_frame=True`` makes this transparent on the printer side (body
    stmts emitted directly without enclosing syntax) — :func:`finalize_module`
    correctly skips auto-wiring since SeqStmt has no roundtrippable
    context-manager form.
    """

    __ffi_ir_traits__ = tr.WithTraits(
        tr.RegionTraits("$field:stmts", None, None, None),
        None, None, None, None, None, True,
    )
    stmts: List[Any]


# ============================================================================
# Function
# ============================================================================


@py_class("mini.tir.PrimFunc", structural_eq="tree")
class PrimFunc(Object):
    """``@T.prim_func\\ndef name(params): body``."""

    __ffi_ir_traits__ = tr.FuncTraits(
        "$field:name",
        tr.RegionTraits("$field:body", "$field:params", None, None),
        None, "T.prim_func", None,
    )
    name: str = dc_field(structural_eq="ignore")
    params: List[Var] = dc_field(structural_eq="def")
    body: List[Any]


# ============================================================================
# Bucket C — iter-holder / with-marker dataclasses
#
# ``finalize_module`` injects ``__ffi_for_handler__`` / ``__ffi_with_handler__``
# onto these classes during auto-wiring (category-C frame pushes happen
# inside the generated methods).
# ============================================================================


@dataclass
class _IterHolder:
    """Runtime value returned by ``T.serial(...)`` / ``T.parallel(...)`` etc."""

    kind: str
    start: Any
    end: Any
    step: Any
    annotations: Optional[Any] = None


@dataclass
class _BlockMarker:
    """Runtime value returned by ``T.block()``."""


# ============================================================================
# Bucket C — ``_DtypeHandle`` dual-mode callable
#
# Mini-TIR's ergonomic ``T.int32()`` (builds a Var) and ``T.int32(42)``
# (builds an IntImm) pattern requires types to be callable. :class:`PrimTy`
# alone isn't — we subclass it with ``__call__`` dispatch.
#
# This is mini-TIR *policy*, not auto-derivable; ``finalize_module``
# sees ``_DtypeHandle`` and uses it as the dtype-handle constructor for
# every name in ``dtypes=[...]``.
# ============================================================================


def _parse_type(
    primty: PrimTy,
    value: Any = None,
    var_name: Optional[str] = None,
) -> Any:
    """Dispatcher for ``T.<dtype>(...)`` calls bound via :class:`_DtypeHandle`."""
    if value is not None and var_name is not None:
        raise TypeError(
            f"T.{primty.dtype}(...): cannot pass both ``value`` and "
            f"``var_name``. Use ``value=`` for literal construction "
            f"(IntImm/FloatImm) or ``var_name=`` for Var construction.",
        )
    if value is not None:
        dt_str = str(primty.dtype)
        if dt_str in ("float16", "float32", "float64"):
            return FloatImm(value=float(value), dtype=ffi_dtype(dt_str))
        return IntImm(value=int(value), dtype=ffi_dtype(dt_str))
    name = var_name if var_name is not None else "_"
    return Var(name=name, ty=PrimTy(dtype=primty.dtype))


@py_class("mini.tir._DtypeHandle", structural_eq="dag")
class _DtypeHandle(PrimTy):
    """Callable :class:`PrimTy` subclass. ``T.int32`` / ``T.float32`` / …
    are ``_DtypeHandle`` instances — callable in two modes:

    * ``T.int32()`` or ``T.int32(var_name="x")`` → :class:`Var`
    * ``T.int32(42)`` → :class:`IntImm` (``IntImm``/``FloatImm`` based
      on the stored dtype)
    """

    def __call__(self, value: Any = None, *, var_name: Optional[str] = None) -> Any:
        return _parse_type(self, value=value, var_name=var_name)


# ============================================================================
# Bucket C — ``T.Buffer(...)`` parameterized factory
# ============================================================================


def Buffer(
    shape: Any,
    dtype: str,
    *,
    strides: Optional[Sequence[int]] = None,
    elem_offset: Optional[int] = None,
    scope: Optional[str] = None,
) -> BufferTy:
    """``T.Buffer((shape...), dtype, ...)`` factory."""
    if isinstance(shape, int):
        shape = [shape]
    return BufferTy(
        shape=list(shape),
        dtype=dtype,
        strides=list(strides) if strides is not None else None,
        offset=elem_offset,
        scope=scope,
    )


# ============================================================================
# Bucket C — ``T.ret`` override (Evaluate-wrap pattern)
#
# Auto-wiring would produce a generic ``ret`` that builds a ``ReturnTraits``
# IR — but mini-TIR has no such class. ``return x`` must become
# ``Evaluate(Call(op_name="ret", args=[x]))`` so the ``$global:_evaluate_is_return``
# check drives the printer back to ``return x`` on roundtrip.
#
# Defined BEFORE ``finalize_module`` so the ``hasattr`` guard skips
# auto-wiring of ``ret``.
# ============================================================================


def ret(_parser: Any, value: Any) -> Evaluate:
    """``return x`` → ``Evaluate(Call(op_name="ret", args=[x]))``."""
    # Wrap raw primitives via the auto-registered default-ty hooks.
    from tvm_ffi.dialect_autogen import _wrap_primitive_via_module  # noqa: PLC0415
    import sys as _sys  # noqa: PLC0415

    _mod = _sys.modules[__name__]
    args = [_wrap_primitive_via_module(value, _mod)] if value is not None else []
    return Evaluate(value=Call(op_name="ret", args=args))


# ============================================================================
# Bucket C — ``T.evaluate(x)`` explicit factory
#
# ``Evaluate`` has a dynamic ``text_printer_kind`` (``$global:``) so
# auto-wiring can't statically determine the factory name. We mount it
# explicitly: ``T.evaluate(expr)`` → :class:`Evaluate`. Primitive-wrapping
# happens via ``_wrap_primitive_via_module`` so raw-int / raw-float args
# become :class:`IntImm` / :class:`FloatImm` on the way in.
# ============================================================================


def evaluate(value: Any) -> Evaluate:  # noqa: A001
    """``T.evaluate(x)`` — wrap ``x`` in :class:`Evaluate`."""
    from tvm_ffi.dialect_autogen import _wrap_primitive_via_module  # noqa: PLC0415
    import sys as _sys  # noqa: PLC0415

    _mod = _sys.modules[__name__]
    return Evaluate(value=_wrap_primitive_via_module(value, _mod))


# ============================================================================
# Bucket C — ``imm`` construction helper (for test fixtures)
# ============================================================================


def imm(value: Union[int, float], dtype: Any) -> Any:  # noqa: UP007
    """``imm(42, "int32") -> IntImm``; ``imm(3.14, "f32") -> FloatImm``.

    Uses ``builtins.{bool,int,float}`` explicitly because
    :func:`finalize_module` mounts dtype handles for ``bool`` on the
    module — shadowing the builtin at the global-name-resolution step.
    """
    import builtins as _bi  # noqa: PLC0415

    dt = ffi_dtype(dtype) if isinstance(dtype, _bi.str) else dtype
    if isinstance(value, (_bi.bool, _bi.int)):
        return IntImm(value=_bi.int(value), dtype=dt)
    return FloatImm(value=_bi.float(value), dtype=dt)


# ============================================================================
# The single finalize_module call — auto-injects all mechanical wiring.
# ============================================================================


finalize_module(
    __name__,
    prefix="T",
    iter_kinds=["serial", "parallel", "unroll", "vectorized"],
    iter_holder=_IterHolder,
    with_marker=_BlockMarker,
    dtypes=[
        "int8", "int16", "int32", "int64",
        "uint8", "uint16", "uint32", "uint64", "bool",
        "float16", "float32", "float64",
    ],
    default_dtypes={"int": "int32", "float": "float32", "bool": "bool"},
)


# ============================================================================
# Back-compat shims for existing tests
#
# Tests written before the refactor reference ``mt.TLang``, ``mt.LANG_MODULES``,
# ``mt.make_var_factory``, and ``mt._OP_KIND_TO_IR_CLASS``. Provide these
# so the ``module-is-dialect`` refactor is drop-in from the test suite's
# point of view.
# ============================================================================


import sys as _sys  # noqa: E402, PLC0415
_this = _sys.modules[__name__]

# ``T = this module`` — the dialect IS the module, per the design doc.
T = _this  # type: ignore[assignment]

# ``TLang`` alias — for back-compat. New code uses ``T``.
TLang = _this  # type: ignore[assignment]


def make_var_factory(name: str, ty: Any) -> Var:
    """Legacy ``var_factory=`` shim for :class:`~tvm_ffi.pyast.IRParser`.

    Delegates to the module's auto-wired ``__ffi_make_var__``.
    """
    return _this.__ffi_make_var__(None, name, ty)  # type: ignore[attr-defined]


# The ``I`` dialect lives in ``mini.ir`` post-split (see design doc §7.7).
# Lazy-import it into ``LANG_MODULES`` so cross-dialect parsers using
# ``@I.ir_module`` keep working without an explicit import on the
# test-author's side.
from tvm_ffi.testing.mini import ir as _ir_mod  # noqa: E402, PLC0415


def _ret_call_factory(*args: Any) -> Call:
    """Bare-name ``ret(...)`` → :class:`Call` with ``op_name="ret"``.

    Used so ``Evaluate(Call(op_name="ret", args=[...]))`` — the mini-TIR
    return encoding — roundtrips through a bare ``ret(...)`` lookup
    that isn't namespaced under ``T``.
    """
    return Call(op_name="ret", args=list(args))


LANG_MODULES = {"T": _this, "I": _ir_mod, "ret": _ret_call_factory}

# Back-compat re-export: tests written pre-split import ``mt.IRModule``
# directly from ``mini.tir``. The class now lives in ``mini.ir`` (one
# module per dialect per §7.7), but we alias it here so the rename is
# transparent to existing fixtures.
IRModule = _ir_mod.IRModule


# ``_OP_KIND_TO_IR_CLASS`` — derivable from the auto-wired
# ``__ffi_op_classes__`` map; exposed for tests that assert on its shape.
def _build_op_kind_to_ir_class() -> dict[int, type]:
    result: dict[int, type] = {}
    for kind_int, ref in _this.__ffi_op_classes__.items():  # type: ignore[attr-defined]
        _, _, cls_name = ref.partition(".")
        cls = getattr(_this, cls_name, None)
        if cls is None:
            # The op_classes map points at the factory function, not the
            # IR class directly, but the factory name matches the class
            # name so look it up globally.
            for _name, _v in _sys.modules[__name__].__dict__.items():
                if _name == cls_name and isinstance(_v, type):
                    cls = _v
                    break
        if cls is not None and isinstance(cls, type):
            result[kind_int] = cls
    return result


_OP_KIND_TO_IR_CLASS = _build_op_kind_to_ir_class()
