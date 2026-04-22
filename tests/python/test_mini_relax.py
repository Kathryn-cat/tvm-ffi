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
"""Mini Relax + TIR cross-dialect roundtrip tests.

Organization (mirrors the design-doc phases):

* **R0 — Atoms**: Relax types + values.
* **R1 — Relax-only**: plain ``@R.function`` bodies with bindings, ops,
  returns.
* **R2 — Dataflow**: ``with R.dataflow(): ... R.output(v)``.
* **R3 — Cross-dialect IRModule**: both ``@T.prim_func`` and
  ``@R.function`` in one module.
* **R4 — Cross-dialect call**: ``R.call_tir("tir_fn", args, out_sinfo)``
  inside a Relax function calling a TIR primfunc in the same module.
* **R5 — Frame-dispatch mechanics**: direct checks that the per-function
  frame push elevates the right dialect.
"""

from __future__ import annotations

from typing import Any

import pytest

from tvm_ffi import pyast
from tvm_ffi.testing.mini import relax as mr
from tvm_ffi.testing.mini import tir as mt
from tvm_ffi.testing.roundtrip import assert_roundtrip


# ============================================================================
# Shared helpers
# ============================================================================


def _rt(ir: Any) -> None:
    """Roundtrip helper using the cross-dialect LANG_MODULES."""
    assert_roundtrip(
        ir,
        lang_modules=mr.LANG_MODULES,
        var_factory=mr.make_var_factory,
    )


def _sinfo(shape, dtype: str) -> mr.TensorStructInfo:
    """Fresh :class:`TensorStructInfo`. **Always call this per-use** —
    sharing one instance across multiple Var.ty / sinfo slots trips
    ``structural_eq='dag'`` at the parse boundary (same caveat as
    mini.mlir's ``_vec`` / ``_mem`` helpers)."""
    return mr.RLang.Tensor(list(shape), dtype)


def _rvar(name: str, shape, dtype: str) -> mr.Var:
    return mr.Var(name=name, ty=_sinfo(shape, dtype))


def _tvar(name: str, dtype: str = "int32") -> mt.Var:
    """Mini-TIR Var builder — for use inside ``@T.prim_func`` bodies."""
    return mt.Var(name=name, ty=mt.PrimTy(dtype=dtype))


# ============================================================================
# R0 — Atoms
# ============================================================================


@pytest.mark.parametrize(
    ("shape", "dtype"),
    [
        ([4], "float32"),
        ([3, 4], "float32"),
        ([128, 128], "float16"),
        ([1], "int32"),
    ],
)
def test_r0_tensor_struct_info_print(shape, dtype):
    """``R.Tensor(shape, dtype)`` → prints as ``T.Tensor(...)`` (printer
    hardcode)."""
    sinfo = mr.RLang.Tensor(shape, dtype)
    text = pyast.to_python(sinfo)
    assert text.startswith("T.Tensor(")
    assert f'"{dtype}"' in text


def test_r0_var_use_site_prints_name():
    v = _rvar("x", [4], "float32")
    assert pyast.to_python(v) == "x"


# ============================================================================
# R1 — Relax-only functions
# ============================================================================


def test_r1_identity_function():
    """Simplest Relax function: ``def identity(x): return x``."""
    x = _rvar("x", [4], "float32")
    fn = mr.Function(
        name="identity",
        params=[x],
        body=[mr.ReturnOp(value=x)],
    )
    text = pyast.to_python(fn)
    assert text.startswith("@R.function")
    assert "return x" in text
    _rt(fn)


def test_r1_bind_and_return():
    """``z: R.Tensor(...) = R.add(x, y); return z``."""
    x = _rvar("x", [4], "float32")
    y = _rvar("y", [4], "float32")
    z = _rvar("z", [4], "float32")
    fn = mr.Function(
        name="add",
        params=[x, y],
        body=[
            mr.Bind(var=z, value=mr.RLang.add(x, y)),
            mr.ReturnOp(value=z),
        ],
    )
    text = pyast.to_python(fn)
    assert "R.add(x, y)" in text
    assert "return z" in text
    _rt(fn)


def test_r1_multiply_chain():
    """``t1 = R.add(x, x); t2 = R.multiply(t1, y); return t2``."""
    x = _rvar("x", [4], "float32")
    y = _rvar("y", [4], "float32")
    t1 = _rvar("t1", [4], "float32")
    t2 = _rvar("t2", [4], "float32")
    fn = mr.Function(
        name="compute",
        params=[x, y],
        body=[
            mr.Bind(var=t1, value=mr.RLang.add(x, x)),
            mr.Bind(var=t2, value=mr.RLang.multiply(t1, y)),
            mr.ReturnOp(value=t2),
        ],
    )
    _rt(fn)


def test_r1_flip_op():
    """``y = R.flip(x); return y``."""
    x = _rvar("x", [4], "float32")
    y = _rvar("y", [4], "float32")
    fn = mr.Function(
        name="f",
        params=[x],
        body=[
            mr.Bind(var=y, value=mr.RLang.flip(x)),
            mr.ReturnOp(value=y),
        ],
    )
    text = pyast.to_python(fn)
    assert "R.flip(x)" in text
    _rt(fn)


# ============================================================================
# R2 — Dataflow blocks
# ============================================================================


def test_r2_dataflow_simple():
    """``with R.dataflow(): lv = R.add(x, y); R.output(lv)``."""
    x = _rvar("x", [4], "float32")
    y = _rvar("y", [4], "float32")
    lv = _rvar("lv", [4], "float32")
    fn = mr.Function(
        name="df",
        params=[x, y],
        body=[
            mr.DataflowBlock(body=[
                mr.Bind(var=lv, value=mr.RLang.add(x, y)),
                mr.RLang.output(lv),
            ]),
            mr.ReturnOp(value=lv),
        ],
    )
    text = pyast.to_python(fn)
    assert "with R.dataflow():" in text
    assert "R.output(lv)" in text
    _rt(fn)


def test_r2_dataflow_multi_bind():
    """Multi-binding dataflow: ``lv0 = R.add; lv1 = R.multiply; R.output(lv1)``."""
    x = _rvar("x", [4], "float32")
    y = _rvar("y", [4], "float32")
    lv0 = _rvar("lv0", [4], "float32")
    lv1 = _rvar("lv1", [4], "float32")
    fn = mr.Function(
        name="df",
        params=[x, y],
        body=[
            mr.DataflowBlock(body=[
                mr.Bind(var=lv0, value=mr.RLang.add(x, y)),
                mr.Bind(var=lv1, value=mr.RLang.multiply(lv0, x)),
                mr.RLang.output(lv1),
            ]),
            mr.ReturnOp(value=lv1),
        ],
    )
    _rt(fn)


# ============================================================================
# R3 — Cross-dialect IRModule (TIR + Relax)
# ============================================================================


def test_r3_module_relax_only():
    """``@I.ir_module class M: @R.function def f(): ...`` — no TIR, just
    validates the new shared :class:`mr.IRModule` can carry Relax funcs."""
    x = _rvar("x", [4], "float32")
    fn = mr.Function(name="f", params=[x], body=[mr.ReturnOp(value=x)])
    mod = mr.IRModule(name="Lib", funcs=[fn])
    text = pyast.to_python(mod)
    assert "@I.ir_module" in text
    assert "class Lib:" in text
    assert "@R.function" in text
    _rt(mod)


def test_r3_module_tir_only():
    """Mini-TIR :class:`PrimFunc` wrapped in the new cross-dialect IRModule."""
    a = _tvar("a", "int32")
    tir_fn = mt.PrimFunc(
        name="f",
        params=[a],
        body=[mt.Bind(var=_tvar("b"), value=a)],
    )
    mod = mr.IRModule(name="Lib", funcs=[tir_fn])
    text = pyast.to_python(mod)
    assert "@T.prim_func" in text
    assert "class Lib:" in text
    _rt(mod)


def test_r3_module_mixed_tir_and_relax():
    """The canonical cross-dialect case: one module holds both a
    ``@T.prim_func`` and a ``@R.function``. Each function's body
    dispatches to its own dialect via the per-function frame push.
    """
    # TIR primfunc: simple scalar add (no body call dependencies).
    ta = _tvar("a", "int32")
    tir_fn = mt.PrimFunc(
        name="tir_add",
        params=[ta],
        body=[mt.Bind(var=_tvar("t"), value=ta)],
    )
    # Relax function: pass-through.
    rx = _rvar("x", [4], "float32")
    relax_fn = mr.Function(
        name="rx_main",
        params=[rx],
        body=[mr.ReturnOp(value=rx)],
    )
    mod = mr.IRModule(name="Mixed", funcs=[tir_fn, relax_fn])
    text = pyast.to_python(mod)
    assert "@T.prim_func" in text
    assert "@R.function" in text
    assert "def tir_add" in text
    assert "def rx_main" in text
    _rt(mod)


def test_r3_module_two_relax_and_one_tir():
    """Multi-function mixed module: shapes should all survive the
    per-function dialect elevation."""
    # Two Relax funcs
    x1 = _rvar("x", [4], "float32")
    r1 = mr.Function(
        name="r1", params=[x1],
        body=[
            mr.Bind(var=_rvar("y", [4], "float32"),
                    value=mr.RLang.add(x1, x1)),
            mr.ReturnOp(value=x1),
        ],
    )
    x2 = _rvar("x", [3, 4], "float32")
    r2 = mr.Function(
        name="r2", params=[x2],
        body=[mr.ReturnOp(value=mr.RLang.flip(x2))],
    )
    # One TIR primfunc
    ta = _tvar("a", "int32")
    t1 = mt.PrimFunc(
        name="tir_helper", params=[ta],
        body=[mt.Bind(var=_tvar("b"), value=ta)],
    )
    mod = mr.IRModule(name="MixedLib", funcs=[t1, r1, r2])
    _rt(mod)


# ============================================================================
# R4 — Cross-dialect call (R.call_tir)
# ============================================================================


def test_r4_call_tir_direct():
    """``R.call_tir("tir_fn", [x], out_sinfo)`` — bare construction.

    Exercises the :class:`CallTIR` roundtrip alone — sinfo as 3rd
    positional arg, callee as string literal.
    """
    x = _rvar("x", [4], "float32")
    y = _rvar("y", [4], "float32")
    fn = mr.Function(
        name="caller",
        params=[x],
        body=[
            mr.Bind(
                var=y,
                value=mr.RLang.call_tir(
                    "add_one", [x], out_sinfo=_sinfo([4], "float32"),
                ),
            ),
            mr.ReturnOp(value=y),
        ],
    )
    text = pyast.to_python(fn)
    assert "R.call_tir(" in text
    assert '"add_one"' in text
    _rt(fn)


def test_r4_call_tir_inside_dataflow():
    """``with R.dataflow(): lv = R.call_tir(...); R.output(lv)``."""
    x = _rvar("x", [4], "float32")
    lv = _rvar("lv", [4], "float32")
    fn = mr.Function(
        name="caller",
        params=[x],
        body=[
            mr.DataflowBlock(body=[
                mr.Bind(
                    var=lv,
                    value=mr.RLang.call_tir(
                        "add_one", [x], out_sinfo=_sinfo([4], "float32"),
                    ),
                ),
                mr.RLang.output(lv),
            ]),
            mr.ReturnOp(value=lv),
        ],
    )
    _rt(fn)


def test_r4_mixed_module_with_cross_dialect_call():
    """The canonical cross-dialect pattern:

    ``@I.ir_module class M:
        @T.prim_func def add_one(X, Y): ...
        @R.function def main(x): ... R.call_tir("add_one", [x], sinfo) ...``
    """
    # TIR primfunc: ``add_one(X, Y)`` — add 1 element-wise. Mini-TIR
    # takes scalar params not buffer params, so this is a simplified
    # stand-in; the interesting bit is the cross-dialect call shape.
    X = _tvar("X", "float32")
    Y = _tvar("Y", "float32")
    tir_fn = mt.PrimFunc(
        name="add_one",
        params=[X, Y],
        # dtype goes through ``mt.imm`` so it's wrapped as the native FFI
        # dtype the parser reconstructs (bare Python strings for dtype
        # would roundtrip to ``ffi_dtype`` instances and fail the DAG
        # check).
        body=[mt.BufferStore(buffer=Y, value=X, indices=[mt.imm(0, "int32")])],
    )
    # Relax function: calls add_one via R.call_tir.
    x = _rvar("x", [4], "float32")
    y = _rvar("y", [4], "float32")
    relax_fn = mr.Function(
        name="main",
        params=[x],
        body=[
            mr.DataflowBlock(body=[
                mr.Bind(
                    var=y,
                    value=mr.RLang.call_tir(
                        "add_one", [x], out_sinfo=_sinfo([4], "float32"),
                    ),
                ),
                mr.RLang.output(y),
            ]),
            mr.ReturnOp(value=y),
        ],
    )
    mod = mr.IRModule(name="Kernel", funcs=[tir_fn, relax_fn])
    text = pyast.to_python(mod)
    assert "@T.prim_func" in text
    assert "@R.function" in text
    assert "R.call_tir(" in text
    assert "with R.dataflow():" in text
    _rt(mod)


# ============================================================================
# R5 — Frame-dispatch mechanics
# ============================================================================


def test_r5_relax_function_pushes_rlang_frame():
    """When ``@R.function`` handler is called, a Frame(dialects=[RLang])
    sits on the stack for the duration of the body parse."""
    # Construct + parse a minimal R.function and observe via a body
    # stmt that the innermost-first ``_lookup_hook("__ffi_assign__")``
    # returns RLang's impl (not a stale TIR one).
    x = _rvar("x", [4], "float32")
    y = _rvar("y", [4], "float32")
    fn = mr.Function(
        name="push_check",
        params=[x],
        body=[
            mr.Bind(var=y, value=mr.RLang.add(x, x)),
            mr.ReturnOp(value=y),
        ],
    )
    # If the frame push didn't work, parse_assign would use
    # mini.tir.Bind instead of mini.relax.Bind and the roundtrip
    # structural_equal would diverge.
    _rt(fn)


def test_r5_tir_prim_func_pushes_tlang_frame_in_mixed_module():
    """Inside a mixed module, TIR functions get TLang hooks; Relax
    functions get RLang hooks — because each decorator handler pushes
    its own dialect frame before body-parsing."""
    ta = _tvar("a", "int32")
    tir_fn = mt.PrimFunc(
        name="tir_only",
        params=[ta],
        body=[
            # If the TIR frame push failed, __ffi_assign__ might route
            # to mini.relax.Bind for this assignment and break the IR.
            mt.Bind(var=_tvar("b"), value=ta),
        ],
    )
    rx = _rvar("x", [4], "float32")
    ry = _rvar("y", [4], "float32")
    relax_fn = mr.Function(
        name="relax_only",
        params=[rx],
        body=[
            mr.Bind(var=ry, value=mr.RLang.add(rx, rx)),
            mr.ReturnOp(value=ry),
        ],
    )
    mod = mr.IRModule(name="Both", funcs=[tir_fn, relax_fn])
    _rt(mod)


def test_r5_lang_modules_has_both_dialects():
    """The cross-dialect :data:`mr.LANG_MODULES` registry exposes both
    ``T`` (type namespace + TIR decorators), ``R`` (Relax dialect), and
    ``I`` (shared module decorator)."""
    assert "T" in mr.LANG_MODULES
    assert "R" in mr.LANG_MODULES
    assert "I" in mr.LANG_MODULES


def test_r5_parser_dialect_auto_register():
    """``IRParser(lang_modules=mr.LANG_MODULES)`` registers Relax,
    TIR, and the module decorator in the base registry."""
    parser = pyast.IRParser(lang_modules=mr.LANG_MODULES)
    class_names = {type(d).__name__ for d in parser._registered_dialects}
    # _TNamespace, RLang, ILang, TLang all expected.
    assert "RLang" in class_names
    assert "TLang" in class_names
    assert "ILang" in class_names
    assert "_TNamespace" in class_names
