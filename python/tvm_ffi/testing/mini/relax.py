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
"""Mini-Relax — Relax-flavored dialect, auto-registered.

Uses :func:`~tvm_ffi.dialect_autogen.finalize_module` (see
``design_docs/parser_auto_registration.md``). The Python module IS the
``R`` dialect — ``from tvm_ffi.testing.mini import relax as R`` then
``R.function`` / ``R.add`` / ``R.dataflow`` all resolve via ordinary
attribute lookup, with every factory / hook auto-injected.

Keeps only Bucket C dialect-specific code:

* ``TensorStructInfo`` / ``Tensor`` factory — Relax struct-info type
  (type annotation shape; ``R.Tensor((shape,), dtype)``).
* ``_DataflowMarker`` — user-provided ``with_marker=`` for
  ``with R.dataflow(): body`` — auto-gets ``__ffi_with_handler__``.
* ``call_tir`` / ``output`` — cross-dialect call shapes outside the
  generic opaque-fallback handler.
* ``_TNamespace`` — shared ``T.`` type prefix used by the printer's
  hardcoded TyTraits prefix.
* ``LANG_MODULES`` / ``make_var_factory`` — back-compat exports.
"""

# ruff: noqa: A003, D102, N802, UP006, UP045

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List  # noqa: UP035

from tvm_ffi import Object
from tvm_ffi import ir_traits as tr
from tvm_ffi.dataclasses import py_class
from tvm_ffi.dataclasses import field as dc_field
from tvm_ffi.dialect_autogen import finalize_module


# ============================================================================
# Struct-info types
# ============================================================================


@py_class("mini.relax.TensorStructInfo", structural_eq="dag")
class TensorStructInfo(Object):
    """``R.Tensor((shape,), dtype)`` — shape + dtype struct info."""

    __ffi_ir_traits__ = tr.TensorTyTraits(
        "$field:shape", "$field:dtype", None,
    )
    shape: Any = None
    dtype: Any = None


# ============================================================================
# SSA Values
# ============================================================================


@py_class("mini.relax.Var", structural_eq="var")
class Var(Object):
    """Relax SSA variable. Struct info on ``ty`` drives the type annotation."""

    __ffi_ir_traits__ = tr.ValueTraits("$field:name", "$field:ty", None)
    name: str = dc_field(structural_eq="ignore")
    ty: Any


# ============================================================================
# Operations
# ============================================================================


@py_class("mini.relax.Call", structural_eq="dag")
class Call(Object):
    """Generic Relax call: ``R.<name>(args)`` — e.g. ``R.add(x, y)``."""

    __ffi_ir_traits__ = tr.CallTraits(
        "$field:op_name", "$field:args", None, None, None, None,
    )
    op_name: str
    args: List[Any]


@py_class("mini.relax.CallTIR", structural_eq="dag")
class CallTIR(Object):
    """``R.call_tir(callee, args, out_sinfo)`` — call a TIR primfunc."""

    __ffi_ir_traits__ = tr.CallTraits(
        "R.call_tir", "$field:args", None, None, None, None,
    )
    args: List[Any]


# ============================================================================
# Statements
# ============================================================================


@py_class("mini.relax.Bind", structural_eq="tree")
class Bind(Object):
    """``v: sinfo = expr`` — SSA binding."""

    __ffi_ir_traits__ = tr.AssignTraits(
        "$field:var", "$field:value", None, None, None, None,
    )
    value: Any
    var: Var = dc_field(structural_eq="def")


@py_class("mini.relax.ReturnOp", structural_eq="tree")
class ReturnOp(Object):
    """``return v``."""

    __ffi_ir_traits__ = tr.ReturnTraits("$field:value")
    value: Any


@py_class("mini.relax.DataflowBlock", structural_eq="tree")
class DataflowBlock(Object):
    """``with R.dataflow(): body`` — the only Relax with-block."""

    __ffi_ir_traits__ = tr.WithTraits(
        tr.RegionTraits("$field:body", None, None, None),
        None, None,
        "R.dataflow",
        None, None, None,
    )
    body: List[Any]


# ============================================================================
# Function
# ============================================================================


@py_class("mini.relax.Function", structural_eq="tree")
class Function(Object):
    """``@R.function def name(params): body``."""

    __ffi_ir_traits__ = tr.FuncTraits(
        "$field:name",
        tr.RegionTraits("$field:body", "$field:params", None, None),
        None, "R.function", None,
    )
    name: str = dc_field(structural_eq="ignore")
    params: List[Var] = dc_field(structural_eq="def")
    body: List[Any]


# ============================================================================
# IRModule — mixed TIR + Relax module.
# ``funcs`` can contain :class:`mini.tir.PrimFunc` and/or Relax :class:`Function`.
# ============================================================================


@py_class("mini.relax.IRModule", structural_eq="tree")
class IRModule(Object):
    """``@I.ir_module class Name: <funcs>`` — mixed TIR + Relax module."""

    __ffi_ir_traits__ = tr.FuncTraits(
        "$field:name",
        tr.RegionTraits("$field:funcs", None, None, None),
        None, "I.ir_module", None,
    )
    name: str = dc_field(structural_eq="ignore")
    funcs: List[Any]


# ============================================================================
# Bucket C — Dataflow with-marker
#
# Relax SSA semantics: vars defined inside a dataflow block are
# visible in the enclosing function scope (unlike TIR blocks that
# introduce a fresh scope). The auto-wired ``__ffi_with_handler__``
# would wrap in ``parser.scoped_frame()`` — dropping those vars on
# block exit. We pre-define the handler so ``finalize_module``
# skips auto-wiring (hasattr check), preserving SSA visibility.
# ============================================================================


@dataclass
class _DataflowMarker:
    """Returned by ``R.dataflow()`` — consumed by ``visit_with``."""

    def __ffi_with_handler__(self, parser: Any, node: Any) -> DataflowBlock:
        from tvm_ffi import pyast as _pyast  # noqa: PLC0415

        with parser.push_frame(_pyast.WithFrame()):
            body = parser.visit_body(node.body)
        return DataflowBlock(body=body)


# ============================================================================
# Bucket C — Tensor struct-info factory
#
# ``R.Tensor((shape,), dtype)`` is a user-facing constructor that
# normalizes ``shape`` to a list. Not auto-derivable from traits.
# ============================================================================


def Tensor(shape: Any = None, dtype: Any = None) -> TensorStructInfo:
    """``R.Tensor(shape, dtype)`` — Relax tensor struct-info factory."""
    if shape is not None:
        shape = list(shape)
    return TensorStructInfo(shape=shape, dtype=dtype)


# ============================================================================
# Bucket C — Cross-dialect call shapes
#
# ``R.call_tir(callee, args, out_sinfo)`` builds a :class:`CallTIR`
# with packed argument shape. ``R.output(*values)`` marks dataflow
# outputs — a cross-cutting Relax sugar that isn't a generic opaque call.
# ============================================================================


def call_tir(callee: Any, args: Any, out_sinfo: Any) -> CallTIR:
    """``R.call_tir(callee, args, out_sinfo)`` — call a TIR primfunc."""
    return CallTIR(args=[callee, list(args), out_sinfo])


def output(*values: Any) -> Call:
    """``R.output(v, ...)`` — marks dataflow-block outputs."""
    return Call(op_name="R.output", args=list(values))


# ============================================================================
# The single finalize_module call — auto-injects mechanical wiring:
#
# * ``function`` class-decorator handler (FuncTraits) — with dialect-A
#   frame push for the function body.
# * ``ir_module`` class-decorator handler (for the cross-dialect IRModule).
# * ``bind`` / ``__ffi_assign__`` / ``ret`` hooks (AssignTraits, ReturnTraits).
# * ``dataflow`` ctx-factory via ``with_marker=_DataflowMarker``.
# * ``__ffi_make_var__`` from :class:`Var` (ValueTraits).
# * ``__getattr__`` fallback → :class:`Call` for ``R.add`` / ``R.multiply`` /
#   ``R.flip`` / any other dynamic op name (from CallTraits).
# ============================================================================


finalize_module(
    __name__,
    prefix="R",
    with_marker=_DataflowMarker,
)


# ============================================================================
# Back-compat shims for existing tests
# ============================================================================


import sys as _sys  # noqa: E402, PLC0415

_this = _sys.modules[__name__]


# ``R = this module`` — the dialect IS the module.
R = _this  # type: ignore[assignment]
RLang = _this  # type: ignore[assignment]

# ``I = this module`` — the cross-dialect IRModule decorator handler is
# auto-wired as ``_this.ir_module``. Tests consume it via ``I.ir_module``.
I = _this  # type: ignore[assignment]
ILang = _this  # type: ignore[assignment]


def make_var_factory(name: str, ty: Any) -> Var:
    """Legacy ``var_factory=`` shim for :class:`IRParser`."""
    return _this.__ffi_make_var__(None, name, ty)  # type: ignore[attr-defined]


# ============================================================================
# Type namespace ``T`` — printer-hardcoded prefix for type traits.
#
# Known limitation per ``design_docs/parser_auto_registration.md``:
# the printer emits ``T.<dtype>`` / ``T.prim_func`` regardless of the
# owning dialect. We re-export mini-TIR's dtype handles + ``prim_func``
# decorator under a ``T`` namespace so cross-dialect tests can round-trip.
# ============================================================================


class _TNamespace:
    """Unified ``T.`` type namespace used for cross-dialect parsing."""

    Tensor = staticmethod(Tensor)


def _mount_tir_into_type_namespace() -> None:
    """Attach mini-TIR's scalar dtype handles + ``prim_func`` decorator
    onto :class:`_TNamespace`."""
    from tvm_ffi.testing.mini import tir as mt  # noqa: PLC0415

    for dtype_name in (
        "int8", "int16", "int32", "int64",
        "uint8", "uint16", "uint32", "uint64", "bool",
        "float16", "float32", "float64",
    ):
        setattr(_TNamespace, dtype_name, getattr(mt, dtype_name))
    _TNamespace.prim_func = staticmethod(mt.prim_func)


_mount_tir_into_type_namespace()
T = _TNamespace()


# ============================================================================
# Parser registry — cross-dialect lang_modules
# ============================================================================


def _build_lang_modules() -> dict[str, Any]:
    """Construct the cross-dialect ``LANG_MODULES`` dict."""
    from tvm_ffi.testing.mini import tir as mt  # noqa: PLC0415

    return {
        "T": T,                  # type namespace (printer hardcode)
        "R": R,                  # Relax dialect (this module)
        "I": I,                  # Cross-dialect IRModule decorator (this module)
        "_tir_singleton": mt,    # TIR dialect hooks (per-function frame)
    }


LANG_MODULES: dict[str, Any] = _build_lang_modules()
