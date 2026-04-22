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
"""Mini-Relax — Relax-flavored fixture for cross-dialect (Relax + TIR) parser validation."""

# ruff: noqa: A003, D102, N802, UP006, UP045

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List  # noqa: UP035

from tvm_ffi import Object, pyast
from tvm_ffi import ir_traits as tr
from tvm_ffi.dataclasses import py_class
from tvm_ffi.dataclasses import field as dc_field


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
# Dataflow with-marker
# ============================================================================


@dataclass
class _DataflowMarker:
    """Returned by ``R.dataflow()`` — consumed by ``visit_with`` via
    :meth:`__ffi_with_handler__` to build a :class:`DataflowBlock`."""

    def __ffi_with_handler__(self, parser, node) -> DataflowBlock:
        with parser.push_frame(pyast.WithFrame()):
            body = parser.visit_body(node.body)
        return DataflowBlock(body=body)


# ============================================================================
# Shared helpers
# ============================================================================


def _make_value(parser: Any, name: str, ty: Any) -> Var:
    """``__ffi_make_var__`` impl for :class:`RLang` — builds a Relax :class:`Var`."""
    return Var(name=name, ty=ty)


def _assign_impl(parser, node: pyast.Assign) -> Any:
    """Dispatch ``y = expr`` to a Relax :class:`Bind`."""
    from tvm_ffi.pyast_trait_parse import parse_assign  # noqa: PLC0415

    return parse_assign(parser, node, Bind)


# ============================================================================
# RLang — the Relax dialect module
# ============================================================================


class RLang:
    """Mini-Relax ``R`` dialect module."""

    # ---- Type constructor ----
    @staticmethod
    def Tensor(shape: Any = None, dtype: Any = None) -> TensorStructInfo:
        if shape is not None:
            shape = list(shape)
        return TensorStructInfo(shape=shape, dtype=dtype)

    # ---- Decorator handler ----
    @staticmethod
    def function(parser, node) -> Function:
        """``@R.function`` → :class:`Function`."""
        from tvm_ffi.pyast_trait_parse import parse_func  # noqa: PLC0415

        with parser.push_frame(pyast.Frame(dialects=[_RLANG])):
            return parse_func(parser, node, Function)

    # ---- Dataflow with-context factory ----
    @staticmethod
    def dataflow() -> _DataflowMarker:
        return _DataflowMarker()

    # ---- Operations ----
    @staticmethod
    def add(x: Any, y: Any) -> Call:
        return Call(op_name="R.add", args=[x, y])

    @staticmethod
    def multiply(x: Any, y: Any) -> Call:
        return Call(op_name="R.multiply", args=[x, y])

    @staticmethod
    def flip(x: Any) -> Call:
        return Call(op_name="R.flip", args=[x])

    @staticmethod
    def output(*values: Any) -> Call:
        """``R.output(v, ...)`` — marks dataflow-block outputs."""
        return Call(op_name="R.output", args=list(values))

    # ---- Cross-dialect call ----
    @staticmethod
    def call_tir(callee: Any, args: Any, out_sinfo: Any) -> CallTIR:
        """``R.call_tir(callee, args, out_sinfo)`` — call a TIR primfunc."""
        return CallTIR(args=[callee, list(args), out_sinfo])

    # ---- Parser protocol hooks ----
    __ffi_make_var__ = staticmethod(_make_value)
    __ffi_assign__ = staticmethod(_assign_impl)

    @staticmethod
    def ret(parser, value: Any) -> ReturnOp:
        """``return v`` → :class:`ReturnOp`."""
        return ReturnOp(value=value)


_RLANG = RLang()


# ============================================================================
# ILang — cross-dialect IRModule decorator
# ============================================================================


class ILang:
    """``I`` dialect — module decorator for mixed TIR + Relax modules."""

    @staticmethod
    def ir_module(parser, node) -> IRModule:
        """``@I.ir_module class Name: <funcs>`` → :class:`IRModule`."""
        funcs: list = []
        with parser.scoped_frame():
            for stmt in node.body:
                if isinstance(stmt, pyast.Function):
                    funcs.append(parser.visit_function(stmt))
        return IRModule(name=node.name.name, funcs=funcs)


# ============================================================================
# Type namespace ``T`` — printer-hardcoded prefix for type traits.
# ============================================================================


class _TNamespace:
    """Unified ``T.`` type namespace used for cross-dialect parsing."""

    Tensor = staticmethod(RLang.Tensor)


def _mount_tir_into_type_namespace() -> None:
    """Attach mini-TIR's scalar dtype handles + ``prim_func`` decorator
    onto :class:`_TNamespace`."""
    from tvm_ffi.testing.mini import tir as mt  # noqa: PLC0415

    for dtype_name in (
        "int8", "int16", "int32", "int64",
        "uint8", "uint16", "uint32", "uint64", "bool",
        "float16", "float32", "float64",
    ):
        setattr(_TNamespace, dtype_name, getattr(mt.TLang, dtype_name))
    _TNamespace.prim_func = staticmethod(mt.TLang.prim_func)


_mount_tir_into_type_namespace()
T = _TNamespace()


# ============================================================================
# Parser registry
# ============================================================================


R = _RLANG
I = ILang()  # noqa: E741


def _build_lang_modules() -> dict[str, Any]:
    """Construct the cross-dialect ``LANG_MODULES`` dict."""
    from tvm_ffi.testing.mini import tir as mt  # noqa: PLC0415

    return {
        "T": T,                 # type namespace (printer hardcode)
        "R": R,                 # Relax dialect
        "I": I,                 # shared module decorator
        "_tir_singleton": mt.T,  # TIR dialect hooks (per-function frame)
    }


LANG_MODULES: dict[str, Any] = _build_lang_modules()


def make_var_factory(name: str, ty: Any) -> Var:
    """Legacy ``var_factory=`` shim for :class:`IRParser`."""
    return _make_value(None, name, ty)
