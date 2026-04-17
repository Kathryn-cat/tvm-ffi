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
"""Mini-Relax — Relax-flavored fixtures for trait validation.

What this dialect stresses (beyond mini.tir):

* ``R.`` prefix instead of ``T.`` — proves prefix is per-dialect.
* :class:`Function` with ``text_printer_kind="R.function"``.
* :class:`Call` with attrs/kwargs — exercises ``CallTraits.attrs`` /
  ``CallTraits.kwargs`` paths.
* :class:`MatchCast` — extra :class:`AssignTraits` shape.
* :class:`TensorStructInfo` / :class:`ShapeStructInfo` — extra
  ``TyTraits`` shapes (complement to :mod:`mini.tir`'s ``BufferTy``).
"""

# ruff: noqa: A003, D102, N802, UP006, UP045

from __future__ import annotations

from typing import Any, List, Optional  # noqa: UP035

from tvm_ffi import Object, pyast
from tvm_ffi import ir_traits as tr
from tvm_ffi.dataclasses import py_class
from tvm_ffi.dataclasses import field as dc_field


# ============================================================================
# Types — Relax struct info
# ============================================================================


@py_class("mini.relax.PrimSI", structural_eq="dag")
class PrimStructInfo(Object):
    """Scalar struct info — ``R.PrimType`` rendering via ``PrimTyTraits``."""

    __ffi_ir_traits__ = tr.PrimTyTraits("$field:dtype")
    dtype: str


@py_class("mini.relax.TensorSI", structural_eq="dag")
class TensorStructInfo(Object):
    """Tensor struct info — ``T.Tensor((shape...), dtype)`` via ``TensorTyTraits``."""

    __ffi_ir_traits__ = tr.TensorTyTraits(
        "$field:shape", "$field:dtype", None,
    )
    shape: Optional[List[Any]] = None
    dtype: Optional[str] = None


@py_class("mini.relax.ShapeSI", structural_eq="dag")
class ShapeStructInfo(Object):
    """Shape struct info — ``T.Shape((dims...))`` via ``ShapeTyTraits``."""

    __ffi_ir_traits__ = tr.ShapeTyTraits("$field:dims", "$field:ndim")
    dims: Optional[List[Any]] = None
    ndim: Optional[int] = None


@py_class("mini.relax.TupleSI", structural_eq="dag")
class TupleStructInfo(Object):
    """Tuple struct info — ``T.Tuple(f1, f2, ...)`` via ``TupleTyTraits``."""

    __ffi_ir_traits__ = tr.TupleTyTraits("$field:fields")
    fields: List[Any]


@py_class("mini.relax.FuncSI", structural_eq="dag")
class FuncStructInfo(Object):
    """Function struct info — ``I.FuncType((params...), ret)`` via ``FuncTyTraits``.
    """

    __ffi_ir_traits__ = tr.FuncTyTraits("$field:params", "$field:ret")
    params: Optional[List[Any]] = None
    ret: Optional[Any] = None


# ============================================================================
# Expressions
# ============================================================================


@py_class("mini.relax.Var", structural_eq="var")
class Var(Object):
    """Relax variable — ``ValueTraits`` with struct_info as the type."""

    __ffi_ir_traits__ = tr.ValueTraits("$field:name", "$field:struct_info", None)
    name: str = dc_field(structural_eq="ignore")
    struct_info: Any


@py_class("mini.relax.Constant", structural_eq="dag")
class Constant(Object):
    """Constant — Level 0 default printer (``mini.relax.Constant(...)``)."""

    value: Any
    dtype: str


@py_class("mini.relax.Call", structural_eq="dag")
class Call(Object):
    """Call with literal callee + ``attrs`` keyword (``CallTraits.attrs`` path)."""

    __ffi_ir_traits__ = tr.CallTraits(
        "$field:op_name",
        "$field:args",
        "$field:attrs",
        None, None, None,
    )
    op_name: str
    args: List[Any]
    attrs: Optional[Any] = None


@py_class("mini.relax.Tuple", structural_eq="dag")
class Tuple(Object):
    """Tuple literal — Level 0."""

    fields: List[Any]


@py_class("mini.relax.TupleGetItem", structural_eq="dag")
class TupleGetItem(Object):
    """Indexed tuple access — Level 0."""

    tuple_value: Any
    index: int


# ============================================================================
# Bindings (statements)
# ============================================================================


@py_class("mini.relax.VarBinding", structural_eq="tree")
class VarBinding(Object):
    """``var: si = value`` — ``AssignTraits``."""

    __ffi_ir_traits__ = tr.AssignTraits(
        "$field:var", "$field:value", None, None, None, None,
    )
    value: Any
    var: Var = dc_field(structural_eq="def")


@py_class("mini.relax.MatchCast", structural_eq="tree")
class MatchCast(Object):
    """``var = R.match_cast(value, struct_info)`` — assign with kind wrapper."""

    __ffi_ir_traits__ = tr.AssignTraits(
        "$field:var", "$method:wrapped_rhs", None, None, None, None,
    )
    value: Any
    struct_info: Any
    var: Var = dc_field(structural_eq="def")

    def wrapped_rhs(self) -> Any:
        return Call(
            op_name="R.match_cast",
            args=[self.value, self.struct_info],
        )


# ============================================================================
# Functions
# ============================================================================


@py_class("mini.relax.Function", structural_eq="tree")
class Function(Object):
    """``@R.function\\ndef name(params): body``."""

    __ffi_ir_traits__ = tr.FuncTraits(
        "$field:name",
        tr.RegionTraits("$field:body", "$field:params", None, None),
        None, "R.function", None,
    )
    name: str = dc_field(structural_eq="ignore")
    params: List[Var] = dc_field(structural_eq="def")
    body: List[Any]


@py_class("mini.relax.IRModule", structural_eq="tree")
class IRModule(Object):
    """``@I.ir_module\\nclass Name: <funcs>`` — same as mini.tir.IRModule."""

    __ffi_ir_traits__ = tr.FuncTraits(
        "$field:name",
        tr.RegionTraits("$field:funcs", None, None, None),
        None, "I.ir_module", None,
    )
    name: str = dc_field(structural_eq="ignore")
    funcs: List[Any]


# ============================================================================
# R language module
# ============================================================================


class RLang:
    """Mini-Relax ``R`` language module."""

    @staticmethod
    def Tensor(
        shape: Optional[Any] = None,
        dtype: Optional[str] = None,
    ) -> TensorStructInfo:
        if shape is not None and not isinstance(shape, (list, tuple)):
            shape = [shape]
        return TensorStructInfo(
            shape=list(shape) if shape is not None else None,
            dtype=dtype,
        )

    @staticmethod
    def Shape(
        dims: Optional[Any] = None,
        ndim: Optional[int] = None,
    ) -> ShapeStructInfo:
        if dims is not None and not isinstance(dims, (list, tuple)):
            dims = [dims]
        return ShapeStructInfo(
            dims=list(dims) if dims is not None else None,
            ndim=ndim,
        )

    @staticmethod
    def Tuple(*fields: Any) -> TupleStructInfo:
        return TupleStructInfo(fields=list(fields))

    @staticmethod
    def match_cast(value: Any, si: Any) -> Call:
        # Construct-time helper: produces a Call so a parser sees the same
        # shape that the printer emits via wrapped_rhs above.
        return Call(op_name="R.match_cast", args=[value, si])


# ---- Hooks (parser-side) ----


def _bind_hook(parser, var: Var, rhs: Any) -> VarBinding:
    return VarBinding(var=var, value=rhs)


def _function_handler(parser, node) -> Function:
    from tvm_ffi.pyast_trait_parse import parse_func  # noqa: PLC0415
    return parse_func(parser, node, Function)


RLang.bind = staticmethod(_bind_hook)
RLang.function = staticmethod(_function_handler)


# ============================================================================
# I language module (shared with mini.tir.IRModule)
# ============================================================================


def _ir_module_handler(parser, node) -> IRModule:
    funcs: list = []
    parser.push_scope()
    try:
        for stmt in node.body:
            if isinstance(stmt, pyast.Function):
                funcs.append(parser.visit_function(stmt))
    finally:
        parser.pop_scope()
    return IRModule(name=node.name.name, funcs=funcs)


class ILang:
    """Mini-Relax ``I`` language module."""

    ir_module = staticmethod(_ir_module_handler)

    @staticmethod
    def FuncType(  # noqa: N802
        params: Optional[Any] = None,
        ret: Optional[Any] = None,
    ) -> FuncStructInfo:
        if params is not None and not isinstance(params, (list, tuple)):
            params = [params]
        return FuncStructInfo(
            params=list(params) if params is not None else None,
            ret=ret,
        )


# ============================================================================
# Parser config
# ============================================================================


R = RLang()
I = ILang()  # noqa: E741

LANG_MODULES: dict[str, Any] = {"R": R, "I": I}


def make_var_factory(name: str, ty: Any) -> Var:
    """Default ``var_factory`` for mini.relax."""
    return Var(name=name, struct_info=ty)
