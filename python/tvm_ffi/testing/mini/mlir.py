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
"""Mini-MLIR — MLIR-flavored fixtures for trait validation."""

# ruff: noqa: A003, D102, N802, UP006, UP045

from __future__ import annotations

from typing import Any, List, Optional  # noqa: UP035

from tvm_ffi import Object, pyast
from tvm_ffi import ir_traits as tr
from tvm_ffi.dataclasses import py_class
from tvm_ffi.dataclasses import field as dc_field


# ============================================================================
# Types — bare MLIR-style names (i32, f32, index, ...)
# ============================================================================


@py_class("mini.mlir.IntegerType", structural_eq="dag")
class IntegerType(Object):
    """``i32``, ``i64``, ``i1``, ``index`` — prints as ``T.<name>`` (FFI hardcode)."""

    __ffi_ir_traits__ = tr.PrimTyTraits("$field:name")
    name: str  # "i32" / "i64" / "i1" / "index" / ...


@py_class("mini.mlir.FloatType", structural_eq="dag")
class FloatType(Object):
    """``f16``, ``f32``, ``f64``."""

    __ffi_ir_traits__ = tr.PrimTyTraits("$field:name")
    name: str


@py_class("mini.mlir.MemRefType", structural_eq="dag")
class MemRefType(Object):
    """``memref<2x3xf32>`` — uses ``BufferTyTraits``."""

    __ffi_ir_traits__ = tr.BufferTyTraits(
        "$field:shape", "$field:elem_type", None, None, None,
    )
    shape: List[int]
    elem_type: Any


@py_class("mini.mlir.TensorType", structural_eq="dag")
class TensorType(Object):
    """``tensor<?x?xf32>``."""

    __ffi_ir_traits__ = tr.TensorTyTraits(
        "$field:shape", "$field:elem_type", None,
    )
    shape: Optional[List[Any]] = None
    elem_type: Optional[Any] = None


# ============================================================================
# Values
# ============================================================================


@py_class("mini.mlir.Value", structural_eq="var")
class Value(Object):
    """SSA value — ``ValueTraits`` (mlir.Value at python-binding level)."""

    __ffi_ir_traits__ = tr.ValueTraits("$field:name", "$field:ty", None)
    name: str = dc_field(structural_eq="ignore")
    ty: Any


@py_class("mini.mlir.IntConstant", structural_eq="dag")
class IntConstant(Object):
    """``arith.constant 0 : i32`` — Level 0 (no trait, default print)."""

    value: int
    ty: Any


# ============================================================================
# Operations (calls)
# ============================================================================


@py_class("mini.mlir.Op", structural_eq="dag")
class Op(Object):
    """Generic dialect op: ``arith.addi(a, b)`` / ``func.call(...)``."""

    __ffi_ir_traits__ = tr.CallTraits(
        "$field:name", "$field:operands",
        "$field:attrs", None, None, None,
    )
    name: str  # e.g. "arith.addi"
    operands: List[Any]
    attrs: Optional[Any] = None


# ============================================================================
# Statements / regions
# ============================================================================


@py_class("mini.mlir.AssignOp", structural_eq="tree")
class AssignOp(Object):
    """``%v = arith.addi %a, %b : i32`` — bind a result of an op."""

    __ffi_ir_traits__ = tr.AssignTraits(
        "$field:result", "$field:op", None, None, None, None,
    )
    op: Any
    result: Value = dc_field(structural_eq="def")


@py_class("mini.mlir.ScfIf", structural_eq="tree")
class ScfIf(Object):
    """``scf.if cond: ... else: ...`` — different ``IfTraits`` shape from mini.tir."""

    __ffi_ir_traits__ = tr.IfTraits(
        "$field:cond",
        tr.RegionTraits("$field:then_body", None, None, None),
        tr.RegionTraits("$field:else_body", None, None, None),
    )
    cond: Any
    then_body: List[Any]
    else_body: List[Any] = dc_field(default_factory=list)


@py_class("mini.mlir.ScfFor", structural_eq="tree")
class ScfFor(Object):
    """``scf.for %i = %lb to %ub step %step: body``."""

    __ffi_ir_traits__ = tr.ForTraits(
        tr.RegionTraits("$field:body", "$field:iv", None, None),
        "$field:lb", "$field:ub", "$field:step",
        None, None, None, None,
    )
    iv: Value = dc_field(structural_eq="def")
    lb: Any
    ub: Any
    step: Any
    body: List[Any]


# ============================================================================
# Functions
# ============================================================================


@py_class("mini.mlir.FuncOp", structural_eq="tree")
class FuncOp(Object):
    """``func.func @name(%a: i32, %b: i32) -> i32: body``."""

    __ffi_ir_traits__ = tr.FuncTraits(
        "$field:name",
        tr.RegionTraits("$field:body", "$field:params", None, None),
        None, "func.func", None,
    )
    name: str = dc_field(structural_eq="ignore")
    params: List[Value] = dc_field(structural_eq="def")
    body: List[Any]


@py_class("mini.mlir.ModuleOp", structural_eq="tree")
class ModuleOp(Object):
    """``builtin.module {...}`` — class-form FuncTraits."""

    __ffi_ir_traits__ = tr.FuncTraits(
        "$field:name",
        tr.RegionTraits("$field:funcs", None, None, None),
        None, "builtin.module", None,
    )
    name: str = dc_field(structural_eq="ignore")
    funcs: List[Any]


# ============================================================================
# Dialect language modules
# ============================================================================


class ArithLang:
    """``arith`` dialect — type/op factories."""

    @staticmethod
    def i1(): return IntegerType(name="i1")
    @staticmethod
    def i8(): return IntegerType(name="i8")
    @staticmethod
    def i16(): return IntegerType(name="i16")
    @staticmethod
    def i32(): return IntegerType(name="i32")
    @staticmethod
    def i64(): return IntegerType(name="i64")
    @staticmethod
    def index(): return IntegerType(name="index")
    @staticmethod
    def f16(): return FloatType(name="f16")
    @staticmethod
    def f32(): return FloatType(name="f32")
    @staticmethod
    def f64(): return FloatType(name="f64")

    @staticmethod
    def constant(value: int, ty: Any) -> Op:
        return Op(name="arith.constant", operands=[IntConstant(value=value, ty=ty)])

    @staticmethod
    def addi(a: Any, b: Any) -> Op: return Op(name="arith.addi", operands=[a, b])
    @staticmethod
    def subi(a: Any, b: Any) -> Op: return Op(name="arith.subi", operands=[a, b])
    @staticmethod
    def muli(a: Any, b: Any) -> Op: return Op(name="arith.muli", operands=[a, b])
    @staticmethod
    def cmpi(pred: str, a: Any, b: Any) -> Op:
        return Op(name="arith.cmpi", operands=[a, b], attrs={"predicate": pred})


class ScfLang:
    """``scf`` dialect — control-flow ops."""

    @staticmethod
    def for_(iv: str, lb: Any, ub: Any, step: Any, body: List[Any]) -> ScfFor:
        return ScfFor(
            iv=Value(name=iv, ty=IntegerType(name="index")),
            lb=lb, ub=ub, step=step, body=body,
        )

    @staticmethod
    def if_(cond: Any, then_body: List[Any], else_body: Optional[List[Any]] = None) -> ScfIf:
        return ScfIf(cond=cond, then_body=then_body, else_body=else_body or [])


class FuncLang:
    """``func`` dialect — calls / function-op factories."""

    @staticmethod
    def call(callee: str, *args: Any) -> Op:
        return Op(name="func.call", operands=list(args), attrs={"callee": callee})

    @staticmethod
    def return_(*values: Any) -> Op:
        return Op(name="func.return", operands=list(values))


# ---- Parser hooks ----


def _bind_hook(parser, var: Value, rhs: Any) -> AssignOp:
    return AssignOp(result=var, op=rhs)


def _func_handler(parser, node) -> FuncOp:
    from tvm_ffi.pyast_trait_parse import parse_func  # noqa: PLC0415
    return parse_func(parser, node, FuncOp)


def _module_handler(parser, node) -> ModuleOp:
    funcs: list = []
    parser.push_scope()
    try:
        for stmt in node.body:
            if isinstance(stmt, pyast.Function):
                funcs.append(parser.visit_function(stmt))
    finally:
        parser.pop_scope()
    return ModuleOp(name=node.name.name, funcs=funcs)


# Decorator hooks live on the dialect module that owns the decorator name.
FuncLang.func = staticmethod(_func_handler)
FuncLang.bind = staticmethod(_bind_hook)


class BuiltinLang:
    """``builtin`` dialect — for ``@builtin.module`` decorator."""

    module = staticmethod(_module_handler)


# ============================================================================
# Parser config
# ============================================================================


arith = ArithLang()
scf = ScfLang()
func = FuncLang()
builtin = BuiltinLang()

LANG_MODULES: dict[str, Any] = {
    "arith": arith,
    "scf": scf,
    "func": func,
    "builtin": builtin,
}


def make_var_factory(name: str, ty: Any) -> Value:
    """Default ``var_factory`` for mini.mlir."""
    if ty is None:
        ty = IntegerType(name="i32")
    return Value(name=name, ty=ty)
