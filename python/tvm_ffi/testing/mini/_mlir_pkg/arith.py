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
"""Mini-MLIR ``arith`` dialect — scalar types, constants, int/float binops.

Owns every MLIR scalar type (``i1``…``index``, ``f16``/``f32``/``f64``)
and the int/float overloads of the standard sugar ops. ``+`` / ``-``
/ ``*`` / ``<`` / ``==`` dispatch via :attr:`__ffi_op_classes__`,
routed through the custom per-op dispatchers (``_op_add`` etc.) which
return ``None`` for non-arith operand types so ``visit_operation``
falls through to the vector dialect (Phase-6 op-classes fall-through).
"""

# ruff: noqa: A003, D102, N802, UP006, UP045

from __future__ import annotations

from typing import Any, List, Optional  # noqa: UP035

from tvm_ffi import Object, pyast
from tvm_ffi import ir_traits as tr
from tvm_ffi.dataclasses import py_class
from tvm_ffi.dialect_autogen import finalize_module

from ._common import (
    _is_float_type,
    _is_int_type,
    _make_value,
    _operand_type,
)


# ============================================================================
# Scalar types
# ============================================================================


@py_class("mini.mlir.IntegerType", structural_eq="dag")
class IntegerType(Object):
    """``i1``, ``i8``, ``i16``, ``i32``, ``i64``, ``index``."""

    __ffi_ir_traits__ = tr.PrimTyTraits("$field:name")
    name: str


@py_class("mini.mlir.FloatType", structural_eq="dag")
class FloatType(Object):
    """``f16``, ``f32``, ``f64``."""

    __ffi_ir_traits__ = tr.PrimTyTraits("$field:name")
    name: str


# ============================================================================
# Constant op (attribute-carrying scalar literal)
# ============================================================================


@py_class("mini.mlir.ConstantOp", structural_eq="dag")
class ConstantOp(Object):
    """``arith.constant(value, ty)`` — attribute-carrying scalar literal."""

    __ffi_ir_traits__ = tr.CallTraits(
        "arith.constant", "$field:args", None, None, None, None,
    )
    args: List[Any]


# ============================================================================
# Binary ops — int/float overloads
# ============================================================================


def _binop_traits(op: str, func_name: str) -> tr.BinOpTraits:
    """BinOp trait factory for arith ops (infix sugar allowed)."""
    return tr.BinOpTraits("$field:lhs", "$field:rhs", op, None, func_name)


@py_class("mini.mlir.AddIOp", structural_eq="dag")
class AddIOp(Object):
    """``arith.addi(a, b)`` — integer addition."""

    __ffi_ir_traits__ = _binop_traits("+", "arith.addi")
    lhs: Any
    rhs: Any


@py_class("mini.mlir.SubIOp", structural_eq="dag")
class SubIOp(Object):
    __ffi_ir_traits__ = _binop_traits("-", "arith.subi")
    lhs: Any
    rhs: Any


@py_class("mini.mlir.MulIOp", structural_eq="dag")
class MulIOp(Object):
    __ffi_ir_traits__ = _binop_traits("*", "arith.muli")
    lhs: Any
    rhs: Any


@py_class("mini.mlir.AddFOp", structural_eq="dag")
class AddFOp(Object):
    """``arith.addf(a, b)`` — float addition."""

    __ffi_ir_traits__ = _binop_traits("+", "arith.addf")
    lhs: Any
    rhs: Any


@py_class("mini.mlir.SubFOp", structural_eq="dag")
class SubFOp(Object):
    __ffi_ir_traits__ = _binop_traits("-", "arith.subf")
    lhs: Any
    rhs: Any


@py_class("mini.mlir.MulFOp", structural_eq="dag")
class MulFOp(Object):
    __ffi_ir_traits__ = _binop_traits("*", "arith.mulf")
    lhs: Any
    rhs: Any


@py_class("mini.mlir.CmpIOp", structural_eq="dag")
class CmpIOp(Object):
    """``arith.cmpi(a, b)`` — integer comparison (defaults to ``<``)."""

    __ffi_ir_traits__ = _binop_traits("<", "arith.cmpi")
    lhs: Any
    rhs: Any


# ============================================================================
# Bucket C — Type-predicate op dispatch
#
# Arith owns both int and float overloads for every sugar op; the
# resolver picks one based on operand types. Returns ``None`` for
# non-arith operand types (e.g. vector) so ``visit_operation`` can
# fall through to the vector dialect. Auto-wiring's generic
# ``_ffi_parse_op`` factory always builds a fixed class and can't do
# this dispatch, so we skip it for these ops (see ``__ffi_op_classes__``
# manual wiring below).
# ============================================================================


def _pick_binop_cls(
    lhs: Any, rhs: Any, int_cls: type, float_cls: type,
) -> Optional[type]:
    lhs_ty, rhs_ty = _operand_type(lhs), _operand_type(rhs)
    if _is_int_type(lhs_ty) and _is_int_type(rhs_ty):
        return int_cls
    if _is_float_type(lhs_ty) and _is_float_type(rhs_ty):
        return float_cls
    return None


def _op_binary(
    parser: pyast.IRParser,
    node: pyast.Operation,
    int_cls: type,
    float_cls: type,
) -> Any:
    if len(node.operands) != 2:
        result = parser.eval_expr(node.operands[0])
        for raw in node.operands[1:]:
            rhs = parser.eval_expr(raw)
            inner = _pick_binop_cls(result, rhs, int_cls, float_cls)
            if inner is None:
                return None
            result = inner(lhs=result, rhs=rhs)
        return result
    lhs = parser.eval_expr(node.operands[0])
    rhs = parser.eval_expr(node.operands[1])
    cls = _pick_binop_cls(lhs, rhs, int_cls, float_cls)
    if cls is None:
        return None
    return cls(lhs=lhs, rhs=rhs)


def _op_add(parser: Any, node: Any) -> Any:
    return _op_binary(parser, node, AddIOp, AddFOp)


def _op_sub(parser: Any, node: Any) -> Any:
    return _op_binary(parser, node, SubIOp, SubFOp)


def _op_mul(parser: Any, node: Any) -> Any:
    return _op_binary(parser, node, MulIOp, MulFOp)


def _op_lt(parser: Any, node: Any) -> Any:
    return _op_binary(parser, node, CmpIOp, CmpIOp)


def _op_eq(parser: Any, node: Any) -> Any:
    return _op_binary(parser, node, CmpIOp, CmpIOp)


# Pre-defined — suppresses auto-wiring's contribution.
__ffi_op_classes__ = {
    pyast.OperationKind.Add: "arith._op_add",
    pyast.OperationKind.Sub: "arith._op_sub",
    pyast.OperationKind.Mult: "arith._op_mul",
    pyast.OperationKind.Lt: "arith._op_lt",
    pyast.OperationKind.Eq: "arith._op_eq",
}


# ============================================================================
# Bucket C — Explicit type attributes and op factories
#
# The scalar type instances (``i1``, ``f32``, …) are PrimTy objects on
# the module. ``finalize_module``'s ``dtypes=[...]`` would build these
# but we want to keep the explicit names as module-level attrs for
# clarity (they're also referenced by cross-dialect code and the ``T.``
# namespace forwarder).
# ============================================================================


i1 = IntegerType(name="i1")
i8 = IntegerType(name="i8")
i16 = IntegerType(name="i16")
i32 = IntegerType(name="i32")
i64 = IntegerType(name="i64")
index = IntegerType(name="index")
f16 = FloatType(name="f16")
f32 = FloatType(name="f32")
f64 = FloatType(name="f64")


def constant(value: Any, ty: Any) -> ConstantOp:
    """``arith.constant(value, ty)`` — attribute-carrying scalar literal."""
    return ConstantOp(args=[value, ty])


def addi(a: Any, b: Any) -> AddIOp:
    return AddIOp(lhs=a, rhs=b)


def subi(a: Any, b: Any) -> SubIOp:
    return SubIOp(lhs=a, rhs=b)


def muli(a: Any, b: Any) -> MulIOp:
    return MulIOp(lhs=a, rhs=b)


def addf(a: Any, b: Any) -> AddFOp:
    return AddFOp(lhs=a, rhs=b)


def subf(a: Any, b: Any) -> SubFOp:
    return SubFOp(lhs=a, rhs=b)


def mulf(a: Any, b: Any) -> MulFOp:
    return MulFOp(lhs=a, rhs=b)


def cmpi(a: Any, b: Any) -> CmpIOp:
    return CmpIOp(lhs=a, rhs=b)


# ============================================================================
# Bucket C — Explicit var-construction hook and default-ty handles
# ============================================================================


__ffi_make_var__ = staticmethod(_make_value)
__ffi_default_int_ty__ = i32
__ffi_default_float_ty__ = f32
__ffi_default_bool_ty__ = i1


# ============================================================================
# The single finalize_module call — auto-injects:
#
# * ``_ffi_parse_op`` staticmethod on each binop class (used by the
#   per-class sugar path — harmless even though __ffi_op_classes__
#   points at the custom dispatchers instead).
# * ``__getattr__`` fallback for unknown ``arith.<name>(...)`` lookups
#   → builds a :class:`ConstantOp`-shaped :class:`CallTraits` IR. In
#   practice ``constant``, ``addi``, ``subi``, ``muli``, … are
#   explicitly defined above so the fallback is rarely hit.
# ============================================================================


finalize_module(__name__, prefix="arith")
