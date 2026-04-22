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
"""Mini-MLIR ``vector`` dialect — Phase-6 op fall-through target.

Subscribes :class:`VectorAddOp` to the ``+`` sugar kind alongside arith.
The arith dispatcher returns ``None`` for vector operands so
``visit_operation`` falls through to vector's handler here.
"""

# ruff: noqa: A003, D102, N802, UP006, UP045

from __future__ import annotations

from typing import Any, List  # noqa: UP035

from tvm_ffi import Object, pyast
from tvm_ffi import ir_traits as tr
from tvm_ffi.dataclasses import py_class
from tvm_ffi.dialect_autogen import finalize_module

from ._common import _is_vector_type, _make_value, _operand_type


# ============================================================================
# Types
# ============================================================================


@py_class("mini.mlir.VectorType", structural_eq="dag")
class VectorType(Object):
    """``vector<4xf32>`` — prints as ``T.Tensor([4], T.f32)``."""

    __ffi_ir_traits__ = tr.TensorTyTraits(
        "$field:shape", "$field:elem_type", None,
    )
    shape: Any = None
    elem_type: Any = None


# ============================================================================
# Ops
# ============================================================================


@py_class("mini.mlir.VectorAddOp", structural_eq="dag")
class VectorAddOp(Object):
    """``vector.addf(a, b)`` — element-wise vector addition."""

    __ffi_ir_traits__ = tr.BinOpTraits(
        "$field:lhs", "$field:rhs", "+", None, "vector.addf",
    )
    lhs: Any
    rhs: Any


# ============================================================================
# Bucket C — Op fall-through handler
#
# Returns a :class:`VectorAddOp` only when both operands are vector-
# typed; otherwise returns ``None`` so ``visit_operation`` can continue
# walking dialects. The explicit ``__ffi_op_classes__`` map suppresses
# the auto-wired ``VectorAddOp._ffi_parse_op`` entry (which would
# build unconditionally).
# ============================================================================


def _op_add(parser: pyast.IRParser, node: pyast.Operation) -> Any:
    if len(node.operands) != 2:
        return None
    lhs = parser.eval_expr(node.operands[0])
    rhs = parser.eval_expr(node.operands[1])
    if _is_vector_type(_operand_type(lhs)) and _is_vector_type(_operand_type(rhs)):
        return VectorAddOp(lhs=lhs, rhs=rhs)
    return None


__ffi_op_classes__ = {
    pyast.OperationKind.Add: "vector._op_add",
}


# ============================================================================
# Bucket C — parameterized type factory + op factories
# ============================================================================


def vector(shape: List[int], elem_type: Any) -> VectorType:
    """``vector.vector(shape, elem_type)`` — vector type constructor."""
    return VectorType(shape=list(shape), elem_type=elem_type)


def addf(a: Any, b: Any) -> VectorAddOp:
    return VectorAddOp(lhs=a, rhs=b)


# ============================================================================
# Bucket C — var-construction hook
# ============================================================================


__ffi_make_var__ = staticmethod(_make_value)


# ============================================================================
# The single finalize_module call
# ============================================================================


finalize_module(__name__, prefix="vector")
