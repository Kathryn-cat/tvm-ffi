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
"""Shared mini-MLIR building blocks: SSA :class:`Value`, generic
:class:`BindOp`, and operand-type introspection helpers.

These are cross-dialect — every mini-MLIR dialect consumes
:class:`Value` for param typing and :class:`BindOp` as the SSA-binding
shape. Lives in ``_common.py`` so every per-dialect module can import
without circularity.
"""

# ruff: noqa: A003, D102, N802, UP006, UP045

from __future__ import annotations

from typing import Any, List  # noqa: UP035

from tvm_ffi import Object
from tvm_ffi import ir_traits as tr
from tvm_ffi.dataclasses import py_class
from tvm_ffi.dataclasses import field as dc_field


# ============================================================================
# SSA Values — shared across all mini-MLIR dialects
# ============================================================================


@py_class("mini.mlir.Value", structural_eq="var")
class Value(Object):
    """SSA value. Every ``c = op(...)`` assignment introduces one."""

    __ffi_ir_traits__ = tr.ValueTraits("$field:name", "$field:ty", None)
    name: str = dc_field(structural_eq="ignore")
    ty: Any


# ============================================================================
# Generic BindOp — the SSA-binding shape used by every dialect
# ============================================================================


@py_class("mini.mlir.BindOp", structural_eq="tree")
class BindOp(Object):
    """SSA binding: ``result = op``. All dialects share this shape."""

    __ffi_ir_traits__ = tr.AssignTraits(
        "$field:result", "$field:op", None, None, None, None,
    )
    op: Any
    result: Value = dc_field(structural_eq="def")


# ============================================================================
# Var-construction hook — every dialect reuses this Value constructor
# ============================================================================


def _make_value(parser: Any, name: str, ty: Any) -> Value:
    """``__ffi_make_var__`` impl for every mini-MLIR dialect — all param
    types (scalar from arith, ref from memref, vector from vector) build
    the same :class:`Value` envelope."""
    return Value(name=name, ty=ty)


# ============================================================================
# Operand-type introspection (used by arith/vector op-dispatch)
# ============================================================================


def _operand_type(v: Any) -> Any:
    """Best-effort type extraction from a parser-side operand.

    Handles:
    * :class:`Value` → its declared ``ty`` field.
    * ``ConstantOp(args=[value, ty])`` → the typed ``ty`` tail.
    * raw Python ``int`` / ``float`` / ``bool`` → synthetic primitive
      type so type-predicate dispatch has something to match.
    * anything else → ``None`` (dispatcher treats as "unknown").
    """
    import builtins as _bi  # noqa: PLC0415

    if isinstance(v, Value):
        return v.ty
    # ``ConstantOp`` check via duck-typing to avoid a circular import
    # from ``arith`` at module-load time.
    if (hasattr(v, "args") and hasattr(v, "__ffi_ir_traits__")
            and isinstance(v.__ffi_ir_traits__, tr.CallTraits)
            and getattr(v.__ffi_ir_traits__, "op", None) == "arith.constant"
            and v.args):
        return v.args[-1]
    if isinstance(v, _bi.bool):
        from .arith import IntegerType  # noqa: PLC0415
        return IntegerType(name="i1")
    if isinstance(v, _bi.int):
        from .arith import IntegerType  # noqa: PLC0415
        return IntegerType(name="i32")
    if isinstance(v, _bi.float):
        from .arith import FloatType  # noqa: PLC0415
        return FloatType(name="f32")
    return None


def _is_int_type(ty: Any) -> bool:
    from .arith import IntegerType  # noqa: PLC0415

    return isinstance(ty, IntegerType)


def _is_float_type(ty: Any) -> bool:
    from .arith import FloatType  # noqa: PLC0415

    return isinstance(ty, FloatType)


def _is_vector_type(ty: Any) -> bool:
    from .vector import VectorType  # noqa: PLC0415

    return isinstance(ty, VectorType)
