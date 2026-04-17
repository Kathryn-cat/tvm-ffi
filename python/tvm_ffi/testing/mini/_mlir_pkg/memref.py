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
"""Mini-MLIR ``memref`` dialect — ref types + load/store ops.

Sugar:
* ``A[i]`` → :class:`LoadOp` via the auto-wired ``load`` hook
  (:class:`LoadTraits`).
* ``A[i] = v`` → :class:`StoreOp` via the auto-wired ``__ffi_assign__``
  router (:class:`StoreTraits`).
"""

# ruff: noqa: A003, D102, N802, UP006, UP045

from __future__ import annotations

from typing import Any, List  # noqa: UP035

from tvm_ffi import Object
from tvm_ffi import ir_traits as tr
from tvm_ffi.dataclasses import py_class
from tvm_ffi.dialect_autogen import finalize_module

from ._common import Value, _make_value


# ============================================================================
# Types
# ============================================================================


@py_class("mini.mlir.MemRefType", structural_eq="dag")
class MemRefType(Object):
    """``memref<2x3xf32>`` — prints as ``T.Buffer([2, 3], T.f32)``."""

    __ffi_ir_traits__ = tr.BufferTyTraits(
        "$field:shape", "$field:elem_type", None, None, None,
    )
    shape: Any
    elem_type: Any


# ============================================================================
# Ops
# ============================================================================


@py_class("mini.mlir.LoadOp", structural_eq="dag")
class LoadOp(Object):
    """``memref.load(ref, [indices...])`` — buffer read."""

    __ffi_ir_traits__ = tr.LoadTraits("$field:ref", "$field:indices", None)
    ref: Value
    indices: List[Any]


@py_class("mini.mlir.StoreOp", structural_eq="tree")
class StoreOp(Object):
    """``memref.store(value, ref, [indices...])`` — buffer write."""

    __ffi_ir_traits__ = tr.StoreTraits(
        "$field:ref", "$field:value", "$field:indices", None,
    )
    ref: Value
    value: Any
    indices: List[Any]


# ============================================================================
# Bucket C — parameterized type factory + user-facing op factories
# ============================================================================


def memref(shape: Any, elem_type: Any) -> MemRefType:
    """``memref.memref(shape, elem_type)`` — MemRef type constructor."""
    return MemRefType(shape=list(shape), elem_type=elem_type)


def load_op(ref: Value, indices: List[Any]) -> LoadOp:
    """Explicit :class:`LoadOp` factory with a name distinct from the
    auto-wired ``load`` hook (which takes ``(parser, obj, indices)``)."""
    return LoadOp(ref=ref, indices=list(indices))


def store_op(value: Any, ref: Value, indices: List[Any]) -> StoreOp:
    """Explicit :class:`StoreOp` factory with a name distinct from the
    auto-wired ``__ffi_assign__`` path."""
    return StoreOp(ref=ref, value=value, indices=list(indices))


# ============================================================================
# Bucket C — var-construction hook
# ============================================================================


__ffi_make_var__ = staticmethod(_make_value)


# ============================================================================
# The single finalize_module call — auto-injects:
#
# * ``load`` hook (LoadTraits) — sugar ``A[i]`` → LoadOp.
# * ``buffer_store`` hook + ``__ffi_assign__`` router (StoreTraits) —
#   ``A[i] = v`` → StoreOp.
# ============================================================================


finalize_module(__name__, prefix="memref")
