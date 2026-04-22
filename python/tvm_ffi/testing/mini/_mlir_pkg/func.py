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
"""Mini-MLIR ``func`` dialect — function op, return, call."""

# ruff: noqa: A003, D102, N802, UP006, UP045

from __future__ import annotations

from typing import Any, List  # noqa: UP035

from tvm_ffi import Object
from tvm_ffi import ir_traits as tr
from tvm_ffi.dataclasses import py_class
from tvm_ffi.dataclasses import field as dc_field
from tvm_ffi.dialect_autogen import finalize_module

from ._common import Value, _make_value


# ============================================================================
# Ops
# ============================================================================


@py_class("mini.mlir.FuncOp", structural_eq="tree")
class FuncOp(Object):
    """``@func.func def name(params): body`` — function definition."""

    __ffi_ir_traits__ = tr.FuncTraits(
        "$field:name",
        tr.RegionTraits("$field:body", "$field:params", None, None),
        None, "func.func", None,
    )
    name: str = dc_field(structural_eq="ignore")
    params: List[Value] = dc_field(structural_eq="def")
    body: List[Any]


@py_class("mini.mlir.ReturnOp", structural_eq="tree")
class ReturnOp(Object):
    """``return v`` → ``func.return(v)``."""

    __ffi_ir_traits__ = tr.ReturnTraits("$field:value")
    value: Any


@py_class("mini.mlir.CallOp", structural_eq="dag")
class CallOp(Object):
    """``func.call("callee", a, b, ...)``."""

    __ffi_ir_traits__ = tr.CallTraits(
        "$field:callee", "$field:args", None, None, None, None,
    )
    callee: str
    args: List[Any]


# ============================================================================
# Bucket C — explicit factories + var-construction
# ============================================================================


def call(callee: str, *args: Any) -> CallOp:
    return CallOp(callee=callee, args=list(args))


__ffi_make_var__ = staticmethod(_make_value)


# ============================================================================
# The single finalize_module call — auto-injects:
#
# * ``func`` decorator handler (FuncTraits) with dialect-elevation
#   frame push.
# * ``ret`` hook (ReturnTraits).
# * ``__getattr__`` fallback for :class:`CallOp` — handles
#   ``func.<name>(args)`` lookups not covered by explicit factories.
# ============================================================================


finalize_module(__name__, prefix="func")
