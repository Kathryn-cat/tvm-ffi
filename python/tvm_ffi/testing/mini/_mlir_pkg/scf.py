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
"""Mini-MLIR ``scf`` dialect — structured control flow.

Exposes ``scf.range(lb, ub, step)`` — an auto-wired
:class:`~tvm_ffi.pyast.ForFrame` factory. ``visit_for`` dispatches on
the frame directly; no ``__ffi_for_handler__`` protocol, no per-dialect
iter-holder dataclass. See ``design_docs/parser_for_handler_refactor.md``.
"""

# ruff: noqa: A003, D102, N802, UP006, UP045

from __future__ import annotations

from typing import Any, List  # noqa: UP035

from tvm_ffi import Object
from tvm_ffi import ir_traits as tr
from tvm_ffi.dataclasses import py_class
from tvm_ffi.dataclasses import field as dc_field
from tvm_ffi.dialect_autogen import finalize_module

from ._common import Value, _make_value
from .arith import IntegerType


# ============================================================================
# Ops
# ============================================================================


@py_class("mini.mlir.ScfForOp", structural_eq="tree")
class ScfForOp(Object):
    """``for i in scf.range(lb, ub, step): body`` — scf.for.

    ``_loop_var_ty`` is read by the auto-generated ``scf.range`` iter
    factory (built by :func:`finalize_module`) so the loop induction
    variable is typed as ``index`` regardless of the ambient
    ``__ffi_default_int_ty__``.
    """

    __ffi_ir_traits__ = tr.ForTraits(
        tr.RegionTraits("$field:body", "$field:iv", None, None),
        "$field:lb", "$field:ub", "$field:step",
        None, None, None, "scf.range",
    )
    iv: Value = dc_field(structural_eq="def")
    lb: Any
    ub: Any
    step: Any
    body: List[Any]

    _loop_var_ty = IntegerType(name="index")


@py_class("mini.mlir.ScfIfOp", structural_eq="tree")
class ScfIfOp(Object):
    """``scf.if`` — covered by Python ``if`` syntax via the ``if_stmt`` hook."""

    __ffi_ir_traits__ = tr.IfTraits(
        "$field:cond",
        tr.RegionTraits("$field:then_body", None, None, None),
        tr.RegionTraits("$field:else_body", None, None, None),
    )
    cond: Any
    then_body: List[Any]
    else_body: List[Any] = dc_field(default_factory=list)


# ============================================================================
# Bucket C — var-construction hook
# ============================================================================


__ffi_make_var__ = staticmethod(_make_value)


# ============================================================================
# The single finalize_module call — auto-injects:
#
# * ``if_stmt`` hook (IfTraits) → builds :class:`ScfIfOp`.
# * ``scf.range`` iter factory (from ``iter_kinds=["range"]``) returning
#   a :class:`~tvm_ffi.pyast.ForFrame` carrying ``for_cls=ScfForOp`` and
#   ``loop_var_ty=IntegerType("index")``.
# ============================================================================


finalize_module(
    __name__,
    prefix="scf",
    iter_kinds=["range"],
)
