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

Exposes ``scf.range(lb, ub, step)`` which returns a :class:`_ScfRange`
iter holder (auto-wired ``__ffi_for_handler__``); ``visit_for``
dispatches through the holder's handler.
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

from ._common import Value, _make_value
from .arith import IntegerType


# ============================================================================
# Ops
# ============================================================================


@py_class("mini.mlir.ScfForOp", structural_eq="tree")
class ScfForOp(Object):
    """``for i in scf.range(lb, ub, step): body`` — scf.for."""

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
# Bucket C — iter-holder dataclass for ``scf.range(...)``
#
# ``finalize_module`` injects ``__ffi_for_handler__`` via
# ``iter_holder=_ScfRange`` + ``iter_kinds=["range"]``.
# ============================================================================


@dataclass
class _ScfRange:
    """Iter object for ``for i in scf.range(...)``.

    The ``_loop_var_ty`` class attribute signals to the auto-wired
    :meth:`__ffi_for_handler__` that the loop induction variable is
    typed as ``index`` regardless of ambient ``__ffi_default_int_ty__``
    — mirrors MLIR's scf.for convention.
    """

    lb: Any
    ub: Any
    step: Any

    _loop_var_ty = IntegerType(name="index")


# ============================================================================
# Bucket C — var-construction hook + range factory
#
# ``range`` is a builtin we override; the auto-wiring would normally
# produce it from ``iter_kinds=["range"]`` but we define it explicitly
# to keep the signature shape identical across call sites.
# ============================================================================


__ffi_make_var__ = staticmethod(_make_value)


def range(*args: Any, step: Any = None) -> _ScfRange:  # noqa: A001
    """``scf.range(...)`` — iter-holder factory supporting 1/2/3 positional args."""
    if len(args) == 1:
        lb, ub = 0, args[0]
    elif len(args) == 2:
        lb, ub = args
    elif len(args) == 3:
        lb, ub, step_pos = args
        if step is not None and step != step_pos:
            raise TypeError("scf.range: positional and kw step disagree")
        step = step_pos
    else:
        raise TypeError(
            f"scf.range: expected 1/2/3 positional args, got {len(args)}",
        )
    return _ScfRange(lb=lb, ub=ub, step=1 if step is None else step)


# ============================================================================
# The single finalize_module call — auto-injects:
#
# * ``if_stmt`` hook (IfTraits) → builds :class:`ScfIfOp`.
# * ``__ffi_for_handler__`` on :class:`_ScfRange` (ForTraits) — uses
#   the ``iter_holder=`` config to attach the handler.
# * Does NOT auto-generate ``range`` — we defined it above explicitly
#   so the ``hasattr`` guard skips the generic factory generator. Our
#   explicit ``range`` has the same shape as the auto-generated one.
# ============================================================================


finalize_module(
    __name__,
    prefix="scf",
    iter_kinds=["range"],
    iter_holder=_ScfRange,
)
