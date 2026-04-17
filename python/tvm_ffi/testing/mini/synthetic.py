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
"""Mini-Synthetic — synthetic fixtures isolating individual printer features.

Naming convention: ``S<Mechanism>Node`` so ``rg "class S\\w+Node"`` finds
all synthetic fixtures at a glance. Language-module prefix is ``S`` to
match (e.g. ``S.manual(...)``, ``S.evaluate(...)``).
"""

# ruff: noqa: A003, D102, N802, UP006, UP045

from __future__ import annotations

from typing import Any, List, Optional  # noqa: UP035

from tvm_ffi import Object, pyast
from tvm_ffi import ir_traits as tr
from tvm_ffi.dataclasses import py_class
from tvm_ffi.dataclasses import field as dc_field


# ============================================================================
# Tier-1: __ffi_text_print__ override
# ============================================================================


@py_class("mini.synthetic.SManualPrintNode", structural_eq="dag")
class SManualPrintNode(Object):
    """Tier-1 fixture: ``__ffi_text_print__`` overrides everything else."""

    __ffi_ir_traits__ = tr.PrimTyTraits("$field:value")  # ignored
    value: str

    def __ffi_text_print__(self, ctx: Any) -> Any:  # noqa: ARG002
        # Render as a Call AST: S.manual(<value>!)
        return pyast.Call(
            func=pyast.Attr(obj=pyast.Id(name="S"), name="manual"),
            args=[pyast.Literal(value=self.value + "!")],
            kwargs=[],
        )


# ============================================================================
# Tier-2 indirection: $method targets
# ============================================================================


@py_class("mini.synthetic.SMethodCalleeNode", structural_eq="dag")
class SMethodCalleeNode(Object):
    """Exercises ``CallTraits.op = "$method:..."`` (callee resolved via method)."""

    __ffi_ir_traits__ = tr.CallTraits(
        "$method:resolved_name",
        "$field:args",
        None, None, None, None,
    )
    op_kind: str
    args: List[Any]

    def resolved_name(self) -> str:
        return f"S.dispatch_{self.op_kind}"


@py_class("mini.synthetic.SCallNode", structural_eq="dag")
class SCallNode(Object):
    """Iconic ``CallTraits`` — exercises **all six fields**."""

    __ffi_ir_traits__ = tr.CallTraits(
        "$field:base_callee",
        "$field:args",
        "$field:attrs",
        "$method:_extract_kwargs",
        "$method:_resolve_callee",
        "$method:_make_pre_hook",
    )
    base_callee: str
    args: List[Any]
    attrs: Optional[Any] = None
    extra_kwargs: Optional[Any] = None
    prefix: Optional[str] = None
    prologue_marker: Optional[str] = None

    def _resolve_callee(self) -> Any:
        if self.prefix is not None:
            return f"{self.prefix}.{self.base_callee}"
        return None

    def _extract_kwargs(self) -> Any:
        return self.extra_kwargs

    def _make_pre_hook(self) -> Any:
        """Return a closure that pushes a marker stmt before the call."""
        if self.prologue_marker is None:
            return None
        marker = self.prologue_marker

        def hook(obj: Any, printer: Any, frame: Any) -> None:
            frame.stmts.append(pyast.ExprStmt(pyast.Id(marker)))

        return hook


@py_class("mini.synthetic.SMethodKindNode", structural_eq="tree")
class SMethodKindNode(Object):
    """Exercises ``ForTraits.text_printer_kind = "$method:kind"``."""

    __ffi_ir_traits__ = tr.ForTraits(
        tr.RegionTraits("$field:body", "$field:loop_var", None, None),
        "$field:start", "$field:end", "$field:step",
        None, None, None,
        "$method:kind",
    )
    loop_var: Any = dc_field(structural_eq="def")
    start: Any
    end: Any
    step: Any
    body: List[Any]
    style: str = "default"

    def kind(self) -> str:
        return f"S.{self.style}"


# ============================================================================
# Default elision (None values disappear)
# ============================================================================


@py_class("mini.synthetic.SOptionalFieldNode", structural_eq="dag")
class SOptionalFieldNode(Object):
    """All trailing optionals — proves the printer elides None defaults.

    Print with all None: ``S.elide(req)``.
    Print with last set: ``S.elide(req, opt3=val)``.
    """

    __ffi_ir_traits__ = tr.CallTraits(
        "S.elide",
        "$field:args",
        "$field:kwargs",
        None, None, None,
    )
    args: List[Any]
    kwargs: Optional[Any] = None


# ============================================================================
# Sugar checking ($method:can_use_sugar)
# ============================================================================


@py_class("mini.synthetic.SSugarCheckNode", structural_eq="dag")
class SSugarCheckNode(Object):
    """Exercises ``BinOpTraits.text_printer_sugar_check = "$method:..."``."""

    __ffi_ir_traits__ = tr.BinOpTraits(
        "$field:lhs", "$field:rhs",
        "$field:op",
        "$method:allow_sugar",
        "$field:fallback",
    )
    lhs: Any
    rhs: Any
    op: str
    fallback: str
    allow_sugar: bool = True


# ============================================================================
# RegionTraits — region-only nodes (no FuncTraits wrapper)
# ============================================================================


@py_class("mini.synthetic.SBareRegionNode", structural_eq="tree")
class SBareRegionNode(Object):
    """A region not nested inside an If/For/With/Func — exercises the
    fallback printer behavior for raw RegionTraits."""

    __ffi_ir_traits__ = tr.RegionTraits(
        "$field:body", "$field:params", None, None,
    )
    params: List[Any] = dc_field(structural_eq="def", default_factory=list)
    body: List[Any] = dc_field(default_factory=list)


# ============================================================================
# Multiple def-sites (a, b = expr) — tuple unpack assign
# ============================================================================


@py_class("mini.synthetic.SMultiDefNode", structural_eq="tree")
class SMultiDefNode(Object):
    """``a, b = S.split(expr)`` — multiple LHS defs in a single AssignTraits."""

    __ffi_ir_traits__ = tr.AssignTraits(
        "$field:vars", "$field:value", None, None, None, None,
    )
    value: Any
    vars: List[Any] = dc_field(structural_eq="def")


# ============================================================================
# Custom AssignTraits.text_printer_kind on expr-stmt mode
# ============================================================================


@py_class("mini.synthetic.SExprStmtKindNode", structural_eq="tree")
class SExprStmtKindNode(Object):
    """``S.evaluate(value)`` — AssignTraits with no def + literal kind."""

    __ffi_ir_traits__ = tr.AssignTraits(
        None, "$field:value", None, None, "S.evaluate", None,
    )
    value: Any


# ============================================================================
# ReturnTraits — isolated coverage of the dedicated return statement trait
# ============================================================================


@py_class("mini.synthetic.SReturnNode", structural_eq="tree")
class SReturnNode(Object):
    """``return value`` — minimal ``ReturnTraits`` fixture."""

    __ffi_ir_traits__ = tr.ReturnTraits("$field:value")
    value: Any


# ============================================================================
# Language module
# ============================================================================


class SLang:
    """Mini-synthetic ``S`` language module."""

    @staticmethod
    def manual(value: str) -> SManualPrintNode:
        # Strip trailing ! the printer added.
        clean = value.rstrip("!")
        return SManualPrintNode(value=clean)

    @staticmethod
    def evaluate(value: Any) -> SExprStmtKindNode:
        return SExprStmtKindNode(value=value)

    @staticmethod
    def elide(*args: Any, **kwargs: Any) -> SOptionalFieldNode:
        return SOptionalFieldNode(
            args=list(args),
            kwargs=kwargs if kwargs else None,
        )

    @staticmethod
    def return_(value: Any) -> SReturnNode:
        """Construct an :class:`SReturnNode` (isolated ReturnTraits fixture)."""
        return SReturnNode(value=value)

    @staticmethod
    def call(
        base_callee: str,
        *args: Any,
        attrs: Optional[Any] = None,
        extra_kwargs: Optional[Any] = None,
        prefix: Optional[str] = None,
        prologue_marker: Optional[str] = None,
    ) -> SCallNode:
        """Construct an :class:`SCallNode` (iconic CallTraits fixture)."""
        return SCallNode(
            base_callee=base_callee,
            args=list(args),
            attrs=attrs,
            extra_kwargs=extra_kwargs,
            prefix=prefix,
            prologue_marker=prologue_marker,
        )


def _dispatch_factory(kind: str):
    def factory(*args: Any) -> SMethodCalleeNode:
        return SMethodCalleeNode(op_kind=kind, args=list(args))

    factory.__name__ = f"dispatch_{kind}"
    return staticmethod(factory)


for _kind in ("add", "sub", "mul"):
    setattr(SLang, f"dispatch_{_kind}", _dispatch_factory(_kind))


S = SLang()

LANG_MODULES: dict[str, Any] = {"S": S}
