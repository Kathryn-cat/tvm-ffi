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
"""Mini-TIR — TIR-flavored fixtures for trait validation."""

# ruff: noqa: A003, D102, N802, UP006, UP045

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Union  # noqa: UP035

from tvm_ffi import Object, pyast, register_global_func
from tvm_ffi import dtype as ffi_dtype
from tvm_ffi import ir_traits as tr
from tvm_ffi.dataclasses import py_class
from tvm_ffi.dataclasses import field as dc_field


# ============================================================================
# Types
# ============================================================================


@py_class("mini.tir.PrimTy", structural_eq="dag")
class PrimTy(Object):
    """Scalar primitive type — prints as ``T.<dtype>``."""

    __ffi_ir_traits__ = tr.PrimTyTraits("$field:dtype")
    dtype: str


@py_class("mini.tir.BufferTy", structural_eq="dag")
class BufferTy(Object):
    """Buffer type — ``T.Buffer((shape...), dtype, ...)`` with default elision."""

    __ffi_ir_traits__ = tr.BufferTyTraits(
        "$field:shape",
        "$field:dtype",
        "$field:strides",
        "$field:offset",
        "$field:scope",
    )
    shape: Any
    dtype: str
    strides: Optional[List[int]] = None
    offset: Optional[int] = None
    scope: Optional[str] = None


# ============================================================================
# Expressions
# ============================================================================


@py_class("mini.tir.Var", structural_eq="var")
class Var(Object):
    """Typed scalar — ``ValueTraits`` (use site = name; def site = name: ty)."""

    __ffi_ir_traits__ = tr.ValueTraits("$field:name", "$field:ty", None)
    name: str = dc_field(structural_eq="ignore")
    ty: PrimTy


@py_class("mini.tir.IntImm", structural_eq="dag")
class IntImm(Object):
    """Integer literal — ``LiteralTraits(format="int")``."""

    __ffi_ir_traits__ = tr.LiteralTraits("$field:value", "int")
    value: int
    dtype: Any


@py_class("mini.tir.FloatImm", structural_eq="dag")
class FloatImm(Object):
    """Float literal — ``LiteralTraits(format="float")``."""

    __ffi_ir_traits__ = tr.LiteralTraits("$field:value", "float")
    value: float
    dtype: Any


def _no_const_fold(lhs: Any, rhs: Any) -> bool:
    """Sugar-check: refuse infix sugar when both operands are literals."""
    return not (
        isinstance(lhs, (IntImm, FloatImm)) and isinstance(rhs, (IntImm, FloatImm))
    )


@register_global_func("mini.tir._binop_sugar_check")
def _binop_sugar_check_global(_printer: Any, obj: Any) -> bool:
    return _no_const_fold(obj.lhs, obj.rhs)


def _binop_traits(op: str, func_name: str) -> tr.BinOpTraits:
    """BinOp trait factory mirroring TIR's ``.def_ir_traits<BinOpTraitsObj>``."""
    return tr.BinOpTraits(
        "$field:lhs",
        "$field:rhs",
        op,
        "$global:mini.tir._binop_sugar_check",
        func_name,
    )


@py_class("mini.tir.Add", structural_eq="dag")
class Add(Object):
    """``a + b`` — refuses sugar when both operands are literals."""

    __ffi_ir_traits__ = _binop_traits("+", "T.Add")
    lhs: Any
    rhs: Any


@py_class("mini.tir.Sub", structural_eq="dag")
class Sub(Object):
    """``a - b`` — refuses sugar when both operands are literals."""

    __ffi_ir_traits__ = _binop_traits("-", "T.Sub")
    lhs: Any
    rhs: Any


@py_class("mini.tir.Mul", structural_eq="dag")
class Mul(Object):
    """``a * b`` — refuses sugar when both operands are literals."""

    __ffi_ir_traits__ = _binop_traits("*", "T.Mul")
    lhs: Any
    rhs: Any


@py_class("mini.tir.Lt", structural_eq="dag")
class Lt(Object):
    """``a < b`` — refuses sugar when both operands are literals."""

    __ffi_ir_traits__ = _binop_traits("<", "T.Lt")
    lhs: Any
    rhs: Any


@py_class("mini.tir.Eq", structural_eq="dag")
class Eq(Object):
    """``a == b`` — refuses sugar when both operands are literals."""

    __ffi_ir_traits__ = _binop_traits("==", "T.Eq")
    lhs: Any
    rhs: Any


@py_class("mini.tir.And", structural_eq="dag")
class And(Object):
    """``a and b`` — boolean conjunction (Python keyword)."""

    __ffi_ir_traits__ = _binop_traits("and", "T.And")
    lhs: Any
    rhs: Any


@py_class("mini.tir.Or", structural_eq="dag")
class Or(Object):
    """``a or b`` — boolean disjunction (Python keyword)."""

    __ffi_ir_traits__ = _binop_traits("or", "T.Or")
    lhs: Any
    rhs: Any


@py_class("mini.tir.Not", structural_eq="dag")
class Not(Object):
    """Boolean negation — ``not a``."""

    __ffi_ir_traits__ = tr.UnaryOpTraits("$field:a", "not")
    a: Any


@py_class("mini.tir.Call", structural_eq="dag")
class Call(Object):
    """Call with literal callee: ``T.<op_name>(args...)``."""

    __ffi_ir_traits__ = tr.CallTraits(
        "$field:op_name",
        "$field:args",
        None, None, None, None,
    )
    op_name: str
    args: List[Any]


@py_class("mini.tir.BufferLoad", structural_eq="dag")
class BufferLoad(Object):
    """``buffer[indices]`` — ``LoadTraits``."""

    __ffi_ir_traits__ = tr.LoadTraits("$field:source", "$field:indices", None)
    source: Var
    indices: List[Any]


@py_class("mini.tir.Cast", structural_eq="dag")
class Cast(Object):
    """Level 0 fixture — no trait, prints as ``mini.tir.Cast(...)``."""

    target: PrimTy
    value: Any


# ============================================================================
# Statements
# ============================================================================


@py_class("mini.tir.Bind", structural_eq="tree")
class Bind(Object):
    """Local binding ``var: ty = value`` — ``AssignTraits``."""

    __ffi_ir_traits__ = tr.AssignTraits(
        "$field:var", "$field:value", None, None, None, None,
    )
    value: Any
    var: Var = dc_field(structural_eq="def")


@py_class("mini.tir.BufferStore", structural_eq="tree")
class BufferStore(Object):
    """``buffer[indices] = value`` — ``StoreTraits``."""

    __ffi_ir_traits__ = tr.StoreTraits(
        "$field:buffer", "$field:value", "$field:indices", None,
    )
    buffer: Var
    value: Any
    indices: List[Any]


def _evaluate_is_ret_call(obj: Any) -> bool:
    """Shared helper: detect ``Call(op_name="ret", args=[single])`` shape."""
    return isinstance(obj.value, Call) and obj.value.op_name == "ret" and len(obj.value.args) == 1


@register_global_func("mini.tir._evaluate_expr")
def _evaluate_expr_global(_printer: Any, obj: Any) -> Any:
    """Global ``rhs`` resolver for :class:`Evaluate`."""
    if _evaluate_is_ret_call(obj):
        return obj.value.args[0]
    return obj.value


@register_global_func("mini.tir._evaluate_kind")
def _evaluate_kind_global(_printer: Any, obj: Any) -> Any:
    """Global ``text_printer_kind`` resolver for :class:`Evaluate`."""
    if _evaluate_is_ret_call(obj):
        return None
    return "T.evaluate"


@register_global_func("mini.tir._evaluate_is_return")
def _evaluate_is_return_global(_printer: Any, obj: Any) -> bool:
    """Global ``text_printer_return_check`` resolver for :class:`Evaluate`."""
    return _evaluate_is_ret_call(obj)


@py_class("mini.tir.Evaluate", structural_eq="tree")
class Evaluate(Object):
    """Expression-statement — ``AssignTraits`` with dynamic kind + return check.

    Support two cases:

    1. ``value = Call(ret, [result])`` → ``return result`` (return inversion)
    2. ``value = non-call expr`` → ``T.evaluate(expr)`` (kind wrap)
    """

    __ffi_ir_traits__ = tr.AssignTraits(
        None,
        "$global:mini.tir._evaluate_expr",
        None,
        None,
        "$global:mini.tir._evaluate_kind",
        "$global:mini.tir._evaluate_is_return",
    )
    value: Any


@py_class("mini.tir.IfThenElse", structural_eq="tree")
class IfThenElse(Object):
    """``if cond: ... else: ...`` — ``IfTraits`` with both regions."""

    __ffi_ir_traits__ = tr.IfTraits(
        "$field:cond",
        tr.RegionTraits("$field:then_body", None, None, None),
        tr.RegionTraits("$field:else_body", None, None, None),
    )
    cond: Any
    then_body: List[Any]
    else_body: List[Any] = dc_field(default_factory=list)


@py_class("mini.tir.While", structural_eq="tree")
class While(Object):
    """``while cond: body`` — ``WhileTraits``."""

    __ffi_ir_traits__ = tr.WhileTraits(
        "$field:cond",
        tr.RegionTraits("$field:body", None, None, None),
    )
    cond: Any
    body: List[Any]


@py_class("mini.tir.AssertStmt", structural_eq="tree")
class AssertStmt(Object):
    """``assert cond, msg`` — ``AssertTraits``."""

    __ffi_ir_traits__ = tr.AssertTraits("$field:cond", "$field:message")
    cond: Any
    message: Optional[str] = None


# NOTE: There is no separate ReturnStmt in mini.tir — TIR-faithful design.
# ``return x`` is represented as ``Evaluate(Call(op_name="ret", args=[x]))``
# (see :class:`Evaluate` above). For ``ReturnTraits`` coverage as an isolated
# trait fixture, see :class:`mini.synthetic.SReturnNode`.


@register_global_func("mini.tir._for_kind_prefix")
def _for_kind_prefix_global(_printer: Any, obj: Any) -> str:
    """Global ``text_printer_kind`` resolver for :class:`For`."""
    return f"T.{obj.kind}"


@py_class("mini.tir.For", structural_eq="tree")
class For(Object):
    """``for i in T.<kind>(start, end, step, annotations={...}): body``."""

    __ffi_ir_traits__ = tr.ForTraits(
        tr.RegionTraits("$field:body", "$field:loop_var", None, None),
        "$field:start",
        "$field:end",
        "$field:step",
        None,
        None,
        "$field:annotations",
        "$global:mini.tir._for_kind_prefix",
    )
    loop_var: Var = dc_field(structural_eq="def")
    start: Any
    end: Any
    step: Any
    body: List[Any]
    kind: str = "serial"
    annotations: Optional[Any] = None


@py_class("mini.tir.Block", structural_eq="tree")
class Block(Object):
    """``with T.block(): body`` — ``WithTraits`` with literal kind."""

    __ffi_ir_traits__ = tr.WithTraits(
        tr.RegionTraits("$field:body", None, None, None),
        None, None,
        "T.block",
        None, None, None,
    )
    body: List[Any]


@py_class("mini.tir.SeqStmt", structural_eq="tree")
class SeqStmt(Object):
    """Inline sequence — ``WithTraits`` with ``text_printer_no_frame=True``."""

    __ffi_ir_traits__ = tr.WithTraits(
        tr.RegionTraits("$field:stmts", None, None, None),
        None, None, None, None, None, True,
    )
    stmts: List[Any]


# ============================================================================
# Functions / modules
# ============================================================================


@py_class("mini.tir.PrimFunc", structural_eq="tree")
class PrimFunc(Object):
    """``@T.prim_func\\ndef name(params): body`` — minimal (no prologue hook)."""

    __ffi_ir_traits__ = tr.FuncTraits(
        "$field:name",
        tr.RegionTraits("$field:body", "$field:params", None, None),
        None, "T.prim_func", None,
    )
    name: str = dc_field(structural_eq="ignore")
    params: List[Var] = dc_field(structural_eq="def")
    body: List[Any]


@py_class("mini.tir.IRModule", structural_eq="tree")
class IRModule(Object):
    """``@I.ir_module\\nclass Name: <funcs>`` — FuncTraits class form."""

    __ffi_ir_traits__ = tr.FuncTraits(
        "$field:name",
        tr.RegionTraits("$field:funcs", None, None, None),
        None, "I.ir_module", None,
    )
    name: str = dc_field(structural_eq="ignore")
    funcs: List[Any]


# ============================================================================
# Surface objects (For-iter holders)
# ============================================================================


@dataclass
class _IterHolder:
    """Returned by ``T.serial(...)`` etc. — consumed by the for_stmt hook."""

    kind: str
    start: Any
    end: Any
    step: Any
    annotations: Optional[Any] = None


@dataclass
class _BlockMarker:
    """Returned by ``T.block()`` — consumed by the with_stmt hook."""


# ============================================================================
# Helpers — IR-construction shortcuts (NOT parser-side)
# ============================================================================


def imm(value: Union[int, float], dtype: Any) -> Any:  # noqa: UP007
    """``imm(42, "int32") -> IntImm``; ``imm(3.14, "f32") -> FloatImm``."""
    dt = ffi_dtype(dtype) if isinstance(dtype, str) else dtype
    if isinstance(value, bool) or isinstance(value, int):
        return IntImm(value=int(value), dtype=dt)
    return FloatImm(value=float(value), dtype=dt)


# ============================================================================
# T language module (everything callable from printed text)
# ============================================================================


_INT_DTYPES = ("int8", "int16", "int32", "int64",
                "uint8", "uint16", "uint32", "uint64", "bool")
_FLOAT_DTYPES = ("float16", "float32", "float64")


class TLang:
    """Mini-TIR ``T`` language module.

    Dtype attributes (``T.int32``, ``T.float32``, ``T.bool`` …) are
    plain :class:`PrimTy` *instances*, not factories. Under the new
    language-module protocol ``eval_expr(T.int32)`` must resolve to the
    type value directly (mirroring how ``PrimTy`` prints from the IR
    side); the wrapped ``T.int64(42)`` literal form is delegated to
    :meth:`PrimTy.__call__` so one attribute serves both roles.
    """

    # ---- BufferTy factory (T.Buffer(shape, dtype, ...)) ----
    @staticmethod
    def Buffer(
        shape: Any,
        dtype: str,
        *,
        strides: Optional[Sequence[int]] = None,
        elem_offset: Optional[int] = None,
        scope: Optional[str] = None,
    ) -> BufferTy:
        if isinstance(shape, int):
            shape = [shape]
        return BufferTy(
            shape=list(shape),
            dtype=dtype,
            strides=list(strides) if strides is not None else None,
            offset=elem_offset,
            scope=scope,
        )

    # ---- Generic Call fallback (T.<unknown_name>(args) -> Call IR) ----
    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)

        def _opaque_call_factory(*args: Any) -> Call:
            return Call(op_name=f"T.{name}", args=list(args))

        _opaque_call_factory.__name__ = f"T.{name}"
        return _opaque_call_factory


def _parse_type(
    primty: PrimTy,
    value: Any = None,
    var_name: str | None = None,
) -> Any:
    """Dispatcher for ``T.<dtype>(...)`` calls (bound via :class:`_DtypeHandle`)."""
    if value is not None and var_name is not None:
        raise TypeError(
            f"T.{primty.dtype}(...): cannot pass both ``value`` and "
            f"``var_name``. Use ``value=`` for literal construction "
            f"(IntImm/FloatImm) or ``var_name=`` for Var construction.",
        )
    if value is not None:
        dt_str = str(primty.dtype)
        if dt_str in ("float16", "float32", "float64"):
            return FloatImm(value=float(value), dtype=ffi_dtype(dt_str))
        return IntImm(value=int(value), dtype=ffi_dtype(dt_str))
    name = var_name if var_name is not None else "_"
    return TLang.__ffi_make_var__(None, name, primty)


@py_class("mini.tir._DtypeHandle", structural_eq="dag")
class _DtypeHandle(PrimTy):
    """Callable :class:`PrimTy` subclass — the concrete type of
    ``T.int32`` / ``T.float32`` / …
    """

    def __call__(self, value: Any = None, *, var_name: str | None = None) -> Any:
        return _parse_type(self, value=value, var_name=var_name)


# ---- Mount each ``T.<dtype>`` as a _DtypeHandle instance. ----
for _dt in _INT_DTYPES + _FLOAT_DTYPES:
    setattr(TLang, _dt, _DtypeHandle(dtype=_dt))


# ---- Default dtype handles for bare-literal wrapping ----
TLang.__ffi_default_int_ty__ = TLang.int32
TLang.__ffi_default_float_ty__ = TLang.float32
TLang.__ffi_default_bool_ty__ = TLang.bool


# ---- For-iter factories (T.serial / parallel / unroll / vectorized) ----
def _make_iter_factory(kind: str):
    def factory(*args: Any, step: Any = None, annotations: Any = None) -> _IterHolder:
        if len(args) == 1:
            start_v, end_v = 0, args[0]
        elif len(args) == 2:
            start_v, end_v = args[0], args[1]
        elif len(args) == 3:
            start_v, end_v, step_pos = args
            if step is not None and step != step_pos:
                raise TypeError(f"T.{kind}: positional and kw step disagree")
            step = step_pos
        else:
            raise TypeError(f"T.{kind} expects 1/2/3 positional args, got {len(args)}")
        return _IterHolder(
            kind=kind,
            start=start_v,
            end=end_v,
            step=1 if step is None else step,
            annotations=annotations,
        )

    factory.__name__ = kind
    return staticmethod(factory)


for _kind in ("serial", "parallel", "unroll", "vectorized"):
    setattr(TLang, _kind, _make_iter_factory(_kind))


# ---- Single-method factories ----
@staticmethod
def _evaluate_factory(value: Any) -> Evaluate:
    return Evaluate(value=value)


@staticmethod
def _block_factory() -> _BlockMarker:
    return _BlockMarker()


TLang.evaluate = _evaluate_factory
TLang.block = _block_factory


# ---- Operation registry — single source of truth ----
# Both derive from the single declaration ``_OP_KIND_TO_IR_CLASS`` —
# adding a new op only requires extending this one dict.
_OP_KIND_TO_IR_CLASS: dict[int, type] = {
    # Binary
    pyast.OperationKind.Add: Add,
    pyast.OperationKind.Sub: Sub,
    pyast.OperationKind.Mult: Mul,
    pyast.OperationKind.Lt: Lt,
    pyast.OperationKind.Eq: Eq,
    pyast.OperationKind.And: And,
    pyast.OperationKind.Or: Or,
    # Unary
    pyast.OperationKind.Not: Not,
}


def _wire_op_classes() -> None:
    """Wire ``__ffi_op_classes__`` and ``T.<Name>`` parse functions."""
    from functools import partial  # noqa: PLC0415

    from tvm_ffi.pyast_trait_parse import parse_binop, parse_unaryop  # noqa: PLC0415

    op_classes_map: dict[int, str] = {}
    for kind, cls in _OP_KIND_TO_IR_CLASS.items():
        op_classes_map[kind] = f"T.{cls.__name__}"
        arity = 1 if kind < pyast.OperationKind._UnaryEnd else 2
        parse_fn = parse_unaryop if arity == 1 else parse_binop
        setattr(TLang, cls.__name__, staticmethod(partial(parse_fn, ir_class=cls)))
    TLang.__ffi_op_classes__ = op_classes_map


_wire_op_classes()


# ============================================================================
# Parser hooks (called by IRParser)
# ============================================================================


def _bind_hook(parser, var: Var, rhs: Any) -> Bind:
    return Bind(var=var, value=rhs)


def _buffer_store_hook(parser, target, indices: list, value: Any) -> BufferStore:
    return BufferStore(buffer=target, value=value, indices=indices)


def _if_stmt_hook(parser, cond, then_body: list, else_body: list) -> IfThenElse:
    return IfThenElse(cond=cond, then_body=then_body, else_body=else_body)


def _while_stmt_hook(parser, cond, body: list) -> While:
    return While(cond=cond, body=body)


def _assert_stmt_hook(parser, cond, msg) -> AssertStmt:
    msg_str = msg if (msg is None or isinstance(msg, str)) else str(msg)
    return AssertStmt(cond=cond, message=msg_str)


def _for_stmt_hook(parser, node, iter_val: Any) -> For:
    if not isinstance(node.lhs, pyast.Id):
        raise NotImplementedError("Only Id loop targets supported")
    annotations: Optional[Any] = None
    if isinstance(iter_val, _IterHolder):
        kind, start, end, step = (
            iter_val.kind, iter_val.start, iter_val.end, iter_val.step,
        )
        annotations = iter_val.annotations
    elif isinstance(iter_val, range):
        kind = "serial"
        start, end, step = iter_val.start, iter_val.stop, iter_val.step
    else:
        raise TypeError(
            f"Unsupported for-iter: {type(iter_val).__name__}",
        )
    parser.push_scope()
    try:
        loop_var = parser.make_var(node.lhs.name, None)
        parser.define(node.lhs.name, loop_var)
        body = parser.visit_body(node.body)
    finally:
        parser.pop_scope()
    return For(
        loop_var=loop_var,
        start=start,
        end=end,
        step=step,
        body=body,
        kind=kind,
        annotations=annotations,
    )


def _with_stmt_hook(parser, node, ctx) -> Any:
    parser.push_scope()
    try:
        body = parser.visit_body(node.body)
    finally:
        parser.pop_scope()
    if isinstance(ctx, _BlockMarker):
        return Block(body=body)
    raise TypeError(f"Unsupported with-context: {type(ctx).__name__}")


def _prim_func_handler(parser, node) -> PrimFunc:
    from tvm_ffi.pyast_trait_parse import parse_func  # noqa: PLC0415
    return parse_func(parser, node, PrimFunc)


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


# Canonical Var constructor — the ``__ffi_make_var__`` protocol.
def _make_var_impl(parser: Any, name: str, ty: Any) -> Var:
    """Build a mini-TIR ``Var`` from a resolved ``(name, ty)`` pair."""
    if isinstance(ty, PrimTy):
        return Var(name=name, ty=PrimTy(dtype=ty.dtype))
    if isinstance(ty, Var):
        return Var(name=name, ty=PrimTy(dtype=ty.ty.dtype))
    return Var(name=name, ty=PrimTy(dtype="int32"))


# Canonical assign hook — the ``__ffi_assign__`` protocol.
def _assign_impl(parser: Any, node: pyast.Assign) -> Any:
    """Dispatch a :class:`pyast.Assign` to the right trait-driven parser."""
    from tvm_ffi.pyast_trait_parse import (  # noqa: PLC0415
        parse_assign,
        parse_store,
    )

    if isinstance(node.lhs, pyast.Index):
        return parse_store(parser, node, BufferStore)
    return parse_assign(parser, node, Bind)


# ---- Wire all hooks onto TLang. ----
TLang.__ffi_make_var__ = staticmethod(_make_var_impl)
TLang.__ffi_assign__ = staticmethod(_assign_impl)
TLang.bind = staticmethod(_bind_hook)
TLang.buffer_store = staticmethod(_buffer_store_hook)
TLang.if_stmt = staticmethod(_if_stmt_hook)
TLang.while_stmt = staticmethod(_while_stmt_hook)
TLang.assert_stmt = staticmethod(_assert_stmt_hook)
TLang.for_stmt = staticmethod(_for_stmt_hook)
TLang.with_stmt = staticmethod(_with_stmt_hook)
TLang.prim_func = staticmethod(_prim_func_handler)


# ============================================================================
# I language module (module-level decorators)
# ============================================================================


class ILang:
    """Mini-TIR ``I`` language module — for ``@I.ir_module``."""

    ir_module = staticmethod(_ir_module_handler)


T = TLang()
I = ILang()  # noqa: E741

LANG_MODULES: dict[str, Any] = {"T": T, "I": I}


def make_var_factory(name: str, ty: Any) -> Var:
    """Legacy ``var_factory=`` shim for :class:`~tvm_ffi.pyast.IRParser`."""
    return _make_var_impl(None, name, ty)
