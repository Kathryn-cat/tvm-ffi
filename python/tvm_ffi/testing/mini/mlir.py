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
"""Mini-MLIR — multi-dialect fixture for cross-dialect parser validation.

Six representative MLIR dialects wired to the trait-driven parser's
frame-based dispatch system (see
``design_docs/parser_frame_dispatch.md``):

* **``builtin``** — ``ModuleOp`` (class decorator).
* **``func``** — ``FuncOp`` (function decorator), ``ReturnOp``, ``CallOp``.
* **``arith``** — scalar types (``i1`` / ``i8`` / ``i32`` / ``i64`` /
  ``index`` / ``f16`` / ``f32`` / ``f64``), constant op, int/float
  binops. Owns ``+`` / ``-`` / ``*`` / ``<`` / ``==`` sugar with
  type-predicate dispatch between int (``AddIOp`` / …) and float
  (``AddFOp`` / …).
* **``scf``** — ``ForOp`` with ``scf.range(lb, ub, step)`` iter-holder.
* **``memref``** — ``MemRefType``, ``LoadOp``, ``StoreOp``.
* **``vector``** — ``VectorType``, ``AddOp``. Subscribes to the same
  sugar-op kinds as ``arith`` so ``a + b`` on vector operands
  fall-throughs arith and reaches vector.

The whole file is self-contained and intended as a stress test for
cross-dialect dispatch:

* Multi-dialect function signatures (``def f(a: arith.i32, b: memref.memref(...))``).
* Iter-type dispatch (``for i in scf.range(...)`` → scf regardless of
  the surrounding frame stack).
* Mixed call sites (``memref.store(arith.addi(a, b), A, [i])``).
* Type-predicate op fall-through (arith vs vector ``+``).
"""

# ruff: noqa: A003, D102, N802, UP006, UP045

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional  # noqa: UP035

from tvm_ffi import Object, pyast
from tvm_ffi import ir_traits as tr
from tvm_ffi.dataclasses import py_class
from tvm_ffi.dataclasses import field as dc_field


# ============================================================================
# Types
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


@py_class("mini.mlir.MemRefType", structural_eq="dag")
class MemRefType(Object):
    """``memref<2x3xf32>`` — prints as ``T.Buffer([2, 3], T.f32)``.

    ``shape`` is typed ``Any`` (not ``List[int]``) on purpose: the
    trait printer's ``$field:shape`` resolution for BufferTyTraits
    round-trips cleanly when the field accepts an FFI :class:`List`,
    but surfaces as ``ffi.List()`` (empty) when the field is a strict
    Python ``list`` annotation. See the equivalent field on
    :class:`mini.tir.BufferTy`.
    """

    __ffi_ir_traits__ = tr.BufferTyTraits(
        "$field:shape", "$field:elem_type", None, None, None,
    )
    shape: Any
    elem_type: Any


@py_class("mini.mlir.VectorType", structural_eq="dag")
class VectorType(Object):
    """``vector<4xf32>`` — prints as ``T.Tensor([4], T.f32)``."""

    __ffi_ir_traits__ = tr.TensorTyTraits(
        "$field:shape", "$field:elem_type", None,
    )
    shape: Any = None
    elem_type: Any = None


# ============================================================================
# SSA Values — shared across all dialects
# ============================================================================


@py_class("mini.mlir.Value", structural_eq="var")
class Value(Object):
    """SSA value. Every ``c = op(...)`` assignment introduces one."""

    __ffi_ir_traits__ = tr.ValueTraits("$field:name", "$field:ty", None)
    name: str = dc_field(structural_eq="ignore")
    ty: Any


# ============================================================================
# Arith dialect — constants + int/float binops
# ============================================================================


@py_class("mini.mlir.ConstantOp", structural_eq="dag")
class ConstantOp(Object):
    """``arith.constant(value, ty)`` — attribute-carrying scalar literal."""

    __ffi_ir_traits__ = tr.CallTraits(
        "arith.constant", "$field:args", None, None, None, None,
    )
    args: List[Any]


def _arith_binop_traits(op: str, func_name: str) -> tr.BinOpTraits:
    """BinOp trait factory for arith ops (infix sugar allowed)."""
    return tr.BinOpTraits("$field:lhs", "$field:rhs", op, None, func_name)


@py_class("mini.mlir.AddIOp", structural_eq="dag")
class AddIOp(Object):
    """``arith.addi(a, b)`` — integer addition."""

    __ffi_ir_traits__ = _arith_binop_traits("+", "arith.addi")
    lhs: Any
    rhs: Any


@py_class("mini.mlir.SubIOp", structural_eq="dag")
class SubIOp(Object):
    __ffi_ir_traits__ = _arith_binop_traits("-", "arith.subi")
    lhs: Any
    rhs: Any


@py_class("mini.mlir.MulIOp", structural_eq="dag")
class MulIOp(Object):
    __ffi_ir_traits__ = _arith_binop_traits("*", "arith.muli")
    lhs: Any
    rhs: Any


@py_class("mini.mlir.AddFOp", structural_eq="dag")
class AddFOp(Object):
    """``arith.addf(a, b)`` — float addition."""

    __ffi_ir_traits__ = _arith_binop_traits("+", "arith.addf")
    lhs: Any
    rhs: Any


@py_class("mini.mlir.SubFOp", structural_eq="dag")
class SubFOp(Object):
    __ffi_ir_traits__ = _arith_binop_traits("-", "arith.subf")
    lhs: Any
    rhs: Any


@py_class("mini.mlir.MulFOp", structural_eq="dag")
class MulFOp(Object):
    __ffi_ir_traits__ = _arith_binop_traits("*", "arith.mulf")
    lhs: Any
    rhs: Any


@py_class("mini.mlir.CmpIOp", structural_eq="dag")
class CmpIOp(Object):
    """``arith.cmpi(a, b)`` — integer comparison (defaults to ``<``)."""

    __ffi_ir_traits__ = _arith_binop_traits("<", "arith.cmpi")
    lhs: Any
    rhs: Any


# ============================================================================
# Memref dialect — buffer accesses
# ============================================================================


@py_class("mini.mlir.LoadOp", structural_eq="dag")
class LoadOp(Object):
    """``memref.load(ref, [indices...])`` — buffer read.

    The sugar ``A[i]`` round-trips via ``LoadTraits``.
    """

    __ffi_ir_traits__ = tr.LoadTraits("$field:ref", "$field:indices", None)
    ref: Value
    indices: List[Any]


@py_class("mini.mlir.StoreOp", structural_eq="tree")
class StoreOp(Object):
    """``memref.store(value, ref, [indices...])`` — buffer write.

    The sugar ``A[i] = v`` round-trips via ``StoreTraits``.
    """

    __ffi_ir_traits__ = tr.StoreTraits(
        "$field:ref", "$field:value", "$field:indices", None,
    )
    ref: Value
    value: Any
    indices: List[Any]


# ============================================================================
# Vector dialect — for Phase-6 op fall-through (arith vs vector ``+``)
# ============================================================================


@py_class("mini.mlir.VectorAddOp", structural_eq="dag")
class VectorAddOp(Object):
    """``vector.addf(a, b)`` — element-wise vector addition.

    Subscribes to the same ``+`` sugar kind as ``arith.AddIOp`` /
    ``AddFOp`` — ``visit_operation`` fall-through decides which one
    wins based on operand types.
    """

    __ffi_ir_traits__ = _arith_binop_traits("+", "vector.addf")
    lhs: Any
    rhs: Any


# ============================================================================
# Scf dialect — structured control flow
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
# Func dialect — function op, return, call
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
# Builtin dialect — module
# ============================================================================


@py_class("mini.mlir.ModuleOp", structural_eq="tree")
class ModuleOp(Object):
    """``@builtin.module class Name: <funcs>`` — module IR, class-form FuncTraits."""

    __ffi_ir_traits__ = tr.FuncTraits(
        "$field:name",
        tr.RegionTraits("$field:funcs", None, None, None),
        None, "builtin.module", None,
    )
    name: str = dc_field(structural_eq="ignore")
    funcs: List[Any]


# ============================================================================
# Bind IR — for ``c: ty = op(...)`` assignments
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
# Iter-holder for scf.range (__ffi_for_handler__ protocol)
# ============================================================================


@dataclass
class _ScfRange:
    """Iter object for ``for i in scf.range(...)``.

    Carries the bounds and defines the :meth:`__ffi_for_handler__`
    protocol method that :meth:`pyast.IRParser.visit_for` dispatches to.
    This is the key multi-dialect mechanism — no matter which dialect
    stack the surrounding function is in, ``for i in scf.range(...)``
    *always* builds a :class:`ScfForOp` because the iter object itself
    owns the handler.
    """

    lb: Any
    ub: Any
    step: Any

    def __ffi_for_handler__(self, parser, node) -> ScfForOp:
        from tvm_ffi.pyast_trait_parse import parse_value_def  # noqa: PLC0415

        if not isinstance(node.lhs, pyast.Id):
            raise NotImplementedError(
                "mini.mlir scf.for: only ``Id`` loop targets are supported",
            )
        with parser.scoped_frame(), parser.push_frame(pyast.ForFrame()):
            iv = parse_value_def(
                parser,
                node.lhs.name,
                annotation=None,
                make_var=_make_value,
                default_ty=IntegerType(name="index"),
            )
            body = parser.visit_body(node.body)
        return ScfForOp(
            iv=iv, lb=self.lb, ub=self.ub, step=self.step, body=body,
        )


# ============================================================================
# Shared ``__ffi_make_var__`` hook — all dialects that own a param type
# delegate to this common Value constructor.
# ============================================================================


def _make_value(parser: Any, name: str, ty: Any) -> Value:
    """Build a :class:`Value` with the given name and type.

    All mini-MLIR dialects that own param types (arith for scalars,
    memref for refs, vector for vectors) hang their ``__ffi_make_var__``
    off this one function — Value is the one SSA container shared
    across the entire mini-MLIR IR.
    """
    return Value(name=name, ty=ty)


# ============================================================================
# Operand-type introspection — used by op fall-through
# ============================================================================


def _operand_type(v: Any) -> Any:
    """Best-effort type extraction from a parser-side operand.

    Handles:
    * ``Value`` → its declared ``ty`` field.
    * raw Python ``int`` / ``float`` / ``bool`` → synthetic primitive
      type so type-predicate dispatch has something to match.
    * anything else → ``None`` (dispatcher will treat as "unknown").
    """
    if isinstance(v, Value):
        return v.ty
    if isinstance(v, ConstantOp) and v.args:
        # The typed literal: ConstantOp(args=[value, ty])
        return v.args[-1]
    if isinstance(v, bool):
        return IntegerType(name="i1")
    if isinstance(v, int):
        return IntegerType(name="i32")
    if isinstance(v, float):
        return FloatType(name="f32")
    return None


def _is_int_type(ty: Any) -> bool:
    return isinstance(ty, IntegerType)


def _is_float_type(ty: Any) -> bool:
    return isinstance(ty, FloatType)


def _is_vector_type(ty: Any) -> bool:
    return isinstance(ty, VectorType)


# ============================================================================
# Arith dialect module
# ============================================================================


class ArithLang:
    """``arith`` dialect — scalar types, constants, int/float binops.

    Owns every MLIR scalar type (``i1``…``index``, ``f16``/``f32``/``f64``)
    and the int/float overloads of the standard sugar ops. ``+`` / ``-``
    / ``*`` / ``<`` / ``==`` dispatch through
    :attr:`__ffi_op_classes__`; the handler routes to ``AddIOp`` /
    ``AddFOp`` etc. based on operand types, and returns ``None`` on
    types it doesn't own (e.g. vector) so ``visit_operation`` can
    fall through to the next dialect (Phase 6).
    """

    # ---- type attributes (plain PrimTy instances) ----
    i1 = IntegerType(name="i1")
    i8 = IntegerType(name="i8")
    i16 = IntegerType(name="i16")
    i32 = IntegerType(name="i32")
    i64 = IntegerType(name="i64")
    index = IntegerType(name="index")
    f16 = FloatType(name="f16")
    f32 = FloatType(name="f32")
    f64 = FloatType(name="f64")

    # ---- op constructors (de-sugared call form) ----
    @staticmethod
    def constant(value: Any, ty: Any) -> ConstantOp:
        return ConstantOp(args=[value, ty])

    @staticmethod
    def addi(a: Any, b: Any) -> AddIOp: return AddIOp(lhs=a, rhs=b)
    @staticmethod
    def subi(a: Any, b: Any) -> SubIOp: return SubIOp(lhs=a, rhs=b)
    @staticmethod
    def muli(a: Any, b: Any) -> MulIOp: return MulIOp(lhs=a, rhs=b)
    @staticmethod
    def addf(a: Any, b: Any) -> AddFOp: return AddFOp(lhs=a, rhs=b)
    @staticmethod
    def subf(a: Any, b: Any) -> SubFOp: return SubFOp(lhs=a, rhs=b)
    @staticmethod
    def mulf(a: Any, b: Any) -> MulFOp: return MulFOp(lhs=a, rhs=b)
    @staticmethod
    def cmpi(a: Any, b: Any) -> CmpIOp: return CmpIOp(lhs=a, rhs=b)

    # ---- Var-construction protocol ----
    __ffi_make_var__ = staticmethod(_make_value)

    # ---- default dtype handles used by parse_literal ----
    __ffi_default_int_ty__ = i32
    __ffi_default_float_ty__ = f32
    __ffi_default_bool_ty__ = i1

    # ---- sugar dispatch — int/float type-predicate, returns None on
    #      non-arith types (vector etc.) for op fall-through ----
    @staticmethod
    def _op_binary(
        parser: pyast.IRParser,
        node: pyast.Operation,
        int_cls: type,
        float_cls: type,
    ) -> Any:
        if len(node.operands) != 2:
            # Fold n-ary left-associatively — same as mini.tir's parse_binop.
            result = parser.eval_expr(node.operands[0])
            for raw in node.operands[1:]:
                rhs = parser.eval_expr(raw)
                inner = ArithLang._pick_binop_cls(result, rhs, int_cls, float_cls)
                if inner is None:
                    return None  # fall-through for any step
                result = inner(lhs=result, rhs=rhs)
            return result
        lhs = parser.eval_expr(node.operands[0])
        rhs = parser.eval_expr(node.operands[1])
        cls = ArithLang._pick_binop_cls(lhs, rhs, int_cls, float_cls)
        if cls is None:
            return None
        return cls(lhs=lhs, rhs=rhs)

    @staticmethod
    def _pick_binop_cls(
        lhs: Any, rhs: Any, int_cls: type, float_cls: type,
    ) -> Optional[type]:
        lhs_ty, rhs_ty = _operand_type(lhs), _operand_type(rhs)
        if _is_int_type(lhs_ty) and _is_int_type(rhs_ty):
            return int_cls
        if _is_float_type(lhs_ty) and _is_float_type(rhs_ty):
            return float_cls
        return None

    @staticmethod
    def _op_add(parser, node):
        return ArithLang._op_binary(parser, node, AddIOp, AddFOp)

    @staticmethod
    def _op_sub(parser, node):
        return ArithLang._op_binary(parser, node, SubIOp, SubFOp)

    @staticmethod
    def _op_mul(parser, node):
        return ArithLang._op_binary(parser, node, MulIOp, MulFOp)

    @staticmethod
    def _op_lt(parser, node):
        # Integer-only predicate for now.
        return ArithLang._op_binary(parser, node, CmpIOp, CmpIOp)

    @staticmethod
    def _op_eq(parser, node):
        return ArithLang._op_binary(parser, node, CmpIOp, CmpIOp)

    __ffi_op_classes__ = {
        pyast.OperationKind.Add: "arith._op_add",
        pyast.OperationKind.Sub: "arith._op_sub",
        pyast.OperationKind.Mult: "arith._op_mul",
        pyast.OperationKind.Lt: "arith._op_lt",
        pyast.OperationKind.Eq: "arith._op_eq",
    }


# ============================================================================
# Memref dialect module
# ============================================================================


def _memref_load_hook(parser, ref: Value, indices: list) -> LoadOp:
    """Subscript-load hook for ``A[i]`` — registered as ``load`` on the
    shared-hooks module so the parser's :meth:`visit_index` dispatches
    through it. Defined at module scope to avoid colliding with
    :meth:`MemRefLang.load_op` (a user-facing factory with a different
    signature)."""
    return LoadOp(ref=ref, indices=list(indices))


class MemRefLang:
    """``memref`` dialect — buffer types + load/store.

    Sugar:
    * ``A[i]`` → :class:`LoadOp` via the ``load`` hook (registered on
      :class:`_SharedHooks`, not here — attribute-name collision with
      :meth:`load_op` would break dispatch).
    * ``A[i] = v`` → :class:`StoreOp` via the ``__ffi_assign__`` hook.
    """

    # Parameterized type constructor
    @staticmethod
    def memref(shape: Any, elem_type: Any) -> MemRefType:
        return MemRefType(shape=list(shape), elem_type=elem_type)

    # Op constructors (user-facing factories — distinct names from the
    # ``load`` / ``store`` parser hooks to avoid signature collisions).
    @staticmethod
    def load_op(ref: Value, indices: List[Any]) -> LoadOp:
        return LoadOp(ref=ref, indices=list(indices))

    @staticmethod
    def store_op(value: Any, ref: Value, indices: List[Any]) -> StoreOp:
        return StoreOp(ref=ref, value=value, indices=list(indices))

    # Var-construction protocol for memref-typed params
    __ffi_make_var__ = staticmethod(_make_value)


# ============================================================================
# Vector dialect module — Phase-6 op fall-through target
# ============================================================================


class VectorLang:
    """``vector`` dialect — vector type + arithmetic that wins ``+``
    sugar over :class:`ArithLang` when operands are vector-typed.
    """

    @staticmethod
    def vector(shape: List[int], elem_type: Any) -> VectorType:
        return VectorType(shape=list(shape), elem_type=elem_type)

    @staticmethod
    def addf(a: Any, b: Any) -> VectorAddOp:
        return VectorAddOp(lhs=a, rhs=b)

    __ffi_make_var__ = staticmethod(_make_value)

    # Sugar fall-through: only accepts vector operands; returns None
    # otherwise so the parser keeps walking dialects.
    @staticmethod
    def _op_add(parser, node):
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
# Scf dialect module
# ============================================================================


class ScfLang:
    """``scf`` dialect — structured control flow.

    Exposes ``scf.range(lb, ub, step)`` which returns a :class:`_ScfRange`
    iter holder; ``visit_for`` dispatches through the holder's
    ``__ffi_for_handler__``.
    """

    @staticmethod
    def range(*args: Any, step: Any = None) -> _ScfRange:
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

    # ``if`` stmt hook → :class:`ScfIfOp`.
    @staticmethod
    def if_stmt(parser, cond, then_body: list, else_body: list) -> ScfIfOp:
        return ScfIfOp(
            cond=cond,
            then_body=then_body,
            else_body=else_body,
        )


# ============================================================================
# Func dialect module
# ============================================================================


class FuncLang:
    """``func`` dialect — the function decorator + ``func.return``/``func.call``."""

    @staticmethod
    def call(callee: str, *args: Any) -> CallOp:
        return CallOp(callee=callee, args=list(args))

    @staticmethod
    def func(parser, node) -> FuncOp:
        from tvm_ffi.pyast_trait_parse import parse_func  # noqa: PLC0415

        return parse_func(parser, node, FuncOp)

    # ``return v`` → func.ReturnOp.
    @staticmethod
    def ret(parser, value: Any) -> ReturnOp:
        return ReturnOp(value=value)

    __ffi_make_var__ = staticmethod(_make_value)


# ============================================================================
# Builtin dialect module
# ============================================================================


class BuiltinLang:
    """``builtin`` dialect — ``@builtin.module`` decorator."""

    @staticmethod
    def module(parser, node) -> ModuleOp:
        funcs: list = []
        with parser.scoped_frame():
            for stmt in node.body:
                if isinstance(stmt, pyast.Function):
                    funcs.append(parser.visit_function(stmt))
        return ModuleOp(name=node.name.name, funcs=funcs)


# ============================================================================
# Assign / bind hook — every dialect uses the same BindOp shape
# ============================================================================


def _assign_impl(parser, node: pyast.Assign) -> Any:
    """Dispatch ``pyast.Assign`` to store-style or bind-style IR."""
    from tvm_ffi.pyast_trait_parse import parse_assign, parse_store  # noqa: PLC0415

    if isinstance(node.lhs, pyast.Index):
        # ``A[i] = v`` → StoreOp. ``A``'s resolved type tells us which
        # dialect owns the store, but for mini-MLIR we always use memref.
        return parse_store(parser, node, StoreOp)
    return parse_assign(parser, node, BindOp)


# ============================================================================
# Binding the hooks onto the dialect modules that own them
# ============================================================================


# ``__ffi_assign__`` isn't dialect-specific — attach to every module
# that might own a param type so lookup always finds it. Simplest: one
# dedicated "shared" namespace at the end and include it in LANG_MODULES.
class _SharedHooks:
    """Hooks common to every dialect — load/assign/make_var/default-ty.

    Registered alongside the dialect modules in :data:`LANG_MODULES`.
    Separated from the per-dialect classes so a test can override a
    hook locally (via :meth:`IRParser.with_dialects`) without mutating
    the dialect singletons.
    """

    __ffi_assign__ = staticmethod(_assign_impl)
    __ffi_make_var__ = staticmethod(_make_value)
    __ffi_default_int_ty__ = ArithLang.i32
    __ffi_default_float_ty__ = ArithLang.f32
    __ffi_default_bool_ty__ = ArithLang.i1
    load = staticmethod(_memref_load_hook)


# ScfLang.if_stmt already defined above — wire as lang-module hook.
ScfLang.if_stmt_hook = ScfLang.if_stmt
# FuncLang.ret already defined — wire as ``ret`` hook.
# (The attribute name ``ret`` matches IRParser's ``visit_return`` lookup.)


# ============================================================================
# Language-module instances + parser registry
# ============================================================================


arith = ArithLang()
memref = MemRefLang()
vector = VectorLang()
scf = ScfLang()
func = FuncLang()
builtin = BuiltinLang()
_shared = _SharedHooks()


# ============================================================================
# The ``T`` type namespace — printer-side hardcoding requires us to
# resolve ``T.<type>`` for all type annotations.
#
# The trait printer in ``pyast_trait_print.cc`` hardcodes ``T.`` as the
# emission prefix for every type trait:
#
# * :class:`PrimTyTraits` → ``T.<name>``     (scalar: i32, f32, …)
# * :class:`BufferTyTraits` → ``T.Buffer(...)`` (memref, buffer-like)
# * :class:`TensorTyTraits` → ``T.Tensor(...)`` (vector, tensor-like)
#
# So mini-MLIR's ``arith.i32``, ``memref.memref([...], elem)``, and
# ``vector.vector([...], elem)`` all round-trip through ``T.i32``,
# ``T.Buffer(...)``, ``T.Tensor(...)``. Providing a unified ``T``
# namespace that forwards those attribute / call accesses to the
# owning dialect reconstructs the right IR class on parse.
#
# A cleaner long-term fix lives on the printer side — give each trait
# a configurable prefix. For now, ``T`` as a shared type-namespace is
# the minimal-invasion workaround for this cross-dialect validation
# suite.
# ============================================================================


class _TNamespace:
    """Unified ``T.`` type-namespace — forwards to the owning dialect.

    Reads like a dialect but has no behavior of its own beyond attribute
    forwarding: ``T.i32`` → :attr:`ArithLang.i32`; ``T.Buffer(...)`` →
    :meth:`MemRefLang.memref`; ``T.Tensor(...)`` → :meth:`VectorLang.vector`.
    """

    # Scalar types — re-exported from ``arith``.
    i1 = ArithLang.i1
    i8 = ArithLang.i8
    i16 = ArithLang.i16
    i32 = ArithLang.i32
    i64 = ArithLang.i64
    index = ArithLang.index
    f16 = ArithLang.f16
    f32 = ArithLang.f32
    f64 = ArithLang.f64

    # Parameterized type factories.
    Buffer = staticmethod(MemRefLang.memref)
    Tensor = staticmethod(VectorLang.vector)


T = _TNamespace()


# Order matters for the op fall-through: ``arith`` is consulted before
# ``vector`` for ``+``, so scalar operands go to arith and vector-typed
# operands naturally fall through arith's ``None`` return into vector.
LANG_MODULES: dict[str, Any] = {
    "T": T,  # printer-hardcoded type namespace
    "arith": arith,
    "memref": memref,
    "vector": vector,
    "scf": scf,
    "func": func,
    "builtin": builtin,
    # ``_shared`` has no prefix name of its own — register it under a
    # sentinel so :meth:`IRParser._lookup_hook` still finds its
    # attributes (``__ffi_assign__``, ``load``, etc.).
    "__shared__": _shared,
}


def make_var_factory(name: str, ty: Any) -> Value:
    """Legacy ``var_factory=`` shim for :class:`IRParser`."""
    return _make_value(None, name, ty)
