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
"""Parser auto-registration — "the module is the dialect".

See ``design_docs/parser_auto_registration.md`` for the full design.

Short version: a dialect author writes only ``@py_class`` declarations
on their IR classes (plus any genuinely-custom hooks), and adds one
``finalize_module(__name__, ...)`` call at the bottom of their .py
file. This module walks the Python module's ``@py_class`` IR classes,
reads each ``__ffi_ir_traits__``, and auto-injects:

* Factory attributes (``T.Add``, ``T.prim_func``, ``T.serial``, …)
* Parser hooks (``bind``, ``buffer_store``, ``load``, ``if_stmt``,
  ``while_stmt``, ``assert_stmt``, ``ret``, ``for_stmt``, ``with_stmt``,
  ``__ffi_assign__``, ``__ffi_make_var__``, ``__ffi_op_classes__``, …)
* Dtype handles and default-literal-type hooks

directly onto the Python module so ``import tvm.script.tirx as T``
plus ``parser = IRParser(lang_modules={"T": T})`` "just works."

Precedence: **user explicit assignment wins** — every wiring rule
uses ``if not hasattr(module, name)`` guards. If the user already
defined ``module.ret`` or ``module.int32`` before calling
``finalize_module``, the auto-wiring skips those names.

Frame semantics follow §4.5 of the design doc — Category A pushes
(dialect elevation on ``@T.prim_func`` / ``@R.function`` body entry)
are emitted inside the auto-generated decorator handlers; Category C
pushes (``ForFrame(kind=...)``, ``WithFrame()``) are emitted inside
auto-generated iter-handler / ctx-handler methods; Category B marker
frames are the parser core's responsibility and this module doesn't
touch them.
"""

from __future__ import annotations

import inspect
import sys
from typing import TYPE_CHECKING, Any, Callable, Optional

from tvm_ffi import pyast
from tvm_ffi import ir_traits as tr

if TYPE_CHECKING:
    from types import ModuleType


# ============================================================================
# Trait-kind → wiring-rule registry
# ============================================================================


_WIRING_RULES: dict[type, Callable] = {}


def _wiring_rule(trait_cls: type) -> Callable:
    """Decorator: register ``fn`` as the wiring rule for ``trait_cls``."""

    def _decorator(fn: Callable) -> Callable:
        _WIRING_RULES[trait_cls] = fn
        return fn

    return _decorator


# ============================================================================
# Utility: trait-ref helpers
# ============================================================================


def _strip_field_prefix(ref: Any) -> Optional[str]:
    """``"$field:name"`` → ``"name"``; returns ``None`` if ``ref`` isn't
    a ``$field:`` ref (may be ``None``, a literal, a ``$method:`` ref,
    etc.)."""
    if isinstance(ref, str) and ref.startswith("$field:"):
        return ref[len("$field:"):]
    return None


def _literal_prefix_name(ref: Optional[str]) -> Optional[tuple[str, str]]:
    """``"T.prim_func"`` → ``("T", "prim_func")``; returns ``None`` when
    ``ref`` is ``None`` or doesn't have exactly one ``.`` separator."""
    if not isinstance(ref, str) or "." not in ref or ref.startswith("$"):
        return None
    parts = ref.split(".", 1)
    if len(parts) != 2:
        return None
    return parts[0], parts[1]


# ============================================================================
# Primitive-wrapping helper (used by auto-generated op factories to lift
# raw Python primitives into Imm IRs via the module's default-ty hooks)
# ============================================================================


def _wrap_primitive_via_module(value: Any, module: "ModuleType") -> Any:
    """Wrap ``value`` using ``module.__ffi_default_{bool,int,float}_ty__``
    handles when available. If the module hasn't declared defaults (or
    the value is already an ffi Object), pass through unchanged."""
    from tvm_ffi import Object  # noqa: PLC0415

    if value is None or isinstance(value, Object):
        return value
    if isinstance(value, bool):
        handle = getattr(module, "__ffi_default_bool_ty__", None)
        if handle is not None and callable(handle):
            return handle(int(value))
        return value
    if isinstance(value, int):
        handle = getattr(module, "__ffi_default_int_ty__", None)
        if handle is not None and callable(handle):
            return handle(int(value))
        return value
    if isinstance(value, float):
        handle = getattr(module, "__ffi_default_float_ty__", None)
        if handle is not None and callable(handle):
            return handle(float(value))
        return value
    return value


# ============================================================================
# OperationKind symbol → kind integer table (for auto op_classes map)
# ============================================================================


_BINOP_SYMBOL_TO_KIND: dict[str, int] = {
    "+": pyast.OperationKind.Add,
    "-": pyast.OperationKind.Sub,
    "*": pyast.OperationKind.Mult,
    "/": pyast.OperationKind.Div,
    "//": pyast.OperationKind.FloorDiv,
    "%": pyast.OperationKind.Mod,
    "**": pyast.OperationKind.Pow,
    "<<": pyast.OperationKind.LShift,
    ">>": pyast.OperationKind.RShift,
    "&": pyast.OperationKind.BitAnd,
    "|": pyast.OperationKind.BitOr,
    "^": pyast.OperationKind.BitXor,
    "<": pyast.OperationKind.Lt,
    "<=": pyast.OperationKind.LtE,
    "==": pyast.OperationKind.Eq,
    "!=": pyast.OperationKind.NotEq,
    ">": pyast.OperationKind.Gt,
    ">=": pyast.OperationKind.GtE,
    "and": pyast.OperationKind.And,
    "or": pyast.OperationKind.Or,
}

_UNARYOP_SYMBOL_TO_KIND: dict[str, int] = {
    "-": pyast.OperationKind.USub,
    "+": pyast.OperationKind.UAdd,
    "~": pyast.OperationKind.Invert,
    "not": pyast.OperationKind.Not,
}


# ============================================================================
# Wiring rules — one per trait kind
# ============================================================================


@_wiring_rule(tr.BinOpTraits)
def _wire_binop(module: "ModuleType", cls: type, trait: Any, ctx: dict) -> None:
    """Attach sugar-path factory to the IR class + register
    ``__ffi_op_classes__`` entry.

    The factory is set as a class attribute ``cls._ffi_parse_op``
    (name chosen to avoid collisions with user class attrs). The
    ``__ffi_op_classes__`` entry points ``OperationKind.X`` →
    ``"<prefix>.<Cls>._ffi_parse_op"``; ``visit_operation`` resolves
    the dotted path at parse time.

    The IR class stays at ``module.<Cls>`` (unchanged from
    ``@py_class`` registration), so ``isinstance(x, mt.Add)`` + keyword
    construction ``mt.Add(lhs=a, rhs=b)`` both work. De-sugared call
    form ``T.Add(a, b)`` at parse time invokes the class constructor
    positionally (py_class supports this), matching the orig
    construction contract.
    """
    from tvm_ffi.pyast_trait_parse import parse_binop  # noqa: PLC0415

    func_name = trait.text_printer_func_name

    def _make_sugar_factory(_cls: type = cls) -> Callable:
        def factory(parser: Any, op_node: Any) -> Any:
            return parse_binop(parser, op_node, ir_class=_cls)

        factory.__name__ = f"_ffi_parse_op_{_cls.__name__}"
        return factory

    if not hasattr(cls, "_ffi_parse_op"):
        cls._ffi_parse_op = staticmethod(_make_sugar_factory())

    op_kind = _BINOP_SYMBOL_TO_KIND.get(trait.op)
    if op_kind is not None:
        prefix = ctx.get("prefix", "T")
        ctx["op_classes_map"].setdefault(
            op_kind, f"{prefix}.{cls.__name__}._ffi_parse_op",
        )


@_wiring_rule(tr.UnaryOpTraits)
def _wire_unaryop(module: "ModuleType", cls: type, trait: Any, ctx: dict) -> None:
    """Unary-op analog of :func:`_wire_binop` — sugar factory on
    ``cls._ffi_parse_op``, op-classes entry points there."""
    from tvm_ffi.pyast_trait_parse import parse_unaryop  # noqa: PLC0415

    def _make_sugar_factory(_cls: type = cls) -> Callable:
        def factory(parser: Any, op_node: Any) -> Any:
            return parse_unaryop(parser, op_node, ir_class=_cls)

        factory.__name__ = f"_ffi_parse_op_{_cls.__name__}"
        return factory

    if not hasattr(cls, "_ffi_parse_op"):
        cls._ffi_parse_op = staticmethod(_make_sugar_factory())

    op_kind = _UNARYOP_SYMBOL_TO_KIND.get(trait.op)
    if op_kind is not None:
        prefix = ctx.get("prefix", "T")
        ctx["op_classes_map"].setdefault(
            op_kind, f"{prefix}.{cls.__name__}._ffi_parse_op",
        )


@_wiring_rule(tr.LoadTraits)
def _wire_load(module: "ModuleType", cls: type, trait: Any, ctx: dict) -> None:
    """Register the ``load`` hook → build a ``LoadTraits`` IR from
    ``A[indices]`` subscripts."""
    source_field = _strip_field_prefix(trait.source) or "source"
    indices_field = _strip_field_prefix(trait.indices) or "indices"

    def _load_hook(_parser: Any, obj: Any, indices: list) -> Any:
        wrapped = [_wrap_primitive_via_module(i, module) for i in indices]
        return cls(**{source_field: obj, indices_field: wrapped})

    if not hasattr(module, "load"):
        module.load = _load_hook  # type: ignore[attr-defined]


@_wiring_rule(tr.AssignTraits)
def _wire_assign(module: "ModuleType", cls: type, trait: Any, ctx: dict) -> None:
    """Dual-mode wiring for ``AssignTraits``:

    * ``def_values`` set (normal bind ``v = expr``): register as the
      ``bind`` hook — contributes to the ``__ffi_assign__`` router.
    * ``def_values=None`` with literal ``text_printer_kind`` (expr-stmt
      like ``T.evaluate(x)``): register a call factory under that name
      — ``visit_call`` resolves to it when parsing the print form.
    """
    if trait.def_values is None:
        # Expression-statement mode: register a call factory under
        # ``text_printer_kind`` name (e.g. ``T.evaluate``).
        parts = _literal_prefix_name(trait.text_printer_kind)
        if not parts:
            return  # dynamic kind — can't auto-name
        attr_name = parts[1]
        rhs_field = _strip_field_prefix(trait.rhs) or "value"

        def _exprstmt_factory(*args: Any) -> Any:
            if len(args) != 1:
                raise TypeError(
                    f"{attr_name}: expected 1 arg, got {len(args)}",
                )
            return cls(**{rhs_field: _wrap_primitive_via_module(args[0], module)})

        _exprstmt_factory.__name__ = attr_name
        if not hasattr(module, attr_name):
            setattr(module, attr_name, _exprstmt_factory)
        return

    # Normal bind: ``v = expr`` → cls(var=v, value=expr).
    var_field = _strip_field_prefix(trait.def_values) or "var"
    rhs_field = _strip_field_prefix(trait.rhs) or "value"

    def _bind_hook(_parser: Any, var: Any, rhs: Any) -> Any:
        return cls(**{var_field: var, rhs_field: rhs})

    if not hasattr(module, "bind"):
        module.bind = _bind_hook  # type: ignore[attr-defined]

    ctx.setdefault("_has_assign_bind", cls)


@_wiring_rule(tr.StoreTraits)
def _wire_store(module: "ModuleType", cls: type, trait: Any, ctx: dict) -> None:
    """Register the ``buffer_store`` hook → build a ``StoreTraits`` IR
    from subscript-target assignments (``A[i] = v``).

    Contributes to the per-module ``__ffi_assign__`` router only when
    the same module also has a bind class (``AssignTraits``). Without a
    bind class, installing a partial router would shadow a cross-dialect
    ``__shared__.__ffi_assign__`` via :meth:`_lookup_hook` — so we just
    register the ``buffer_store`` hook and leave routing to the host.
    """
    target_field = _strip_field_prefix(trait.target) or "target"
    value_field = _strip_field_prefix(trait.value) or "value"
    indices_field = _strip_field_prefix(trait.indices) if trait.indices else None

    def _buffer_store_hook(
        _parser: Any, target: Any, indices: list, value: Any,
    ) -> Any:
        kwargs: dict[str, Any] = {
            target_field: target,
            value_field: _wrap_primitive_via_module(value, module),
        }
        if indices_field is not None:
            kwargs[indices_field] = [
                _wrap_primitive_via_module(i, module) for i in indices
            ]
        return cls(**kwargs)

    if not hasattr(module, "buffer_store"):
        module.buffer_store = _buffer_store_hook  # type: ignore[attr-defined]

    ctx.setdefault("_has_assign_store", cls)


def _ensure_assign_router(module: "ModuleType", ctx: dict) -> None:
    """Install the ``__ffi_assign__`` router once at least one of
    bind/store has been wired on this module.

    Routes Id-lhs to :func:`parse_assign` with the bind class; Index-lhs
    to :func:`parse_store` with the store class. When one side is
    missing on this module, the router delegates to other active
    dialects' ``__ffi_assign__`` hooks — this keeps cross-dialect
    compositions like mini.mlir's ``_SharedHooks`` (full router) +
    per-dialect partial router (e.g. memref has only store) from
    shadowing each other.
    """
    store_cls = ctx.get("_has_assign_store")
    bind_cls = ctx.get("_has_assign_bind")
    if store_cls is None and bind_cls is None:
        return

    def _router(parser: Any, node: Any,
                _store: Optional[type] = store_cls,
                _bind: Optional[type] = bind_cls,
                _this_module: "ModuleType" = module) -> Any:
        from tvm_ffi.pyast_trait_parse import parse_assign, parse_store  # noqa: PLC0415

        is_index = isinstance(node.lhs, pyast.Index)
        if is_index and _store is not None:
            return parse_store(parser, node, _store)
        if not is_index and _bind is not None:
            return parse_assign(parser, node, _bind)
        # Partial router — delegate to another dialect that provides
        # the missing half.
        for dialect in parser.active_dialects():
            if dialect is _this_module:
                continue
            other = getattr(dialect, "__ffi_assign__", None)
            if other is not None:
                return other(parser, node)
        raise NotImplementedError(
            f"{_this_module.__name__}: partial ``__ffi_assign__`` router "
            f"cannot handle {'Index' if is_index else 'Id'}-lhs and no "
            "other active dialect provides a fallback.",
        )

    if not hasattr(module, "__ffi_assign__"):
        module.__ffi_assign__ = _router  # type: ignore[attr-defined]


@_wiring_rule(tr.ReturnTraits)
def _wire_return(module: "ModuleType", cls: type, trait: Any, ctx: dict) -> None:
    """Register the ``ret`` hook → build a ``ReturnTraits`` IR from
    ``return expr``."""
    value_field = _strip_field_prefix(trait.value) or "value"

    def _ret_hook(_parser: Any, value: Any) -> Any:
        return cls(**{value_field: _wrap_primitive_via_module(value, module)})

    if not hasattr(module, "ret"):
        module.ret = _ret_hook  # type: ignore[attr-defined]


@_wiring_rule(tr.IfTraits)
def _wire_if(module: "ModuleType", cls: type, trait: Any, ctx: dict) -> None:
    """Register the ``if_stmt`` hook → build an ``IfTraits`` IR.

    No frame push here: ``pyast.IRParser.visit_if`` already pushes
    :class:`pyast.IfFrame` around each branch before calling the hook.
    """
    cond_field = _strip_field_prefix(trait.cond) or "cond"
    then_field = _strip_field_prefix(trait.then_region.body) or "then_body"
    else_field: Optional[str] = None
    if trait.else_region is not None:
        else_field = _strip_field_prefix(trait.else_region.body) or "else_body"

    def _if_stmt_hook(
        _parser: Any, cond: Any, then_body: list, else_body: list,
    ) -> Any:
        kwargs: dict[str, Any] = {cond_field: cond, then_field: then_body}
        if else_field is not None:
            kwargs[else_field] = else_body
        return cls(**kwargs)

    if not hasattr(module, "if_stmt"):
        module.if_stmt = _if_stmt_hook  # type: ignore[attr-defined]


@_wiring_rule(tr.WhileTraits)
def _wire_while(module: "ModuleType", cls: type, trait: Any, ctx: dict) -> None:
    """Register the ``while_stmt`` hook → build a ``WhileTraits`` IR."""
    cond_field = _strip_field_prefix(trait.cond) or "cond"
    body_field = _strip_field_prefix(trait.region.body) or "body"

    def _while_hook(_parser: Any, cond: Any, body: list) -> Any:
        return cls(**{cond_field: cond, body_field: body})

    if not hasattr(module, "while_stmt"):
        module.while_stmt = _while_hook  # type: ignore[attr-defined]


@_wiring_rule(tr.AssertTraits)
def _wire_assert(module: "ModuleType", cls: type, trait: Any, ctx: dict) -> None:
    """Register the ``assert_stmt`` hook → build an ``AssertTraits`` IR."""
    cond_field = _strip_field_prefix(trait.cond) or "cond"
    msg_field: Optional[str] = None
    if trait.message is not None:
        msg_field = _strip_field_prefix(trait.message) or "message"

    def _assert_hook(_parser: Any, cond: Any, msg: Any) -> Any:
        kwargs: dict[str, Any] = {cond_field: cond}
        if msg_field is not None:
            kwargs[msg_field] = msg if (msg is None or isinstance(msg, str)) else str(msg)
        return cls(**kwargs)

    if not hasattr(module, "assert_stmt"):
        module.assert_stmt = _assert_hook  # type: ignore[attr-defined]


@_wiring_rule(tr.FuncTraits)
def _wire_func(module: "ModuleType", cls: type, trait: Any, ctx: dict) -> None:
    """Register the function/class decorator handler.

    Emits a **Category-A frame push** (``Frame(dialects=[module])``)
    around the body parse — elevates this dialect to the innermost
    position in the dispatch stack for the duration of the function
    body. This is the crux of cross-dialect: ``@T.prim_func`` pushes
    ``T``, ``@R.function`` pushes ``R``, per-function hook lookup
    resolves accordingly.

    Detection heuristic (function-decorator vs class-decorator):
    the class-form (``@I.ir_module class M:``) typically has no
    ``params`` declared in ``region.def_values``; the function-form
    has ``region.def_values = "$field:params"``. We use that signal.
    """
    from tvm_ffi.pyast_trait_parse import parse_func  # noqa: PLC0415

    kind = trait.text_printer_kind
    parts = _literal_prefix_name(kind)
    if not parts:
        return  # Dynamic or missing kind — caller wires manually.
    _prefix, handler_name = parts

    # Heuristic for class-form: no def_values in region (empty params).
    is_class_form = trait.region.def_values is None

    body_field = _strip_field_prefix(trait.region.body) or "body"
    name_field = _strip_field_prefix(trait.symbol) or "name"

    def _func_handler(parser: Any, node: Any) -> Any:
        with parser.push_frame(pyast.Frame(dialects=[module])):
            return parse_func(parser, node, cls)

    def _class_handler(parser: Any, node: Any) -> Any:
        funcs: list = []
        with parser.push_frame(pyast.Frame(dialects=[module])), \
                parser.scoped_frame():
            for stmt in node.body:
                if isinstance(stmt, pyast.Function):
                    funcs.append(parser.visit_function(stmt))
        return cls(**{name_field: node.name.name, body_field: funcs})

    handler = _class_handler if is_class_form else _func_handler
    if not hasattr(module, handler_name):
        setattr(module, handler_name, handler)


@_wiring_rule(tr.ForTraits)
def _wire_for(module: "ModuleType", cls: type, trait: Any, ctx: dict) -> None:
    """Register ``__ffi_for_handler__`` on the iter-holder dataclass +
    per-kind iter factories on the module.

    Emits a **Category-C frame push** (``ForFrame(kind=self.kind)``)
    inside the auto-generated handler so cross-cutting code can detect
    "am I inside a for-loop of kind K?" via ``isinstance(f, ForFrame)``
    plus ``f.kind``.

    Requires ``iter_holder=...`` and ``iter_kinds=[...]`` config in
    :func:`finalize_module`. Silently skips when either is missing.
    """
    from tvm_ffi.pyast_trait_parse import parse_value_def  # noqa: PLC0415

    iter_holder = ctx.get("iter_holder")
    iter_kinds = ctx.get("iter_kinds") or []
    if iter_holder is None or not iter_kinds:
        return

    loop_var_field = _strip_field_prefix(trait.region.def_values) or "loop_var"
    body_field = _strip_field_prefix(trait.region.body) or "body"
    start_field = _strip_field_prefix(trait.start) if trait.start else None
    end_field = _strip_field_prefix(trait.end) if trait.end else None
    step_field = _strip_field_prefix(trait.step) if trait.step else None
    ann_field = _strip_field_prefix(trait.attrs) if trait.attrs else None

    def _make_handler(_cls: type = cls) -> Callable:
        def _handler(self_holder: Any, parser: Any, node: Any) -> Any:
            if not isinstance(node.lhs, pyast.Id):
                raise NotImplementedError(
                    "auto-generated For handler: only ``Id`` loop targets supported",
                )
            # Loop-var type resolution priority:
            # 1. Holder-class attribute ``_loop_var_ty`` (scf.for uses
            #    ``IntegerType(name="index")`` regardless of ambient
            #    default-int-ty).
            # 2. Module's own ``__ffi_default_int_ty__``.
            # 3. Cross-dialect ``_lookup_hook("__ffi_default_int_ty__")``
            #    — picks up e.g. ``arith.i32`` for a scf frame that
            #    doesn't own its own scalar defaults.
            default_int_ty = (
                getattr(type(self_holder), "_loop_var_ty", None)
                or getattr(module, "__ffi_default_int_ty__", None)
                or parser._lookup_hook("__ffi_default_int_ty__")
            )
            make_var = (
                getattr(module, "__ffi_make_var__", None)
                or parser._lookup_hook("__ffi_make_var__")
            )
            with parser.scoped_frame(), parser.push_frame(
                pyast.ForFrame(kind=getattr(self_holder, "kind", None)),
            ):
                loop_var = parse_value_def(
                    parser,
                    node.lhs.name,
                    annotation=None,
                    make_var=make_var,
                    default_ty=default_int_ty,
                )
                body = parser.visit_body(node.body)
            kwargs: dict[str, Any] = {
                loop_var_field: loop_var,
                body_field: body,
            }
            # Pull bounds off the holder using the IR class's own field
            # names first (holders that mirror the IR class field names
            # are the common case — e.g. mini.mlir's ``_ScfRange(lb, ub,
            # step)`` maps directly onto ``ScfForOp(lb, ub, step)``).
            # Fall back to the canonical ``start``/``end``/``step`` names
            # for holders like mini.tir's ``_IterHolder(start, end, step)``
            # that used the canonical vocabulary.
            if start_field:
                kwargs[start_field] = getattr(
                    self_holder, start_field,
                    getattr(self_holder, "start", None),
                )
            if end_field:
                kwargs[end_field] = getattr(
                    self_holder, end_field,
                    getattr(self_holder, "end", None),
                )
            if step_field:
                kwargs[step_field] = getattr(
                    self_holder, step_field,
                    getattr(self_holder, "step", None),
                )
            if ann_field:
                kwargs[ann_field] = getattr(
                    self_holder, ann_field,
                    getattr(self_holder, "annotations", None),
                )
            if hasattr(self_holder, "kind") and "kind" in _class_field_names(_cls):
                kwargs["kind"] = self_holder.kind
            return _cls(**kwargs)

        return _handler

    if not hasattr(iter_holder, "__ffi_for_handler__"):
        iter_holder.__ffi_for_handler__ = _make_handler()

    for kind_name in iter_kinds:
        if not hasattr(module, kind_name):
            setattr(module, kind_name, _make_iter_factory(kind_name, iter_holder))

    # Plain Python ``range`` fallback — assume the first iter_kind is
    # the default (``serial`` for TIR).
    if iter_kinds and not hasattr(module, "for_stmt"):
        default_kind = iter_kinds[0]
        holder_cls = iter_holder

        def _for_stmt_fallback(parser: Any, node: Any, iter_val: Any) -> Any:
            if isinstance(iter_val, holder_cls):
                return iter_val.__ffi_for_handler__(parser, node)
            if isinstance(iter_val, range):
                rng = holder_cls(
                    kind=default_kind,
                    start=iter_val.start,
                    end=iter_val.stop,
                    step=iter_val.step,
                )
                return rng.__ffi_for_handler__(parser, node)
            raise TypeError(
                f"Unsupported for-iter: {type(iter_val).__name__}",
            )

        module.for_stmt = _for_stmt_fallback  # type: ignore[attr-defined]


@_wiring_rule(tr.WithTraits)
def _wire_with(module: "ModuleType", cls: type, trait: Any, ctx: dict) -> None:
    """Register ``__ffi_with_handler__`` on the ctx-marker class + a
    ``with_stmt`` fallback hook on the module.

    Emits a **Category-C frame push** (``WithFrame()``) inside the
    auto-generated handler. Requires either:

    * a user-provided marker class registered via ``with_marker=...``
      on :func:`finalize_module` — the handler is attached to it, OR
    * a literal ``text_printer_kind`` (e.g. ``"T.block"``) — an
      auto-generated marker class is created and a factory registered
      on the module.

    Skips when neither condition is met (e.g. ``no_frame=True`` SeqStmt
    fixtures which are inherently lossy on roundtrip).
    """
    body_field = _strip_field_prefix(trait.region.body) or "body"
    parts = _literal_prefix_name(trait.text_printer_kind)
    # SeqStmt-like: no_frame=True — transparent, no handler.
    if getattr(trait, "text_printer_no_frame", False):
        return

    marker_cls = ctx.get("with_marker")
    factory_name = parts[1] if parts else None

    # If no user-provided marker and no literal kind, skip.
    if marker_cls is None and factory_name is None:
        return

    def _make_handler(_cls: type = cls) -> Callable:
        def _handler(self_marker: Any, parser: Any, node: Any) -> Any:
            with parser.scoped_frame(), parser.push_frame(pyast.WithFrame()):
                body = parser.visit_body(node.body)
            return _cls(**{body_field: body})

        return _handler

    # Case 1: user supplied a marker class via ``with_marker=``.
    # Auto-install the handler only if the user didn't set one already.
    if marker_cls is not None and not hasattr(marker_cls, "__ffi_with_handler__"):
        marker_cls.__ffi_with_handler__ = _make_handler()

    # Case 2: literal kind → register the ``<name>()`` factory on the
    # module. When the user supplied a marker class, the factory must
    # return an instance of *that* class so its (possibly user-provided)
    # ``__ffi_with_handler__`` runs. Fall back to a synthetic class only
    # when no marker class was given.
    if factory_name is not None:
        if marker_cls is not None:
            target_cls = marker_cls
        else:
            target_cls = type(f"_{_cls_name_safe(cls)}Marker", (), {})
            target_cls.__ffi_with_handler__ = _make_handler()

        def _ctx_factory(_mcls: type = target_cls) -> Any:
            return _mcls()

        _ctx_factory.__name__ = factory_name
        if not hasattr(module, factory_name):
            setattr(module, factory_name, _ctx_factory)


@_wiring_rule(tr.CallTraits)
def _wire_call(module: "ModuleType", cls: type, trait: Any, ctx: dict) -> None:
    """Record the CallTraits class in ``ctx`` for the end-of-finalize
    ``__getattr__`` install.

    Only registers a fallback when the trait carries a ``$field:`` callee
    (generic ``<prefix>.<name>(args)`` shape). A literal callee
    (e.g. ``"R.call_tir"`` on :class:`CallTIR`) means the class is
    a fixed-shape op — not a generic catch-all — so it's skipped.

    The actual install happens in :func:`finalize_module` after every
    other rule has run — installing it earlier would shadow ``hasattr``
    checks in downstream rules (``hasattr(module, "prim_func")`` etc.)
    and cause them to skip their ``setattr`` registrations.
    """
    callee_field = _strip_field_prefix(trait.op)
    if callee_field is None:
        return  # Literal callee — fixed-shape op, no opaque fallback.
    ctx["_call_fallback_cls"] = cls
    ctx["_call_fallback_callee_field"] = callee_field
    ctx["_call_fallback_args_field"] = (
        _strip_field_prefix(trait.args) or "args"
    )


# NOTE: LiteralTraits / PrimTyTraits / ValueTraits /
# BufferTyTraits / TensorTyTraits are intentionally NOT in the auto-wiring
# table — they describe IR types the user already reaches via the class
# name (e.g. ``module.Add``). There's nothing to wire beyond the class
# registration that ``@py_class`` already handles.


# ============================================================================
# Helpers used by wiring rules
# ============================================================================


def _class_field_names(cls: type) -> set[str]:
    """Return the declared field names on a py_class IR class."""
    info = getattr(cls, "__tvm_ffi_type_info__", None)
    if info is None:
        return set()
    # Attribute access on the instance works; fall back to annotations.
    ann = getattr(cls, "__annotations__", {})
    return set(ann.keys())


def _cls_name_safe(cls: type) -> str:
    """Make a type-name string safe for use as a synthetic class name."""
    return "".join(c if c.isalnum() else "_" for c in cls.__name__)


def _make_iter_factory(kind: str, iter_holder: type) -> Callable:
    """Build a ``T.<kind>(lb, ub, step, annotations=...)`` factory."""

    def factory(*args: Any, step: Any = None, annotations: Any = None) -> Any:
        if len(args) == 1:
            start_v, end_v = 0, args[0]
        elif len(args) == 2:
            start_v, end_v = args
        elif len(args) == 3:
            start_v, end_v, step_pos = args
            if step is not None and step != step_pos:
                raise TypeError(f"{kind}: positional and kw step disagree")
            step = step_pos
        else:
            raise TypeError(
                f"{kind}: expected 1/2/3 positional args, got {len(args)}",
            )
        return iter_holder(
            kind=kind,
            start=start_v,
            end=end_v,
            step=1 if step is None else step,
            annotations=annotations,
        )

    factory.__name__ = kind
    return factory


# ============================================================================
# Top-level entry: finalize_module
# ============================================================================


def finalize_module(
    module_name: str,
    *,
    prefix: Optional[str] = None,
    iter_kinds: Optional[list[str]] = None,
    dtypes: Optional[list[str]] = None,
    default_dtypes: Optional[dict[str, str]] = None,
    iter_holder: Optional[type] = None,
    with_marker: Optional[type] = None,
) -> None:
    """Scan the module's ``@py_class`` IR classes and auto-inject
    lang-module attributes derived from ``__ffi_ir_traits__``.

    See ``design_docs/parser_auto_registration.md`` (§4) for the full
    spec. Called once at module load, at the bottom of the dialect's
    .py file.

    Parameters
    ----------
    module_name
        Typically ``__name__`` — the dialect module to scan and mutate.
    prefix
        Short prefix used when constructing ``__ffi_op_classes__``
        entries for unary ops (which have no ``text_printer_func_name``).
        Defaults to the last component of ``module_name`` (e.g.
        ``"tir"`` from ``"tvm_ffi.testing.mini.tir"``).
    iter_kinds
        Iter-kind strings (e.g. ``["serial", "parallel", …]``) to
        register as ``module.<kind>(lb, ub, step)`` factories. Requires
        ``iter_holder``.
    dtypes
        Dtype names (e.g. ``["int32", "float32"]``). Auto-registered as
        plain attribute instances on the module via the user-supplied
        dtype-handle class (if present via ``module._DtypeHandle``) or
        the :class:`PrimTy` class discovered in the module.
    default_dtypes
        Maps literal category → dtype name. Produces
        ``__ffi_default_{int,float,bool}_ty__`` attrs pointing at the
        corresponding ``module.<dtype>`` entries.
    iter_holder
        Dataclass used as the runtime value for ``T.serial(...)`` etc.
        ``finalize_module`` auto-injects ``__ffi_for_handler__`` on it.
    with_marker
        Optional class used as the runtime value for a custom
        ``with``-construct — ``__ffi_with_handler__`` will be injected.
    """
    module = sys.modules[module_name]
    if prefix is None:
        prefix = module_name.rsplit(".", 1)[-1]

    # Discover IR classes belonging to this module (or its subpackages).
    ir_classes: list[type] = []
    for _, cls in inspect.getmembers(module):
        if not (isinstance(cls, type) and hasattr(cls, "__ffi_ir_traits__")):
            continue
        cls_mod = getattr(cls, "__module__", None)
        if cls_mod is None:
            continue
        if cls_mod == module_name or cls_mod.startswith(module_name + "."):
            ir_classes.append(cls)

    # Per-class wiring context — accumulates state across rules.
    ctx: dict[str, Any] = {
        "op_classes_map": {},
        "iter_holder": iter_holder,
        "iter_kinds": iter_kinds or [],
        "with_marker": with_marker,
        "prefix": prefix,
    }

    # Apply wiring rules.
    for cls in ir_classes:
        trait = cls.__ffi_ir_traits__
        rule = _WIRING_RULES.get(type(trait))
        if rule is not None:
            rule(module, cls, trait, ctx)

    # Mount composite attrs accumulated by wiring rules.
    if ctx["op_classes_map"] and not hasattr(module, "__ffi_op_classes__"):
        module.__ffi_op_classes__ = dict(ctx["op_classes_map"])

    # Install the ``__ffi_assign__`` router AFTER every rule has run —
    # deferring here lets both bind and store classes accumulate in
    # ``ctx`` before the router is built (otherwise the first-run
    # ``hasattr`` guard would lock in the partial router early).
    _ensure_assign_router(module, ctx)

    # Dtype handles — plain PrimTy instances unless a user-provided
    # dtype-handle class is present on the module.
    if dtypes:
        handle_cls = getattr(module, "_DtypeHandle", None)
        prim_ty_cls = _find_class_with_trait(module, tr.PrimTyTraits)
        if handle_cls is None and prim_ty_cls is None:
            raise RuntimeError(
                f"{module_name}: ``dtypes=`` requires either a "
                "``_DtypeHandle`` class (for callable-dtype dialects) "
                "OR a PrimTy-trait IR class on the module.",
            )
        ctor = handle_cls if handle_cls is not None else prim_ty_cls
        field_name = _infer_primty_field_name(ctor)
        for dt in dtypes:
            if not hasattr(module, dt):
                setattr(module, dt, ctor(**{field_name: dt}))

    # Default literal-ty hooks.
    if default_dtypes:
        for category, dt_name in default_dtypes.items():
            attr_name = f"__ffi_default_{category}_ty__"
            if not hasattr(module, attr_name):
                handle = getattr(module, dt_name, None)
                if handle is None:
                    raise RuntimeError(
                        f"{module_name}: default_dtypes references "
                        f"{dt_name!r} but the module has no such attribute",
                    )
                setattr(module, attr_name, handle)

    # A ``__ffi_make_var__`` default: if the module has a ValueTraits
    # IR class and no user-assigned hook, wire one.
    if not hasattr(module, "__ffi_make_var__"):
        val_cls = _find_class_with_trait(module, tr.ValueTraits)
        if val_cls is not None:
            name_field = "name"
            ty_field = "ty"
            trait = val_cls.__ffi_ir_traits__
            n = _strip_field_prefix(trait.name)
            if n is not None:
                name_field = n
            if trait.ty is not None:
                t = _strip_field_prefix(trait.ty)
                if t is not None:
                    ty_field = t

            # Normalization base: the canonical PrimTy class in this
            # module. When the parser passes a ``_DtypeHandle`` subclass
            # instance (from ``T.int32`` attribute lookup), we reconstruct
            # a fresh instance of the base PrimTy class so py_class's
            # exact-class structural-eq stays stable across roundtrips.
            prim_ty_cls = _find_class_with_trait(module, tr.PrimTyTraits)
            prim_ty_field = (
                _infer_primty_field_name(prim_ty_cls) if prim_ty_cls else None
            )

            def _make_var(_parser: Any, name: str, ty: Any,
                          _cls: type = val_cls,
                          _nf: str = name_field, _tf: str = ty_field,
                          _pty: Optional[type] = prim_ty_cls,
                          _pf: Optional[str] = prim_ty_field) -> Any:
                if (_pty is not None and _pf is not None
                        and isinstance(ty, _pty) and type(ty) is not _pty):
                    ty = _pty(**{_pf: getattr(ty, _pf)})
                return _cls(**{_nf: name, _tf: ty})

            module.__ffi_make_var__ = _make_var  # type: ignore[attr-defined]

    # Install the CallTraits ``__getattr__`` fallback LAST so every
    # other rule's ``hasattr(module, name)`` check sees the real module
    # state and not the fallback catch-all. ``__getattr__`` is only
    # consulted after the module dict lookup fails, so previously-set
    # attributes (``prim_func``, ``int32``, ``serial`` …) win; only
    # genuinely-unknown names like ``T.mma`` route to the fallback.
    #
    # Names matching parser-protocol hooks are excluded so cross-dialect
    # ``_lookup_hook("load")`` / ``"bind"`` / ``"if_stmt"`` / … don't
    # get intercepted by a dialect that doesn't own them and can't fall
    # through to the dialect that does.
    call_cls = ctx.get("_call_fallback_cls")
    if call_cls is not None and "__getattr__" not in module.__dict__:
        callee_field = ctx["_call_fallback_callee_field"]
        args_field = ctx["_call_fallback_args_field"]

        def _module_getattr(name: str,
                            _cls: type = call_cls,
                            _cf: str = callee_field,
                            _af: str = args_field,
                            _prefix: str = prefix) -> Any:
            if name.startswith("_") or name in _PARSER_HOOK_NAMES:
                raise AttributeError(name)
            full_name = f"{_prefix}.{name}"

            def _factory(*args: Any, **kwargs: Any) -> Any:
                if kwargs:
                    raise TypeError(
                        f"{full_name}: opaque call fallback does not accept "
                        f"keyword arguments (got {list(kwargs)})",
                    )
                return _cls(**{
                    _cf: full_name,
                    _af: [_wrap_primitive_via_module(a, module) for a in args],
                })

            _factory.__name__ = full_name
            return _factory

        module.__getattr__ = _module_getattr  # type: ignore[attr-defined]


# Parser-protocol hook names — names the :class:`IRParser` looks up via
# ``_lookup_hook`` on active dialects. Must NOT be served by the
# ``__getattr__`` catch-all (that would block cross-dialect fall-through
# to a dialect that genuinely provides the hook).
_PARSER_HOOK_NAMES: frozenset[str] = frozenset([
    "load", "bind", "buffer_store", "if_stmt", "while_stmt", "assert_stmt",
    "for_stmt", "with_stmt", "ret",
])


def _find_class_with_trait(module: "ModuleType", trait_cls: type) -> Optional[type]:
    """Return the first IR class in ``module`` whose
    ``__ffi_ir_traits__`` is an instance of ``trait_cls``."""
    for _, cls in inspect.getmembers(module):
        if not (isinstance(cls, type) and hasattr(cls, "__ffi_ir_traits__")):
            continue
        if isinstance(cls.__ffi_ir_traits__, trait_cls):
            return cls
    return None


def _infer_primty_field_name(cls: type) -> str:
    """Infer the dtype-name field on a PrimTy class (e.g. ``"dtype"``
    for mini.tir, ``"name"`` for mini.mlir's IntegerType)."""
    trait = getattr(cls, "__ffi_ir_traits__", None)
    if trait is None:
        # PrimTyTraits not applied; assume a subclass used for handles.
        for base in cls.__mro__[1:]:
            base_trait = getattr(base, "__ffi_ir_traits__", None)
            if base_trait is not None:
                trait = base_trait
                break
    if trait is None:
        return "name"
    if isinstance(trait, tr.PrimTyTraits):
        field = _strip_field_prefix(trait.dtype)
        if field is not None:
            return field
    return "name"
