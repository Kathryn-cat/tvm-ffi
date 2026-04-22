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
"""Three-tier parser dispatch for ``@py_class`` IR classes.

See ``design_docs/parser_tier_dispatch.md`` for the full design.

This module supplies the runtime machinery that lets every
``@py_class`` IR class roundtrip through :meth:`IRParser.visit_call`,
by choosing at call time between:

* **Tier 1** — the class declared a custom ``__ffi_text_parse__``
  classmethod. The dispatcher invokes it directly. This is the opt-in
  escape hatch for IR shapes the trait vocabulary can't express.
* **Tier 2** — the class declared ``__ffi_ir_traits__``. The dispatcher
  routes to the trait-specific ``parse_*`` implementation in
  :mod:`tvm_ffi.pyast_trait_parse` via :data:`_TRAIT_TO_PARSE_FN`.
* **Tier 3** — the class declared neither. The dispatcher falls back
  to :func:`_default_parse`, which is the reflection-based inverse of
  the printer's ``DefaultPrint`` — value-eager evaluation of
  ``node.args`` / ``node.kwargs`` followed by ``cls(*args, **kwargs)``.

The dispatcher is **lazy**: all three paths are re-checked on every
call. A user adding ``__ffi_text_parse__`` post-registration takes
effect immediately on the next parse — no re-registration required.
This mirrors the printer's ``IRPrintDispatch`` semantics.

IR classes stay pure data. The dispatcher closure is stored on the
language module as a parallel registry ``module.__ffi_parsers__`` —
NEVER as an attribute on the class itself. The marker
``__ffi_parse_aware__ = True`` lives on the dispatcher function, not
the class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from tvm_ffi import ir_traits as tr
from tvm_ffi.pyast_trait_parse import (
    parse_assert,
    parse_assign,
    parse_for,
    parse_func,
    parse_if,
    parse_return,
    parse_store,
    parse_while,
    parse_with,
)

if TYPE_CHECKING:
    from types import ModuleType

    from tvm_ffi import pyast

__all__ = [
    "register_parser",
    "lookup_parser",
]


# ============================================================================
# Trait → trait-parser registry
# ============================================================================


_TRAIT_TO_PARSE_FN: dict[type, Callable[..., Any]] = {
    tr.FuncTraits: parse_func,
    tr.AssignTraits: parse_assign,
    tr.StoreTraits: parse_store,
    tr.ReturnTraits: parse_return,
    tr.AssertTraits: parse_assert,
    tr.IfTraits: parse_if,
    tr.WhileTraits: parse_while,
    tr.ForTraits: parse_for,
    tr.WithTraits: parse_with,
}


# ============================================================================
# Tier-3 default parse — reflection-based inverse of DefaultPrint
# ============================================================================


def _default_parse(parser: "pyast.IRParser", node: Any, cls: type) -> Any:
    """Leaf default parse — evaluate args/kwargs and call ``cls(...)``.

    The inverse of the printer's ``DefaultPrint``: a leaf IR class
    prints as ``ClassName(field=value, ...)`` and parses back the same
    way. Valid only for classes without scoped bindings (loop vars,
    function params, region bodies) — use :class:`__ffi_ir_traits__`
    (Tier 2) or ``__ffi_text_parse__`` (Tier 1) for structured IR.

    When ``cls`` declares ``__ffi_ir_traits__`` (typically a BinOp /
    UnaryOp de-sugared call like ``T.Add(1, 2)`` emitted when the
    sugar gate refuses infix), raw Python primitives in the args /
    kwargs are wrapped via the active dialect's
    ``__ffi_default_{int,float,bool}_ty__`` hooks so the parse result
    structurally matches the pre-print form.

    Raises
    ------
    TypeError
        If ``node`` isn't a :class:`pyast.Call`. Tier 3 only handles
        the ``ClassName(...)`` call shape.
    NameError
        Propagated from :meth:`parser.eval_expr` when an arg references
        an undefined identifier. The error is rewrapped with a hint
        pointing the user at Tier-1/Tier-2 opt-ins so they can see
        which classes need them.
    """
    from tvm_ffi import pyast as _pyast  # noqa: PLC0415

    if not isinstance(node, _pyast.Call):
        raise TypeError(
            f"Tier-3 default parse for {cls.__name__!r} expects "
            f"pyast.Call, got {type(node).__name__}. If "
            f"{cls.__name__!r} has scoped bindings or region bodies, "
            "use Tier 1 (__ffi_text_parse__) or Tier 2 "
            "(__ffi_ir_traits__). Tier 3 is leaf-only.",
        )

    try:
        args = [parser.eval_expr(a) for a in node.args]
        kwargs: dict[str, Any] = {
            k: parser.eval_expr(v)
            for k, v in zip(node.kwargs_keys, node.kwargs_values)
        }
    except NameError as exc:
        raise NameError(
            f"Tier-3 default parse for {cls.__name__!r} encountered an "
            f"undefined name: {exc}. If {cls.__name__!r} has scoped "
            "bindings (loop vars, function params, etc.), declare "
            "__ffi_ir_traits__ or __ffi_text_parse__. Tier 3 only "
            "supports leaf IR.",
        ) from exc

    # Per-field normalization — only wrap primitives when the target
    # field expects an IR Object (annotation ``Any`` or an Object
    # subclass). Fields typed as plain ``int`` / ``float`` / ``str``
    # / ``bool`` take the literal value unchanged.
    field_tys = _field_type_map(cls)
    wrapped_args: list[Any] = []
    for idx, value in enumerate(args):
        field_name = _positional_field_name(cls, idx)
        wrapped_args.append(
            _normalize_field_value(parser, value, field_tys.get(field_name)),
        )
    wrapped_kwargs: dict[str, Any] = {
        k: _normalize_field_value(parser, v, field_tys.get(k))
        for k, v in kwargs.items()
    }

    return cls(*wrapped_args, **wrapped_kwargs)


def _field_type_map(cls: type) -> dict[str, Any]:
    """Return a ``{field_name: declared_type}`` map for ``cls``, walking
    the ``parent_type_info`` chain so inherited fields are included."""
    info = getattr(cls, "__tvm_ffi_type_info__", None)
    out: dict[str, Any] = {}
    while info is not None:
        for f in info.fields:
            out.setdefault(f.name, getattr(f, "ty", None))
        info = info.parent_type_info
    return out


def _positional_field_name(cls: type, idx: int) -> str | None:
    """Return the declared name of the idx-th positional field on
    ``cls``, or ``None`` if idx is out of range. Walks the
    ``parent_type_info`` chain for inherited fields in declared order."""
    info = getattr(cls, "__tvm_ffi_type_info__", None)
    names: list[str] = []
    stack: list[Any] = []
    while info is not None:
        stack.append(info)
        info = info.parent_type_info
    for inf in reversed(stack):
        names.extend(f.name for f in inf.fields)
    if idx < len(names):
        return names[idx]
    return None


_PRIMITIVE_ORIGIN_NAMES: frozenset[str] = frozenset({
    "int", "float", "bool", "str", "bytes",
})


def _is_primitive_field(field_ty: Any) -> bool:
    """Return ``True`` when the field expects a raw Python primitive
    (``int`` / ``float`` / ``bool`` / ``str``) rather than an IR object.

    FFI ``TypeSchema`` wraps the annotation; ``origin`` is the string
    name of the root type (e.g. ``"int"``). Literal ``int`` / ``float``
    / etc. annotations also flow through as actual types — handle both.
    """
    if field_ty is None:
        return False
    origin = getattr(field_ty, "origin", field_ty)
    if isinstance(origin, type):
        return origin in (int, float, bool, str, bytes)
    if isinstance(origin, str):
        return origin in _PRIMITIVE_ORIGIN_NAMES
    return False


def _normalize_field_value(
    parser: "pyast.IRParser", value: Any, field_ty: Any,
) -> Any:
    """Round-trip normalization: wrap raw primitives into Imm IR when
    the field expects an Object; normalize ``PrimTy`` subclasses to
    the base PrimTy so exact-class structural eq holds.

    Pure primitive fields (``int`` / ``float`` / ``bool`` / ``str``)
    take the literal value unchanged.
    """
    from tvm_ffi import Object as _Object  # noqa: PLC0415

    if value is None:
        return value

    if _is_primitive_field(field_ty):
        return value

    if isinstance(value, (bool, int, float)) and not isinstance(value, _Object):
        value = parser._wrap_primitive_ast(value)

    return _normalize_primty_subclass(parser, value)


def _normalize_primty_subclass(parser: "pyast.IRParser", value: Any) -> Any:
    """Return a fresh :class:`PrimTy` instance when ``value`` is an
    instance of a *subclass* of the active dialect's PrimTy class.

    The printer emits ``T.int32`` / ``T.float32`` as attribute lookups
    into the dialect module; those attributes are user-defined
    ``_DtypeHandle`` instances (PrimTy subclasses with a ``__call__``
    override). Inside IR data, though, we want the plain PrimTy so
    ``py_class`` class-exact structural equality roundtrips cleanly.
    Passes non-PrimTy values through unchanged.
    """
    if value is None:
        return value
    for dialect in parser.active_dialects():
        prim_ty_cls = getattr(dialect, "__ffi_prim_ty__", None)
        if prim_ty_cls is None:
            continue
        if isinstance(value, prim_ty_cls) and type(value) is not prim_ty_cls:
            # Extract the dtype-name field dynamically — the field
            # could be named ``dtype`` (mini.tir) or ``name`` (mini.mlir).
            # Walk ``parent_type_info`` since a :class:`_DtypeHandle`
            # subclass typically inherits its fields.
            info = getattr(prim_ty_cls, "__tvm_ffi_type_info__", None)
            while info is not None and not info.fields:
                info = info.parent_type_info
            if info is not None and info.fields:
                field_name = info.fields[0].name
                return prim_ty_cls(**{field_name: getattr(value, field_name)})
    return value


# ============================================================================
# Lazy universal dispatcher — re-checks all three tiers on every call
# ============================================================================


def _make_parser(cls: type) -> Callable[..., Any]:
    """Build a lazy universal dispatcher for ``cls``.

    The dispatcher resolves the active tier at invocation time:

    1. If ``cls.__dict__`` contains a callable ``__ffi_text_parse__``
       (inherited overrides are ignored — a subclass opts in only if
       it declares the method itself), Tier 1 wins.
    2. Else if ``cls.__ffi_ir_traits__`` is set and the trait kind has
       a dedicated ``parse_*`` entry in :data:`_TRAIT_TO_PARSE_FN`,
       Tier 2 fires.
    3. Otherwise Tier 3 (:func:`_default_parse`) handles the call.

    The returned function carries the sentinel
    ``__ffi_parse_aware__ = True`` so :meth:`IRParser.visit_call` can
    distinguish parse-aware dispatchers from plain value-eager
    callables.
    """

    def dispatcher(parser: "pyast.IRParser", node: Any) -> Any:
        # Tier 1 — custom parser on this class. Check ``__dict__`` so
        # we don't inherit a parent class's override.
        custom = cls.__dict__.get("__ffi_text_parse__")
        if custom is not None:
            fn = custom.__func__ if isinstance(custom, classmethod) else custom
            if callable(fn):
                if isinstance(custom, classmethod):
                    return fn(cls, parser, node)
                return fn(parser, node)

        # Tier 2 — trait-driven.
        trait = getattr(cls, "__ffi_ir_traits__", None)
        if trait is not None:
            parse_fn = _TRAIT_TO_PARSE_FN.get(type(trait))
            if parse_fn is not None:
                return parse_fn(parser, node, cls, trait)

        # Tier 3 — reflection default.
        return _default_parse(parser, node, cls)

    dispatcher.__ffi_parse_aware__ = True  # type: ignore[attr-defined]
    dispatcher.__ffi_dispatch_cls__ = cls  # type: ignore[attr-defined]
    dispatcher.__name__ = f"_ffi_parse_{cls.__name__}"
    dispatcher.__qualname__ = f"_ffi_parse_{cls.__qualname__}"
    dispatcher.__doc__ = (
        f"Auto-generated three-tier parser dispatcher for "
        f"{cls.__module__}.{cls.__name__}."
    )
    return dispatcher


# ============================================================================
# Module-level registry of dispatchers
# ============================================================================


_PARSERS_REGISTRY_ATTR = "__ffi_parsers__"


def register_parser(module: "ModuleType", cls: type) -> Callable[..., Any]:
    """Ensure ``module.__ffi_parsers__[cls.__name__]`` is a dispatcher
    for ``cls`` and return it.

    The dispatcher is lazy (:func:`_make_parser`) so re-decoration of
    the class at runtime — e.g., a test monkeypatching
    ``__ffi_text_parse__`` onto ``cls`` after registration — is picked
    up on the next parse without re-calling this function.

    The registry lives on the module as a plain ``dict`` keyed by class
    name. It is NOT an attribute on the class (IR classes stay pure
    data). A user-authored ``module.__ffi_parsers__`` with pre-defined
    entries is respected — existing keys are not overwritten, so a
    dialect can replace any class's parser with a bespoke callable.
    """
    registry = getattr(module, _PARSERS_REGISTRY_ATTR, None)
    if registry is None:
        registry = {}
        setattr(module, _PARSERS_REGISTRY_ATTR, registry)

    if cls.__name__ not in registry:
        registry[cls.__name__] = _make_parser(cls)
    return registry[cls.__name__]


def lookup_parser(
    module: "ModuleType", cls_name: str,
) -> Callable[..., Any] | None:
    """Return the dispatcher for ``cls_name`` registered on ``module``.

    Returns ``None`` if the module has no registry or the name isn't
    present. Intended for use by :meth:`IRParser.visit_call` when the
    resolved callee is an IR class — the parser uses the lang module's
    registry to find the matching dispatcher so the three-tier logic
    runs at parse time.
    """
    registry = getattr(module, _PARSERS_REGISTRY_ATTR, None)
    if registry is None:
        return None
    return registry.get(cls_name)
