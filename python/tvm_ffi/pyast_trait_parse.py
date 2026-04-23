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
"""Trait-driven parse functions: PyAST nodes -> IR objects."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tvm_ffi import ir_traits as tr
from tvm_ffi import pyast
from tvm_ffi.core import _lookup_type_attr

if TYPE_CHECKING:
    from tvm_ffi.pyast import IRParser


_SENTINEL = object()


# ============================================================================
# Internal helpers
# ============================================================================


_REF_PREFIXES: tuple[tuple[str, str], ...] = (
    ("$global:", "global"),
    ("$method:", "method"),
    ("$field:", "field"),
)


def _classify_ref(ref: str) -> tuple[str, str]:
    """Split a trait ref into ``(kind, payload)``."""
    for prefix, kind in _REF_PREFIXES:
        if ref.startswith(prefix):
            return (kind, ref[len(prefix) :])
    return ("literal", ref)


def _resolve_field_ref(ref: str, *, trait_field: str) -> str:
    """Return the IR-field name a ``$field:`` ref writes into."""
    kind, payload = _classify_ref(ref)
    if kind == "field":
        return payload
    raise TypeError(
        f"{trait_field} holds a {kind} ref ({ref!r}); trait-driven "
        f"parsing cannot support $method:/$global: refs. Override "
        f"``__ffi_text_parse__`` on the IR class to define its parser "
        f"behavior manually.",
    )


def _trait_of(ir_class: type) -> tr.IRTraits | None:
    """Look up ``__ffi_ir_traits__`` for an IR *class*."""
    info = getattr(ir_class, "__tvm_ffi_type_info__", None)
    if info is None:
        return None
    return _lookup_type_attr(info.type_index, "__ffi_ir_traits__")


def _resolve_trait(
    ir_class: type,
    trait: tr.IRTraits | None,
    expected_cls: type,
    fn_name: str,
) -> Any:
    """Resolve + type-check the trait for a ``parse_X`` entry point."""
    if trait is None:
        trait = _trait_of(ir_class)
    if not isinstance(trait, expected_cls):
        raise TypeError(
            f"{ir_class.__name__} has no {expected_cls.__name__}; cannot use {fn_name}",
        )
    return trait


def _bind_var(
    parser: IRParser,
    name: str,
    *,
    annotation: Any = None,
    ty: Any = _SENTINEL,
) -> Any:
    """Define a fresh Var in the current scope."""
    if annotation is not None:
        resolved_ty = parser.eval_expr(annotation)
        if callable(resolved_ty):
            resolved_ty = resolved_ty()
    elif ty is not _SENTINEL:
        resolved_ty = ty
    else:
        resolved_ty = None
    var = parser.make_var(name, resolved_ty)
    parser.define(name, var)
    return var


def _define_region_vars(
    parser: IRParser,
    args: list[pyast.Assign],
    region: tr.RegionTraits,
    fields: dict[str, Any],
) -> None:
    """Bind a region's ``def_values`` from a list of PyAST ``Assign`` args."""
    if region.def_values is None:
        return
    params: list[Any] = []
    for i, arg in enumerate(args):
        if not isinstance(arg, pyast.Assign):
            raise TypeError(
                f"Region def-var arg {i}: expected pyast.Assign, got {type(arg).__name__}.",
            )
        if not isinstance(arg.lhs, pyast.Id):
            raise TypeError(
                f"Region def-var arg {i}: expected lhs of type pyast.Id, "
                f"got {type(arg.lhs).__name__}.",
            )
        if arg.annotation is None:
            raise TypeError(
                f"Region def-var arg {i} ({arg.lhs.name!r}): annotation "
                f"is required so __ffi_make_var__ can be auto-discovered "
                f"from its root identifier (e.g. ``T`` in ``T.int32``).",
            )
        params.append(
            parse_value_def(
                parser,
                arg.lhs.name,
                arg.annotation,
                make_var=None,
                default_ty=None,
            ),
        )
    fields[_resolve_field_ref(region.def_values, trait_field="RegionTraits.def_values")] = params


def _maybe_wrap(value: Any, lit_wrap: Any) -> Any:
    """Wrap a non-:class:`Object` value into an IR literal via ``lit_wrap``."""
    if lit_wrap is None or value is None:
        return value
    from tvm_ffi import Object  # noqa: PLC0415

    if isinstance(value, Object):
        return value
    return lit_wrap(value)


def _resolve_lang_module_from_annotation(parser: IRParser, annotation: Any) -> Any:
    """Resolve the language module owning an annotation's root identifier."""
    node = annotation
    while isinstance(node, pyast.Attr):
        node = node.obj
    if not isinstance(node, pyast.Id):
        raise TypeError(
            f"Cannot resolve language module from annotation: expected "
            f"the root of the .obj chain to be pyast.Id, got "
            f"{type(node).__name__}.",
        )
    return parser.eval_expr(node)


def parse_value_def(
    parser: IRParser,
    name: str,
    annotation: Any,
    make_var: Any,
    *,
    default_ty: Any = None,
) -> Any:
    """Parse a value def-site (name + optional type annotation) into an IR Var."""
    if annotation is not None:
        ty = parser.eval_expr(annotation)
    else:
        ty = default_ty

    if make_var is None and annotation is not None:
        root_id = _annotation_root_id(annotation)
        if isinstance(root_id, pyast.Id):
            lang_module = parser._lang_modules.get(root_id.name)
            if lang_module is not None:
                make_var = getattr(lang_module, "__ffi_make_var__", None)

    if make_var is None:
        make_var = parser._lookup_hook("__ffi_make_var__")

    if make_var is None:
        raise TypeError(
            f"parse_value_def({name=}): no ``__ffi_make_var__`` hook "
            f"resolvable — neither the annotation's root dialect nor "
            f"any registered language module exposes one. Either pass "
            f"``make_var=`` explicitly or register an ``__ffi_make_var__`` "
            f"attribute on the relevant language module.",
        )

    var = make_var(parser, name, ty)
    parser.define(name, var)
    return var


def _annotation_root_id(annotation: Any) -> Any:
    """Return the root :class:`pyast.Id` of an annotation's attribute chain."""
    node = annotation
    while True:
        if isinstance(node, pyast.Attr):
            node = node.obj
            continue
        if isinstance(node, pyast.Call):
            node = node.callee
            continue
        break
    return node


#: Trait attributes on :class:`FuncTraits` that may carry ``$field:`` refs
#: whose values are populated by parse hooks (frame-backed fields). See
#: ``design_docs/parser_frame_hooks.md`` §4.4.
_FUNC_FRAME_BACKED_TRAIT_ATTRS: tuple[str, ...] = ("attrs",)


def parse_func(
    parser: IRParser,
    node: pyast.Function,
    ir_class: type,
    trait: tr.FuncTraits | None = None,
) -> Any:
    """Parse a :class:`pyast.Function` into an IR object via ``FuncTraits``.

    Handles prologue / epilogue calls registered via
    ``__ffi_parse_hooks__`` by reading back any ``$field:`` refs on
    frame-backed trait attributes from the enclosing :class:`FuncFrame`
    after ``visit_body`` has finished. Non-``$field:`` refs (``$method:``
    / ``$global:``) are skipped — those compute their value at print
    time and have no parse-side frame analogue.
    """
    trait = _resolve_trait(ir_class, trait, tr.FuncTraits, "parse_func")
    region = trait.region
    fields: dict[str, Any] = {
        _resolve_field_ref(trait.symbol, trait_field="FuncTraits.symbol"): node.name.name,
    }

    # Discover frame-backed fields: trait attributes (``attrs`` today,
    # extended as more such fields appear) whose ref points at a
    # ``$field:NAME`` on the IR class. Those NAMEs get populated from
    # the enclosing :class:`FuncFrame` after the body parse completes.
    frame_backed: dict[str, str] = {}
    for trait_attr in _FUNC_FRAME_BACKED_TRAIT_ATTRS:
        ref = getattr(trait, trait_attr, None)
        field_name = _field_from_ref(ref)
        if field_name is not None:
            frame_backed[field_name] = field_name

    with parser.scoped_frame():
        _define_region_vars(parser, node.args, region, fields)

        body_ast = list(node.body)
        if region.ret is not None and body_ast and isinstance(body_ast[-1], pyast.Return):
            ret_node = body_ast.pop()
            ret_val = (
                parser.eval_expr(ret_node.value) if ret_node.value is not None else None
            )
            fields[_resolve_field_ref(region.ret, trait_field="RegionTraits.ret")] = ret_val
        fields[_resolve_field_ref(region.body, trait_field="RegionTraits.body")] = (
            parser.visit_body(body_ast)
        )

        if frame_backed:
            frame = parser.find_frame(
                pyast.FuncFrame, origin="parse_func frame-backed fields",
            )
            for ir_field, frame_attr in frame_backed.items():
                val = frame.__dict__.get(frame_attr)
                if val is not None:
                    fields[ir_field] = val

    return ir_class(**fields)


def _field_from_ref(ref: Any) -> str | None:
    """Extract the field name from a ``$field:NAME`` ref, or :data:`None`
    for ``None`` / ``$method:`` / ``$global:`` / literal refs."""
    if isinstance(ref, str) and ref.startswith("$field:"):
        return ref[len("$field:"):]
    return None


def parse_assign(
    parser: IRParser,
    node: pyast.Assign,
    ir_class: type,
    trait: tr.AssignTraits | None = None,
) -> Any:
    """Parse a ``pyast.Assign`` into a trait-driven ``AssignTraits`` IR."""
    trait = _resolve_trait(ir_class, trait, tr.AssignTraits, "parse_assign")

    if trait.def_values is None:
        raise NotImplementedError(
            f"parse_assign: {ir_class.__name__} has AssignTraits.def_values=None "
            f"(PrintAssign Mode 1, expression-statement mode). Not supported; "
            f"that mode emits pyast.ExprStmt, not pyast.Assign.",
        )

    if node.aug_op != pyast.OperationKind.Undefined:
        raise NotImplementedError(
            f"parse_assign: aug-assignment (aug_op={node.aug_op}) is not yet "
            f"supported. The caller must pre-desugar ``b <op>= rhs`` to "
            f"``b = b <op> rhs`` before invoking parse_assign.",
        )

    if not isinstance(node.lhs, pyast.Id):
        raise NotImplementedError(
            f"parse_assign: subscript / non-Id LHS ({type(node.lhs).__name__}) "
            f"is PrintAssign Sub-case 2c (non-Var def_values). Not supported "
            f"by parse_assign; use parse_store for StoreTraits IR classes.",
        )
    name = node.lhs.name

    if node.annotation is None:
        raise NotImplementedError(
            f"parse_assign: ``{name} = ...`` without a type annotation requires "
            f"rhs-driven type inference (e.g. infer ``{name}``'s type from "
            f"the rhs expression's dtype). Not yet supported. Add an explicit "
            f"annotation: ``{name}: T.<dtype> = ...``",
        )
    var = parse_value_def(parser, name, node.annotation, make_var=None)

    rhs = _wrap_if_literal(parser, node.rhs) if node.rhs is not None else None

    if rhs is None:
        raise NotImplementedError(
            f"parse_assign: bare declaration ``{name}: ...`` (no rhs) is "
            f"PrintAssign Sub-case 2a (RHS null → expr-stmt of LHS). "
            f"Not supported.",
        )

    fields: dict[str, Any] = {
        _resolve_field_ref(trait.rhs, trait_field="AssignTraits.rhs"): rhs,
        _resolve_field_ref(trait.def_values, trait_field="AssignTraits.def_values"): var,
    }
    return ir_class(**fields)


#: Dtype-name substrings that mark a dtype as floating-point. Matches
#: both the TIR convention (``"float16"`` / ``"float32"`` / ``"float64"``)
#: and the MLIR convention (``"f16"`` / ``"f32"`` / ``"f64"``). Extend
#: here if a future dialect introduces a new float-ish prefix.
_FLOAT_DTYPE_PREFIXES: tuple[str, ...] = ("float", "f")


def _dtype_is_float(dtype_str: str) -> bool:
    """``True`` when ``dtype_str`` names a floating-point scalar."""
    if dtype_str.startswith("float"):
        return True
    # Short MLIR-style names (``f16`` / ``f32`` / ``f64``).
    return dtype_str in {"f16", "f32", "f64"}


def _discover_literal_classes(module: "Any") -> dict[str, type]:
    """Scan ``module`` for ``@py_class`` IR classes with
    :class:`LiteralTraits`, returning a ``{format: cls}`` map.

    The dialect-autogen refactor uses this so the synthesized
    ``_DtypeHandle`` can reach the right ``IntImm`` / ``FloatImm`` class
    without the user passing one explicitly. Walks ``module.__dict__``
    because that dict is what :func:`finalize_module` already operates
    on.
    """
    import inspect  # noqa: PLC0415

    out: dict[str, type] = {}
    for _, cls in inspect.getmembers(module):
        if not (isinstance(cls, type) and hasattr(cls, "__ffi_ir_traits__")):
            continue
        trait = cls.__ffi_ir_traits__
        if isinstance(trait, tr.LiteralTraits):
            out.setdefault(trait.format, cls)
    return out


def make_imm_for_dtype(module: "Any", dtype_str: str, value: Any) -> Any:
    """Build an ``Imm`` IR whose dtype matches ``dtype_str``.

    Lifted out of :func:`parse_literal` so the synthesized
    ``_DtypeHandle`` (see :func:`tvm_ffi.dialect_autogen._make_dtype_handle_class`)
    can share a single implementation. Discovers the dialect's
    ``IntImm`` / ``FloatImm`` classes by scanning ``module`` for
    :class:`LiteralTraits` with ``format="int"`` / ``"float"``; bool
    literals are routed through the ``int`` class (every dialect in
    this tree represents ``bool`` as an ``IntImm`` with ``dtype=bool``).

    Raises :class:`NotImplementedError` when the needed literal class
    isn't registered on ``module``.
    """
    from tvm_ffi import dtype as ffi_dtype  # noqa: PLC0415

    classes = _discover_literal_classes(module)
    if _dtype_is_float(dtype_str):
        float_cls = classes.get("float")
        if float_cls is None:
            raise NotImplementedError(
                f"make_imm_for_dtype: {module.__name__} has no "
                'LiteralTraits class with format="float"; cannot build '
                f"a float Imm for dtype {dtype_str!r}.",
            )
        return float_cls(value=float(value), dtype=ffi_dtype(dtype_str))

    int_cls = classes.get("int")
    if int_cls is None:
        raise NotImplementedError(
            f"make_imm_for_dtype: {module.__name__} has no "
            'LiteralTraits class with format="int"; cannot build an '
            f"int Imm for dtype {dtype_str!r}.",
        )
    return int_cls(value=int(value), dtype=ffi_dtype(dtype_str))


def parse_literal(
    parser: IRParser,
    node: pyast.Literal,
    trait: tr.LiteralTraits | None = None,  # noqa: ARG001 (unused; kept for parse_* family signature shape)
) -> Any:
    """Parse a :class:`pyast.Literal` into the dialect's default-dtype Imm IR."""
    value = node.value

    # bool is a subclass of int in Python — check it FIRST so ``True``
    # / ``False`` don't silently dispatch to the int-default handle.
    if isinstance(value, bool):
        hook_name = "__ffi_default_bool_ty__"
    elif isinstance(value, int):
        hook_name = "__ffi_default_int_ty__"
    elif isinstance(value, float):
        hook_name = "__ffi_default_float_ty__"
    else:
        raise NotImplementedError(
            f"parse_literal: no default IR mapping for Python type "
            f"{type(value).__name__} (value={value!r}).",
        )

    handle = parser._lookup_hook(hook_name)
    if handle is None:
        raise NotImplementedError(
            f"parse_literal: no registered language module exposes "
            f"``{hook_name}``. Register the default type-handle for "
            f"{type(value).__name__} literals on your lang module "
            f"(e.g. ``TLang.{hook_name} = TLang.int32``).",
        )
    return handle(value)


def _wrap_if_literal(parser: IRParser, node: pyast.Expr) -> Any:
    """Evaluate a :class:`pyast.Expr` at a consumer site and wrap bare
    primitives as Imm IR.
    """
    value = parser.eval_expr(node)
    if value is None:
        return None
    from tvm_ffi import Object  # noqa: PLC0415

    if isinstance(value, Object):
        return value
    return parse_literal(parser, pyast.Literal(value=value))


def parse_unaryop(
    parser: IRParser,
    node: pyast.Operation,
    ir_class: type,
    trait: tr.UnaryOpTraits | None = None,
) -> Any:
    """Parse a :class:`pyast.Operation` (1 operand) into a ``UnaryOpTraits`` IR."""
    if len(node.operands) != 1:
        raise ValueError(
            f"parse_unaryop: expected 1 operand, got {len(node.operands)}",
        )
    trait = _resolve_trait(ir_class, trait, tr.UnaryOpTraits, "parse_unaryop")
    operand = _wrap_if_literal(parser, node.operands[0])
    fields: dict[str, Any] = {
        _resolve_field_ref(trait.operand, trait_field="UnaryOpTraits.operand"): operand,
    }
    return ir_class(**fields)


def parse_binop(
    parser: IRParser,
    node: pyast.Operation,
    ir_class: type,
    trait: tr.BinOpTraits | None = None,
) -> Any:
    """Parse a :class:`pyast.Operation` into a ``BinOpTraits`` IR."""
    operands = node.operands
    if len(operands) < 2:
        raise ValueError(
            f"parse_binop: expected >=2 operands, got {len(operands)}",
        )
    trait = _resolve_trait(ir_class, trait, tr.BinOpTraits, "parse_binop")
    lhs_field = _resolve_field_ref(trait.lhs, trait_field="BinOpTraits.lhs")
    rhs_field = _resolve_field_ref(trait.rhs, trait_field="BinOpTraits.rhs")

    # Left-fold: ``[a, b, c, ...]`` → ``ir_class(ir_class(a, b), c)`` …
    result = _wrap_if_literal(parser, operands[0])
    for raw in operands[1:]:
        rhs = _wrap_if_literal(parser, raw)
        result = ir_class(**{lhs_field: result, rhs_field: rhs})
    return result


def parse_return(
    parser: IRParser,
    node: pyast.Return,
    ir_class: type,
    trait: tr.ReturnTraits | None = None,
) -> Any:
    """Parse a :class:`pyast.Return` into an IR object via ``ReturnTraits``."""
    trait = _resolve_trait(ir_class, trait, tr.ReturnTraits, "parse_return")
    value = _wrap_if_literal(parser, node.value) if node.value is not None else None
    return ir_class(
        **{_resolve_field_ref(trait.value, trait_field="ReturnTraits.value"): value},
    )


def parse_if(
    parser: IRParser,
    node: pyast.If,
    ir_class: type,
    trait: tr.IfTraits | None = None,
) -> Any:
    """Parse an ``if / else`` into an IR object via ``IfTraits``."""
    trait = _resolve_trait(ir_class, trait, tr.IfTraits, "parse_if")
    fields: dict[str, Any] = {
        _resolve_field_ref(trait.cond, trait_field="IfTraits.cond"): _wrap_if_literal(
            parser, node.cond,
        ),
    }

    with parser.scoped_frame():
        fields[
            _resolve_field_ref(
                trait.then_region.body,
                trait_field="IfTraits.then_region.body",
            )
        ] = parser.visit_body(node.then_branch)

    if trait.else_region is not None and node.else_branch:
        with parser.scoped_frame():
            fields[
                _resolve_field_ref(
                    trait.else_region.body,
                    trait_field="IfTraits.else_region.body",
                )
            ] = parser.visit_body(node.else_branch)

    return ir_class(**fields)


def parse_for(
    parser: IRParser,
    node: pyast.For,
    ir_class: type,
    trait: tr.ForTraits | None = None,
    loop_var_ty: Any = None,
) -> Any:
    """Parse a ``for`` loop into an IR object via ``ForTraits``."""
    trait = _resolve_trait(ir_class, trait, tr.ForTraits, "parse_for")

    if not isinstance(node.lhs, pyast.Id):
        raise NotImplementedError(
            f"Only Id loop targets supported, got {type(node.lhs).__name__}",
        )

    iter_info = parser.eval_expr(node.rhs)
    fields: dict[str, Any] = {}

    with parser.scoped_frame():
        loop_var = _bind_var(parser, node.lhs.name, ty=loop_var_ty)
        if trait.region.def_values is not None:
            fields[
                _resolve_field_ref(
                    trait.region.def_values,
                    trait_field="ForTraits.region.def_values",
                )
            ] = loop_var
        fields[
            _resolve_field_ref(
                trait.region.body,
                trait_field="ForTraits.region.body",
            )
        ] = parser.visit_body(node.body)

    def _ix(key: str) -> Any:
        if isinstance(iter_info, dict):
            return iter_info.get(key)
        return getattr(iter_info, key, None)

    def _wrap_iter_field(key: str) -> Any:
        raw = _ix(key)
        if raw is None:
            return None
        return _wrap_if_literal(parser, pyast.Literal(value=raw))

    if trait.start is not None:
        fields[_resolve_field_ref(trait.start, trait_field="ForTraits.start")] = (
            _wrap_iter_field("start")
        )
    if trait.end is not None:
        fields[_resolve_field_ref(trait.end, trait_field="ForTraits.end")] = (
            _wrap_iter_field("end")
        )
    if trait.step is not None:
        fields[_resolve_field_ref(trait.step, trait_field="ForTraits.step")] = (
            _wrap_iter_field("step")
        )

    return ir_class(**fields)


def parse_with(
    parser: IRParser,
    node: pyast.With,
    ir_class: type,
    trait: tr.WithTraits | None = None,
    as_var_ty: Any = None,
) -> Any:
    """Parse a ``with`` stmt into an IR object via ``WithTraits``."""
    trait = _resolve_trait(ir_class, trait, tr.WithTraits, "parse_with")

    parser.eval_expr(node.rhs)

    fields: dict[str, Any] = {}
    with parser.scoped_frame():
        if trait.region.def_values is not None and node.lhs is not None:
            if not isinstance(node.lhs, pyast.Id):
                raise NotImplementedError(
                    f"Only Id as-var supported, got {type(node.lhs).__name__}",
                )
            as_var = _bind_var(parser, node.lhs.name, ty=as_var_ty)
            fields[
                _resolve_field_ref(
                    trait.region.def_values,
                    trait_field="WithTraits.region.def_values",
                )
            ] = as_var
        fields[
            _resolve_field_ref(
                trait.region.body,
                trait_field="WithTraits.region.body",
            )
        ] = parser.visit_body(node.body)

    return ir_class(**fields)


def parse_while(
    parser: IRParser,
    node: pyast.While,
    ir_class: type,
    trait: tr.WhileTraits | None = None,
) -> Any:
    """Parse a ``while`` loop into an IR object via ``WhileTraits``."""
    trait = _resolve_trait(ir_class, trait, tr.WhileTraits, "parse_while")
    fields: dict[str, Any] = {
        _resolve_field_ref(trait.cond, trait_field="WhileTraits.cond"): _wrap_if_literal(
            parser, node.cond,
        ),
    }
    with parser.scoped_frame():
        fields[
            _resolve_field_ref(
                trait.region.body,
                trait_field="WhileTraits.region.body",
            )
        ] = parser.visit_body(node.body)
    return ir_class(**fields)


def parse_store(
    parser: IRParser,
    node: pyast.Assign,
    ir_class: type,
    trait: tr.StoreTraits | None = None,
    lit_wrap: Any = None,
    precomputed_rhs: Any = _SENTINEL,
) -> Any:
    """Parse a subscript-target assignment ``target[indices] = value``."""
    if not isinstance(node.lhs, pyast.Index):
        raise TypeError(
            f"parse_store requires Index lhs, got {type(node.lhs).__name__}",
        )
    trait = _resolve_trait(ir_class, trait, tr.StoreTraits, "parse_store")
    target = parser.eval_expr(node.lhs.obj)
    if lit_wrap is not None:
        indices = [_maybe_wrap(parser.eval_expr(i), lit_wrap) for i in node.lhs.idx]
    else:
        indices = [_wrap_if_literal(parser, i) for i in node.lhs.idx]

    if precomputed_rhs is not _SENTINEL:
        if lit_wrap is not None:
            value = _maybe_wrap(precomputed_rhs, lit_wrap)
        elif precomputed_rhs is None:
            value = None
        else:
            value = _wrap_if_literal(parser, pyast.Literal(value=precomputed_rhs))
    elif node.rhs is None:
        value = None
    elif lit_wrap is not None:
        value = _maybe_wrap(parser.eval_expr(node.rhs), lit_wrap)
    else:
        value = _wrap_if_literal(parser, node.rhs)

    fields: dict[str, Any] = {
        _resolve_field_ref(trait.target, trait_field="StoreTraits.target"): target,
        _resolve_field_ref(trait.value, trait_field="StoreTraits.value"): value,
    }
    if trait.indices is not None:
        fields[_resolve_field_ref(trait.indices, trait_field="StoreTraits.indices")] = indices
    return ir_class(**fields)


def parse_assert(
    parser: IRParser,
    node: pyast.Assert,
    ir_class: type,
    trait: tr.AssertTraits | None = None,
) -> Any:
    """Parse a :class:`pyast.Assert` into an IR object via ``AssertTraits``."""
    trait = _resolve_trait(ir_class, trait, tr.AssertTraits, "parse_assert")
    fields: dict[str, Any] = {
        _resolve_field_ref(trait.cond, trait_field="AssertTraits.cond"): _wrap_if_literal(
            parser, node.cond,
        ),
    }
    if trait.message is not None:
        msg = parser.eval_expr(node.msg) if node.msg is not None else None
        fields[_resolve_field_ref(trait.message, trait_field="AssertTraits.message")] = msg
    return ir_class(**fields)


# Dispatch through three tiers (Tier 1 ``__ffi_text_parse__`` →
# Tier 2 trait → Tier 3 reflection default) now lives in
# :mod:`tvm_ffi.parse_dispatch` — see ``design_docs/parser_tier_dispatch.md``.
