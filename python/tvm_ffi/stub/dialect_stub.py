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
"""``.pyi`` stub generator for ``@py_class`` dialect modules.

See ``design_docs/parser_auto_registration.md`` §5. The module is
driven from :mod:`tvm_ffi.stub.cli` via the ``dialects`` subcommand:

.. code-block:: bash

    tvm-ffi-stubgen dialects tvm_ffi.testing.mini.tir
    tvm-ffi-stubgen dialects --all

For each target module the generator:

1. Imports the module so :func:`~tvm_ffi.dialect_autogen.finalize_module`
   has run and every auto-wired attribute (dtype handles, iter factories,
   decorator handlers, parser hooks, parse hooks, ``__ffi_*`` metadata
   dicts) is observable via ``module.__dict__``.
2. Walks that dict, classifies each entry, and emits a typed ``.pyi``
   declaration. Imported names (``typing.Any``, ``tvm_ffi.Object``,
   ``dataclass``, …) are filtered out so the stub only reflects the
   dialect's own surface.

The ``.pyi`` is a *supplementary* stub — type-checkers that consume
both ``.py`` and ``.pyi`` will prefer the stub, so the generator
redeclares every public name so nothing disappears. Fields on IR
classes survive this round-trip via the class body; signatures for
user-defined factories are lifted with :func:`inspect.signature` when
available.
"""

from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

__all__ = [
    "generate_dialect_stub",
    "write_dialect_stub",
    "discover_finalized_modules",
]


# ============================================================================
# Heuristics for picking dialect-owned names out of ``module.__dict__``
# ============================================================================


#: Names that almost every dialect module imports from the framework.
#: They're legitimate attributes of ``module.__dict__`` but clearly not
#: part of the dialect's own surface — the stub drops them silently.
_IMPORT_ONLY_NAMES: frozenset[str] = frozenset({
    # typing
    "Any", "Optional", "List", "Dict", "Set", "Tuple", "Callable",
    "Sequence", "Mapping", "Union", "ClassVar", "TypeVar", "Generic",
    "TYPE_CHECKING",
    # tvm_ffi re-exports
    "Object", "dataclass", "dc_field", "py_class", "field",
    "register_global_func", "finalize_module", "ffi_dtype", "tr",
    "pyast",
    # stdlib
    "annotations",
})

#: ``typing`` names we emit a single ``from typing import …`` line for
#: at the top of every generated stub. Generously inclusive — unused
#: imports in a ``.pyi`` are harmless and keeping the list fixed avoids
#: having to parse every annotation to figure out what's actually needed.
_TYPING_IMPORTS: tuple[str, ...] = (
    "Any", "Callable", "Dict", "List", "Mapping", "MutableMapping",
    "Optional", "Sequence", "Set", "Tuple", "Union",
)

#: Module prefixes that mark an attribute's owning module as "framework-level".
#: Auto-wired helpers live under ``tvm_ffi.dialect_autogen``; those are
#: part of the dialect's surface (included), in contrast with Objects
#: imported from elsewhere.
_DIALECT_OWNED_PREFIXES: tuple[str, ...] = (
    "tvm_ffi.dialect_autogen",
    "tvm_ffi.pyast",  # frame_setter / frame_merger factories
)


def _is_dialect_owned(
    name: str, value: Any, dialect_module: ModuleType,
) -> bool:
    """Return True if ``value`` should appear in the dialect's ``.pyi``."""
    if name in _IMPORT_ONLY_NAMES:
        return False
    # Private helpers are kept only when they are explicitly metadata
    # (dtype handles, ``__ffi_*`` dicts, ``__getattr__`` fallback …).
    if name.startswith("_") and not (
        name.startswith("__ffi_") or name in {"__getattr__"}
    ):
        return False

    # Modules are generally imports (``import tvm_ffi.pyast as pyast``);
    # the one exception is a module *self-alias* like ``T = _this`` /
    # ``TLang = _this`` — that's a back-compat façade the dialect owns.
    if isinstance(value, ModuleType):
        return value is dialect_module

    val_mod = getattr(value, "__module__", None)
    if val_mod is None:
        return True
    if val_mod == dialect_module.__name__:
        return True
    if val_mod.startswith(dialect_module.__name__ + "."):
        return True
    if any(val_mod.startswith(p) for p in _DIALECT_OWNED_PREFIXES):
        return True
    # Anything else (``typing``, ``builtins``, sibling packages …) is
    # an import; drop it.
    return False


# ============================================================================
# Per-entry emitters
# ============================================================================


def _is_ir_class(value: Any) -> bool:
    """True when ``value`` is a registered ``@py_class`` IR class."""
    return isinstance(value, type) and hasattr(value, "__tvm_ffi_type_info__")


def _annot_str(ty: Any) -> str:
    """Best-effort stringifier for an annotation object."""
    if ty is None:
        return "None"
    if isinstance(ty, str):
        return ty
    if hasattr(ty, "__name__"):
        return ty.__name__
    return "Any"


def _emit_ir_class(name: str, cls: type) -> str:
    """Render a ``class Name(Object): field: ty`` stub for an IR class."""
    parent = cls.__mro__[1].__name__ if len(cls.__mro__) > 1 else "object"
    ann = getattr(cls, "__annotations__", {}) or {}
    body_lines: list[str] = []
    for fname, fty in ann.items():
        body_lines.append(f"    {fname}: {_annot_str(fty)}")
    if not body_lines:
        body_lines.append("    pass")
    return f"class {name}({parent}):\n" + "\n".join(body_lines) + "\n"


def _emit_non_ir_class(name: str, cls: type) -> str:
    """Render a stub for a non-IR helper class (e.g. ``_IterHolder``).

    Keeps annotated fields (dataclass-style) when present; otherwise
    emits a single-pass body.
    """
    parent = cls.__mro__[1].__name__ if len(cls.__mro__) > 1 else "object"
    if parent in {"object", "type"}:
        header = f"class {name}:"
    else:
        header = f"class {name}({parent}):"
    ann = getattr(cls, "__annotations__", {}) or {}
    body_lines = [f"    {fname}: {_annot_str(fty)}" for fname, fty in ann.items()]
    if not body_lines:
        body_lines.append("    pass")
    return f"{header}\n" + "\n".join(body_lines) + "\n"


def _emit_function(name: str, fn: Any) -> str:
    """Render a typed ``def`` stub for a user-defined or auto-wired callable.

    When :func:`inspect.signature` can read the signature, it's echoed
    verbatim (positional args + defaults) with ``Any`` fallbacks for
    unknown annotations. Otherwise — e.g. closures that use ``*args``
    — the stub degrades to ``def name(*args: Any, **kwargs: Any) -> Any``.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return f"def {name}(*args: Any, **kwargs: Any) -> Any: ...\n"

    parts: list[str] = []
    for p in sig.parameters.values():
        piece = p.name
        if p.kind is inspect.Parameter.VAR_POSITIONAL:
            piece = "*" + piece
        elif p.kind is inspect.Parameter.VAR_KEYWORD:
            piece = "**" + piece
        if p.annotation is not inspect.Parameter.empty:
            piece += f": {_annot_str(p.annotation)}"
        else:
            piece += ": Any"
        if p.default is not inspect.Parameter.empty and p.kind not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            piece += " = ..."
        parts.append(piece)
    ret_ann = "Any"
    if sig.return_annotation is not inspect.Signature.empty:
        ret_ann = _annot_str(sig.return_annotation)
    return f"def {name}({', '.join(parts)}) -> {ret_ann}: ...\n"


def _emit_instance_attr(name: str, value: Any) -> str:
    """Render a ``name: Type`` line for module-level instances (dtype
    handles, ``__ffi_parsers__`` dict, module-self aliases, …)."""
    ty = type(value).__name__
    # Translate a few common built-in names to their typing equivalents
    # to keep the stub clean.
    if ty == "dict":
        return f"{name}: dict[Any, Any]\n"
    if ty == "list":
        return f"{name}: list[Any]\n"
    if ty == "tuple":
        return f"{name}: tuple[Any, ...]\n"
    if ty in {"int", "str", "float", "bool"}:
        return f"{name}: {ty}\n"
    if ty == "NoneType":
        return f"{name}: None\n"
    if ty == "module":
        return f"{name}: Any  # module alias\n"
    return f"{name}: {ty}\n"


# ============================================================================
# Top-level driver
# ============================================================================


_STUB_HEADER = '''\
# Auto-generated by ``tvm-ffi-stubgen dialects``. DO NOT EDIT.
#
# Source module: {module_name}
# Regenerate with:
#     tvm-ffi-stubgen dialects {module_name}
#
# This stub declares every dialect attribute that
# :func:`tvm_ffi.dialect_autogen.finalize_module` injects (dtype
# handles, factory / decorator handlers, parser hooks, ``__ffi_*``
# metadata, parse hooks). IR-class bodies mirror the ``.py`` source
# for ``@py_class`` classes so IDEs and type-checkers keep seeing
# their fields.

from __future__ import annotations

from types import ModuleType
from typing import {typing_imports}

from tvm_ffi import Object
from tvm_ffi.pyast import Frame, FuncFrame, IRParser
'''


def _is_plain_function(value: Any) -> bool:
    """True for bona-fide functions (``def`` / ``lambda`` / closure /
    staticmethod-unwrapped). Class instances that happen to implement
    ``__call__`` don't qualify — those are data, not functions, and
    should be emitted as typed instance attributes."""
    return (
        inspect.isfunction(value)
        or inspect.isbuiltin(value)
        or inspect.ismethod(value)
    )


def _classify_and_sort(
    module: ModuleType,
) -> dict[str, list[tuple[str, Any]]]:
    """Group module-dict entries by emission category.

    Categories (emitted in this order):

    * ``ir_classes`` — ``@py_class`` IR types owned by this module.
    * ``other_classes`` — helper classes like ``_IterHolder`` /
      ``_DtypeHandle`` / ``_BlockMarker``.
    * ``functions`` — real Python functions, user factories + auto-
      wired closures. Callable instances (``_DtypeHandle`` instances
      with ``__call__``) go into ``instances`` instead.
    * ``instances`` — non-function module-level data (dtype handle
      instances, ``__ffi_*`` dicts, module self-aliases, …).
    """
    out: dict[str, list[tuple[str, Any]]] = {
        "ir_classes": [], "other_classes": [], "functions": [], "instances": [],
    }

    for name in sorted(module.__dict__):
        value = module.__dict__[name]
        if not _is_dialect_owned(name, value, module):
            continue
        if isinstance(value, type):
            (out["ir_classes"] if _is_ir_class(value) else out["other_classes"]).append(
                (name, value),
            )
        elif _is_plain_function(value):
            out["functions"].append((name, value))
        else:
            out["instances"].append((name, value))
    return out


def generate_dialect_stub(module: ModuleType | str) -> str:
    """Return the ``.pyi`` text for ``module``.

    Accepts a Python module object or a dotted module name. The module
    is imported first so ``finalize_module`` has already populated its
    auto-wired attributes.
    """
    if isinstance(module, str):
        module = importlib.import_module(module)

    header = _STUB_HEADER.format(
        module_name=module.__name__,
        typing_imports=", ".join(_TYPING_IMPORTS),
    )
    parts: list[str] = [header]
    groups = _classify_and_sort(module)

    if groups["ir_classes"]:
        parts.append("\n\n# ===== IR classes =====\n\n")
        parts.append("\n\n".join(_emit_ir_class(n, c) for n, c in groups["ir_classes"]))

    if groups["other_classes"]:
        parts.append("\n\n\n# ===== Helper classes =====\n\n")
        parts.append(
            "\n\n".join(_emit_non_ir_class(n, c) for n, c in groups["other_classes"]),
        )

    if groups["functions"]:
        parts.append("\n\n\n# ===== Factories / hooks =====\n\n")
        parts.append("".join(_emit_function(n, f) for n, f in groups["functions"]))

    if groups["instances"]:
        parts.append("\n# ===== Dtype handles & module metadata =====\n\n")
        parts.append("".join(_emit_instance_attr(n, v) for n, v in groups["instances"]))

    text = "".join(parts)
    if not text.endswith("\n"):
        text += "\n"
    return text


def write_dialect_stub(
    module: ModuleType | str, output_path: Path | None = None,
) -> Path:
    """Generate and write the ``.pyi`` for ``module``.

    Defaults to writing next to the module's ``.py`` source. Returns
    the path that was written.
    """
    if isinstance(module, str):
        module = importlib.import_module(module)

    if output_path is None:
        source_file = getattr(module, "__file__", None)
        if source_file is None:
            raise RuntimeError(
                f"Cannot determine output path for {module.__name__!r}: "
                "module has no ``__file__``",
            )
        output_path = Path(source_file).with_suffix(".pyi")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(generate_dialect_stub(module))
    return output_path


def discover_finalized_modules() -> list[str]:
    """Return the dotted names of every already-imported module that
    was finalized by :func:`~tvm_ffi.dialect_autogen.finalize_module`.

    Detection is duck-typed: a module is "finalized" iff it owns a
    ``__ffi_parsers__`` or ``__ffi_op_classes__`` attribute. Nothing
    new is imported — callers should preload any candidate modules
    they care about (e.g. via ``--imports``).
    """
    out: list[str] = []
    for name, module in sys.modules.items():
        if not isinstance(module, ModuleType):
            continue
        if hasattr(module, "__ffi_parsers__") or hasattr(module, "__ffi_op_classes__"):
            out.append(name)
    return sorted(out)


def generate_for_modules(
    module_names: Iterable[str], *, output_root: Path | None = None,
) -> list[Path]:
    """Write ``.pyi`` files for each module name in ``module_names``.

    When ``output_root`` is given, each stub goes to
    ``output_root/<dotted/path>.pyi``; otherwise the stub lands next to
    the module's ``.py`` source. Returns the list of written paths.
    """
    written: list[Path] = []
    for module_name in module_names:
        module = importlib.import_module(module_name)
        target: Path | None = None
        if output_root is not None:
            target = output_root / (module_name.replace(".", "/") + ".pyi")
        written.append(write_dialect_stub(module, target))
    return written
