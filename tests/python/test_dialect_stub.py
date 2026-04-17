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
"""Tests for ``tvm_ffi.stub.dialect_stub`` (the ``.pyi`` generator for
``@py_class`` dialect modules — ``design_docs/parser_auto_registration.md``
§5)."""

from __future__ import annotations

import ast

import pytest

import tvm_ffi.testing.mini.tir as mt
from tvm_ffi.stub.dialect_stub import (
    discover_finalized_modules,
    generate_dialect_stub,
    write_dialect_stub,
)


@pytest.fixture(scope="module")
def tir_stub() -> str:
    """Generated ``.pyi`` text for mini-TIR — computed once per session."""
    return generate_dialect_stub("tvm_ffi.testing.mini.tir")


class TestStubStructure:
    """The generated stub is valid Python and follows the header / section
    shape documented in :mod:`tvm_ffi.stub.dialect_stub`."""

    def test_stub_is_valid_python(self, tir_stub: str) -> None:
        """``ast.parse`` accepts the stub as syntactically valid."""
        ast.parse(tir_stub)

    def test_stub_header_identifies_source(self, tir_stub: str) -> None:
        """The header records which module produced the stub so edits
        can be traced back."""
        assert "Source module: tvm_ffi.testing.mini.tir" in tir_stub
        assert "tvm-ffi-stubgen dialects tvm_ffi.testing.mini.tir" in tir_stub

    def test_stub_imports_typing_and_object(self, tir_stub: str) -> None:
        """Every declared annotation resolves against the header
        imports — ``Any`` / ``Optional`` / ``List`` from typing,
        ``Object`` from tvm_ffi."""
        assert "from typing import " in tir_stub
        assert "from tvm_ffi import Object" in tir_stub


class TestDialectSurfaceCaptured:
    """The stub covers every class of auto-wired attribute on mini-TIR.

    Combined with :class:`TestStubStructure`, this pins down "the
    generator records what ``finalize_module`` injected" — the core
    guarantee users care about for IDE autocomplete.
    """

    def test_ir_classes_emit_class_bodies(self, tir_stub: str) -> None:
        """Each registered IR class appears as a ``class`` block in the
        stub, keeping field annotations so type-checkers / IDEs still
        see ``Add.lhs: Any`` etc."""
        for name in ("Add", "Sub", "Mul", "Var", "PrimFunc", "IfThenElse"):
            assert f"class {name}(Object):" in tir_stub, (
                f"{name} missing from generated stub"
            )
        assert "    lhs: Any" in tir_stub
        assert "    rhs: Any" in tir_stub

    def test_decorator_handler_emitted_as_function(self, tir_stub: str) -> None:
        """The auto-wired ``@T.prim_func`` handler shows up as
        ``def prim_func(...)``, reachable via ``T.prim_func``."""
        assert "def prim_func(" in tir_stub

    def test_iter_factories_emitted(self, tir_stub: str) -> None:
        """Every iter kind declared in ``iter_kinds=[...]`` is a
        module-level factory."""
        for kind in ("serial", "parallel", "unroll", "vectorized"):
            assert f"def {kind}(" in tir_stub, f"iter factory {kind!r} missing"

    def test_parser_hooks_emitted(self, tir_stub: str) -> None:
        """Parser-protocol hooks (auto-wired by the trait rules) surface
        as typed functions in the stub."""
        for hook in ("bind", "buffer_store", "load", "if_stmt", "while_stmt"):
            assert f"def {hook}(" in tir_stub, f"hook {hook!r} missing"

    def test_parse_hooks_emitted(self, tir_stub: str) -> None:
        """``__ffi_parse_hooks__`` entries (here ``func_attr``) land as
        typed callables — this is the point of stubgen for the
        :mod:`parser_frame_hooks` feature."""
        assert "def func_attr(" in tir_stub

    def test_dtype_handles_emit_as_instances(self, tir_stub: str) -> None:
        """Dtype handles are data, not callables-to-declare — each one
        is a typed module attribute."""
        for name in ("int32", "float32", "bool", "float16", "int8"):
            assert f"{name}: _DtypeHandle" in tir_stub, (
                f"dtype handle {name!r} should be an instance attr"
            )

    def test_ffi_metadata_emitted(self, tir_stub: str) -> None:
        """The per-module FFI metadata dicts the parser reads are
        declared so static tools see their types."""
        for attr in (
            "__ffi_op_classes__",
            "__ffi_parsers__",
            "__ffi_default_int_ty__",
            "__ffi_default_float_ty__",
            "__ffi_default_bool_ty__",
        ):
            assert attr in tir_stub, f"metadata attr {attr!r} missing"


class TestImportFiltering:
    """Imported names (``typing.Any``, ``tvm_ffi.Object``, internal
    framework helpers) must not leak into the stub as top-level
    declarations — only the dialect's own surface should appear."""

    def test_typing_imports_not_redeclared(self, tir_stub: str) -> None:
        """``typing.Any`` etc. are imported, never redeclared."""
        # These names only appear on the ``from typing import`` line,
        # never as standalone ``class Any:`` / ``Any: type`` entries.
        assert "class Any" not in tir_stub
        assert "class Optional" not in tir_stub

    def test_framework_helpers_not_redeclared(self, tir_stub: str) -> None:
        """``py_class``, ``finalize_module``, ``register_global_func``
        are import-only; they belong to the framework, not the dialect."""
        for name in ("py_class", "finalize_module", "register_global_func"):
            assert f"def {name}(" not in tir_stub, (
                f"{name!r} leaked into the stub"
            )


class TestWriteAndDiscover:
    """File-writing and discovery glue — short, focused tests."""

    def test_write_dialect_stub_creates_file(self, tmp_path) -> None:
        """``write_dialect_stub`` honors an explicit ``output_path``
        and returns the path it wrote."""
        target = tmp_path / "tir.pyi"
        written = write_dialect_stub("tvm_ffi.testing.mini.tir", target)
        assert written == target
        assert target.exists()
        ast.parse(target.read_text())

    def test_discover_finalized_modules_includes_mini_tir(self) -> None:
        """Once mini-TIR has been imported, discovery picks it up via
        its ``__ffi_parsers__`` / ``__ffi_op_classes__`` markers."""
        _ = mt  # force-import for the test to be independent of collection order
        found = discover_finalized_modules()
        assert "tvm_ffi.testing.mini.tir" in found
