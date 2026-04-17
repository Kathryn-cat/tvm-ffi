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

Split into six per-dialect modules per ``design_docs/parser_auto_registration.md``
§7.7 ("one module = one dialect"):

* :mod:`.arith`    — scalar types, constants, int/float binops
* :mod:`.memref`   — ref types + load/store
* :mod:`.vector`   — vector type + fall-through-dispatched ``+``
* :mod:`.scf`      — structured control flow (``for``, ``if``)
* :mod:`.func`     — function op, return, call
* :mod:`.builtin`  — module op

Shared building blocks live in :mod:`._common` (SSA :class:`Value`,
:class:`BindOp`, :func:`_make_value`, type introspection).

User code imports this package as ``mini.mlir`` and reaches any
symbol via:

.. code-block:: python

    from tvm_ffi.testing.mini import mlir as mm
    mm.arith.i32          # dialect attribute
    mm.IntegerType        # re-exported class
    mm.LANG_MODULES       # parser-ready dialect registry
"""

# ruff: noqa: A003, D102, N802, UP006, UP045, F401, F403

from __future__ import annotations

from typing import Any

# Re-export every per-dialect module as a subpackage attribute so
# callers can write ``mm.arith``, ``mm.memref``, …
from . import arith, builtin, func, memref, scf, vector

# Common/shared symbols.
from ._common import (
    BindOp,
    Value,
    _is_float_type,
    _is_int_type,
    _is_vector_type,
    _make_value,
    _operand_type,
)

# Dialect-specific class re-exports — tests use them as ``mm.ClassName``.
from .arith import (
    AddFOp,
    AddIOp,
    CmpIOp,
    ConstantOp,
    FloatType,
    IntegerType,
    MulFOp,
    MulIOp,
    SubFOp,
    SubIOp,
)
from .builtin import ModuleOp
from .func import CallOp, FuncOp, ReturnOp
from .memref import LoadOp, MemRefType, StoreOp
from .scf import ScfForOp, ScfIfOp
from .vector import VectorAddOp, VectorType


# ============================================================================
# Back-compat ``*Lang`` aliases — each dialect IS its module.
# ============================================================================


ArithLang = arith
MemRefLang = memref
VectorLang = vector
ScfLang = scf
FuncLang = func
BuiltinLang = builtin


# ============================================================================
# ``T`` type namespace — printer-hardcoded prefix forwarder.
#
# Known limitation per design doc §7.1: the printer emits ``T.<name>``
# for every TyTraits regardless of owning dialect, so we need a single
# ``T`` module that forwards scalar lookups to arith, shape-types to
# memref/vector, and ``T.Buffer``/``T.Tensor`` factories to their
# respective dialects.
# ============================================================================


class _TNamespace:
    """Unified ``T.`` type namespace — forwards to the owning dialect."""

    # Scalar types — re-exported from arith.
    i1 = arith.i1
    i8 = arith.i8
    i16 = arith.i16
    i32 = arith.i32
    i64 = arith.i64
    index = arith.index
    f16 = arith.f16
    f32 = arith.f32
    f64 = arith.f64

    # Parameterized type factories.
    Buffer = staticmethod(memref.memref)
    Tensor = staticmethod(vector.vector)


T = _TNamespace()


# ============================================================================
# Shared assign-router and load hook
#
# Since mini.mlir spans multiple dialects, the parser needs a single
# ``__ffi_assign__`` router that handles both store-style (``A[i] = v``)
# and bind-style (``c = op(...)``) assignments. Also a shared ``load``
# hook for subscript reads.
# ============================================================================


def _assign_impl(parser: Any, node: Any) -> Any:
    """Router for :class:`pyast.Assign` — store-style vs bind-style."""
    from tvm_ffi import pyast as _pyast  # noqa: PLC0415
    from tvm_ffi.pyast_trait_parse import parse_assign, parse_store  # noqa: PLC0415

    if isinstance(node.lhs, _pyast.Index):
        return parse_store(parser, node, StoreOp)
    return parse_assign(parser, node, BindOp)


def _shared_load_hook(parser: Any, ref: Value, indices: list) -> LoadOp:
    """Subscript-load hook — ``A[i]`` → :class:`LoadOp`."""
    return LoadOp(ref=ref, indices=list(indices))


class _SharedHooks:
    """Hooks shared across all mini-MLIR dialects."""

    __ffi_assign__ = staticmethod(_assign_impl)
    __ffi_make_var__ = staticmethod(_make_value)
    __ffi_default_int_ty__ = arith.i32
    __ffi_default_float_ty__ = arith.f32
    __ffi_default_bool_ty__ = arith.i1
    load = staticmethod(_shared_load_hook)


_shared = _SharedHooks()


# ============================================================================
# LANG_MODULES — parser-ready dialect registry.
#
# Order matters for op fall-through: ``arith`` is consulted before
# ``vector`` for ``+``, so scalar operands go to arith and vector-typed
# operands naturally fall through arith's ``None`` return into vector.
# ============================================================================


LANG_MODULES: dict[str, Any] = {
    "T": T,                 # printer-hardcoded type namespace
    "arith": arith,
    "memref": memref,
    "vector": vector,
    "scf": scf,
    "func": func,
    "builtin": builtin,
    "__shared__": _shared,  # cross-dialect ``__ffi_assign__`` + ``load``
}


def make_var_factory(name: str, ty: Any) -> Value:
    """Legacy ``var_factory=`` shim for :class:`IRParser`."""
    return _make_value(None, name, ty)
