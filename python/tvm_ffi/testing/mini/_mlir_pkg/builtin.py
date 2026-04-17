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
"""Mini-MLIR ``builtin`` dialect — module op.

Contains the single :class:`ModuleOp` class that wraps a list of
functions from any mini-MLIR dialect. Class-form :class:`FuncTraits`
(``def_values=None`` in the region) makes ``finalize_module`` wire a
class-decorator handler at ``builtin.module``.
"""

# ruff: noqa: A003, D102, N802, UP006, UP045

from __future__ import annotations

from typing import Any, List  # noqa: UP035

from tvm_ffi import Object
from tvm_ffi import ir_traits as tr
from tvm_ffi.dataclasses import py_class
from tvm_ffi.dataclasses import field as dc_field
from tvm_ffi.dialect_autogen import finalize_module


# ============================================================================
# Ops
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
# The single finalize_module call — auto-injects the ``module``
# class-decorator handler (FuncTraits class-form with
# ``region.def_values=None``).
# ============================================================================


finalize_module(__name__, prefix="builtin")
