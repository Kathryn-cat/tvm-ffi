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
"""Mini ``I`` dialect — shared IRModule container.

Single-class dialect for the ``@I.ir_module class Name: ...`` decorator
that holds a mixed list of functions from other dialects
(``@T.prim_func``, ``@R.function``). Split into its own file per
``design_docs/parser_auto_registration.md`` §7.7 (module-is-dialect
requires one-module-per-dialect).
"""

# ruff: noqa: D102, N802, UP006, UP045

from __future__ import annotations

from typing import Any, List  # noqa: UP035

from tvm_ffi import Object
from tvm_ffi import ir_traits as tr
from tvm_ffi.dataclasses import py_class
from tvm_ffi.dataclasses import field as dc_field
from tvm_ffi.dialect_autogen import finalize_module


@py_class("mini.ir.IRModule", structural_eq="tree")
class IRModule(Object):
    """``@I.ir_module class Name: <funcs>`` — generic module container.

    ``funcs`` holds a heterogeneous list — TIR :class:`~mini.tir.PrimFunc`
    and / or Relax :class:`~mini.relax.Function` instances can coexist.
    Each inner ``def`` is dispatched via its own decorator handler
    (``@T.prim_func`` → TLang; ``@R.function`` → RLang), so cross-dialect
    modules drop out naturally.
    """

    __ffi_ir_traits__ = tr.FuncTraits(
        "$field:name",
        tr.RegionTraits("$field:funcs", None, None, None),
        None, "I.ir_module", None,
    )
    name: str = dc_field(structural_eq="ignore")
    funcs: List[Any]


# Auto-wires the ``@I.ir_module`` class-decorator handler onto this
# module. With ``FuncTraits.region.def_values=None`` (no params),
# ``_wire_func`` detects the class-form and generates the right
# class-body walker.
finalize_module(__name__)
