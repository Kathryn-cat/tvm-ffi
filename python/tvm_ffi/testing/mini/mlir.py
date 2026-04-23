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
"""Mini-MLIR — thin re-export shim.

The actual multi-dialect layout lives in :mod:`._mlir_pkg` (one
submodule per dialect per ``design_docs/parser_auto_registration.md``
§7.7). This file exists as an import-aliasing shim because the
editable install's known-source-files registry references
``mini/mlir.py`` at a fixed path — reinstalling to pick up the
subpackage rename would require a full cmake rebuild, which is
impractical in the development loop. The ideal layout is
``mini/mlir/`` (directory), and tests already consume this module
through ``from tvm_ffi.testing.mini import mlir as mm`` so the file
vs. directory distinction is transparent.
"""

# ruff: noqa: F401, F403

from __future__ import annotations

from ._mlir_pkg import *  # re-export everything the __init__ exposes
from ._mlir_pkg import (
    LANG_MODULES,
    ArithLang,
    BindOp,
    BuiltinLang,
    FuncLang,
    MemRefLang,
    ScfLang,
    T,
    VectorLang,
    Value,
    _make_value,
    _operand_type,
    _shared,
    _SharedHooks,
    _TNamespace,
    arith,
    builtin,
    func,
    make_var_factory,
    memref,
    scf,
    vector,
    # Per-class re-exports
    AddFOp,
    AddIOp,
    CallOp,
    CmpIOp,
    ConstantOp,
    FloatType,
    FuncOp,
    IntegerType,
    LoadOp,
    MemRefType,
    ModuleOp,
    MulFOp,
    MulIOp,
    ReturnOp,
    ScfForOp,
    ScfIfOp,
    StoreOp,
    SubFOp,
    SubIOp,
    VectorAddOp,
    VectorType,
)
