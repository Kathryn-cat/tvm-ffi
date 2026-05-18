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

from __future__ import annotations

import tvm_ffi
import tvm_ffi.testing


def test_registered_structural_visit_trace() -> None:
    root = tvm_ffi.get_global_func("testing.make_structural_visit_ir")()
    trace = list(tvm_ffi.get_global_func("testing.structural_visit_trace")(root))

    assert trace == [
        "Func|none",
        "Var(x)|recursive",
        "Var(y)|recursive",
        "Add|none",
        "Var(x)|none",
        "Add|none",
        "Var(y)|none",
        "Const(1)|none",
    ]
