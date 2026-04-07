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
"""Roundtrip test: to_python → parse → structural_equal."""

from __future__ import annotations

from tvm_ffi import pyast, structural_equal
from tvm_ffi.pyast import to_python
from tvm_ffi.pyast_parser import IRParser, SurfaceObject
from tvm_ffi.testing.testing import (
    TraitToyAssign,
    TraitToyDecoratedFunc,
    TraitToyVar,
)


class _PrimFuncSurface(SurfaceObject):
    """Surface object for @prim_func decorator.

    This is an L2 surface object: it has dialect-specific knowledge
    about what body statements mean (assignments produce TraitToyAssign).
    """

    def parse_function(self, parser, node):
        with parser.var_table.frame():
            # Parse params
            params = []
            for arg in node.args:
                name = arg.lhs.name
                var = TraitToyVar(name=name)
                parser.var_table.define(name, var)
                params.append(var)

            # Parse body — assignments produce TraitToyAssign
            body_stmts = []
            for stmt in node.body:
                if isinstance(stmt, pyast.Assign) and stmt.rhs is not None:
                    target = TraitToyVar(name=stmt.lhs.name)
                    value = parser.eval_expr(stmt.rhs)
                    parser.var_table.define(stmt.lhs.name, target)
                    body_stmts.append(
                        TraitToyAssign(target=target, value=value)
                    )
                else:
                    result = parser.visit_stmt(stmt)
                    if result is not None:
                        body_stmts.append(result)

            return TraitToyDecoratedFunc(
                name=node.name.name,
                params=params,
                body=body_stmts,
            )


def test_roundtrip_decorated_func():
    """@prim_func def kernel(a): x = a"""
    # Build IR
    a = TraitToyVar(name="a")
    x = TraitToyVar(name="x")
    assign = TraitToyAssign(target=x, value=a)
    f = TraitToyDecoratedFunc(name="kernel", params=[a], body=[assign])

    # Print
    text = to_python(f)
    assert "@prim_func" in text
    assert "def kernel(a):" in text
    assert "x = a" in text

    # Parse
    parser = IRParser(lang_modules={"prim_func": _PrimFuncSurface()})
    f2 = parser.parse(text)

    # Compare
    assert structural_equal(f, f2)
