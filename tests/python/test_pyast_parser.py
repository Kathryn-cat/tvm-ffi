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

from tvm_ffi import structural_equal
from tvm_ffi.pyast import to_python
from tvm_ffi.pyast_parser import IRParser, SurfaceObject
from tvm_ffi.testing.testing import (
    TraitToyAssign,
    TraitToyDecoratedFunc,
    TraitToyVar,
)


class _PrimFuncSurface(SurfaceObject):
    """Surface object for @prim_func decorator.

    L2 surface object: sets dialect callbacks on the parser, then
    delegates body parsing to parser.visit_body() recursively.
    """

    def parse_function(self, parser, node):
        with parser.var_table.frame():
            # Set dialect callbacks so visit_body handles stmts correctly
            old_create_var = parser.create_var
            old_make_assign = parser.make_assign
            parser.create_var = lambda name, ann=None: TraitToyVar(name=name)
            parser.make_assign = lambda target, value: TraitToyAssign(target=target, value=value)
            try:
                # Parse params
                params = []
                for arg in node.args:
                    name = arg.lhs.name
                    var = parser.create_var(name)
                    parser.var_table.define(name, var)
                    params.append(var)

                # Parse body — recursively via parser.visit_body
                body_stmts = parser.visit_body(node.body)

                return TraitToyDecoratedFunc(
                    name=node.name.name,
                    params=params,
                    body=body_stmts,
                )
            finally:
                parser.create_var = old_create_var
                parser.make_assign = old_make_assign


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


def test_roundtrip_decorated_func_multi_stmt():
    """@prim_func def kernel(a, b): x = a; y = b; z = x"""
    a = TraitToyVar(name="a")
    b = TraitToyVar(name="b")
    x = TraitToyVar(name="x")
    y = TraitToyVar(name="y")
    z = TraitToyVar(name="z")
    f = TraitToyDecoratedFunc(
        name="kernel",
        params=[a, b],
        body=[
            TraitToyAssign(target=x, value=a),
            TraitToyAssign(target=y, value=b),
            TraitToyAssign(target=z, value=x),
        ],
    )

    text = to_python(f)
    assert "@prim_func" in text
    assert "x = a" in text
    assert "y = b" in text
    assert "z = x" in text

    parser = IRParser(lang_modules={"prim_func": _PrimFuncSurface()})
    f2 = parser.parse(text)
    import pdb

    pdb.set_trace()

    assert structural_equal(f, f2)
