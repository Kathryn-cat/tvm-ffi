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
"""Roundtrip tests: to_python → parse → structural_equal."""

from __future__ import annotations

from tvm_ffi import pyast, structural_equal
from tvm_ffi.pyast import to_python
from tvm_ffi.pyast_parser import IRParser, SurfaceObject
from tvm_ffi.testing.testing import (
    TraitToyAdd,
    TraitToyAssign,
    TraitToyAssertNode,
    TraitToyDecoratedFunc,
    TraitToyDecoratedModule,
    TraitToyForNode,
    TraitToyForRangeNode,
    TraitToyFuncNode,
    TraitToyIfElseNode,
    TraitToyIfNode,
    TraitToyLt,
    TraitToyModuleNode,
    TraitToyMul,
    TraitToyStore,
    TraitToyVar,
    TraitToyWhileNode,
    TraitToyWithNode,
)


# ============================================================================
# Surface objects for testing dialect
# ============================================================================


class _PrimFuncSurface(SurfaceObject):
    """Surface object for @prim_func decorator."""

    def parse_function(self, parser, node):
        with parser.var_table.frame():
            old = _save_and_set_dialect(parser)
            try:
                params = []
                for arg in node.args:
                    var = parser.create_var(arg.lhs.name)
                    parser.var_table.define(arg.lhs.name, var)
                    params.append(var)
                body_stmts = parser.visit_body(node.body)
                return TraitToyDecoratedFunc(
                    name=node.name.name, params=params, body=body_stmts,
                )
            finally:
                _restore_dialect(parser, old)


class _LaunchSurface(SurfaceObject):
    """Surface object for `with launch() as ctx:` — TraitToyWithNode."""

    def parse_with(self, parser, node):
        as_var = parser.create_var(node.lhs.name)
        parser.var_table.define(node.lhs.name, as_var)
        body_stmts = parser.visit_body(node.body)
        return TraitToyWithNode(as_var=as_var, body=body_stmts)


class _IRModuleSurface(SurfaceObject):
    """Surface object for @ir_module class."""

    def parse_function(self, parser, node):
        """FuncTraits renders Module as ClassAST which is also Function-dispatched."""
        with parser.var_table.frame():
            old = _save_and_set_dialect(parser)
            try:
                body_stmts = parser.visit_body(node.body)
                return TraitToyDecoratedModule(
                    name=node.name.name, body=body_stmts,
                )
            finally:
                _restore_dialect(parser, old)


# ============================================================================
# Dialect callback helpers
# ============================================================================


def _save_and_set_dialect(parser):
    """Save current dialect state and set testing-IR callbacks."""
    old = (
        parser.create_var, parser.make_assign, parser.make_for,
        parser.make_store, parser.handle_if, parser.handle_while,
        parser.handle_assert,
    )
    parser.create_var = lambda name, ann=None: TraitToyVar(name=name)
    parser.make_assign = lambda target, value: TraitToyAssign(
        target=target, value=value,
    )
    parser.make_for = _make_for
    parser.make_store = lambda target, value, indices: TraitToyStore(
        buf=target, val=value, indices=indices,
    )
    parser.handle_if = _handle_if
    parser.handle_while = _handle_while
    parser.handle_assert = _handle_assert
    parser.make_func = _make_func
    return old


def _restore_dialect(parser, old):
    (
        parser.create_var, parser.make_assign, parser.make_for,
        parser.make_store, parser.handle_if, parser.handle_while,
        parser.handle_assert,
    ) = old


def _make_func(name, params, body, ret):
    return TraitToyFuncNode(name=name, params=params, body=body, ret=ret)


def _make_for(loop_var, start, end, step, body):
    if start == 0 and step == 1:
        return TraitToyForNode(loop_var=loop_var, extent=end, body=body)
    return TraitToyForRangeNode(
        loop_var=loop_var, start=start, end=end, step=step, body=body,
    )


def _handle_if(parser, node):
    cond = parser.eval_expr(node.cond)
    then_body = parser.visit_body(node.then_branch)
    if node.else_branch and len(node.else_branch) > 0:
        else_body = parser.visit_body(node.else_branch)
        return TraitToyIfElseNode(
            cond=cond, then_body=then_body, else_body=else_body,
        )
    return TraitToyIfNode(cond=cond, then_body=then_body)


def _handle_while(parser, node):
    cond = parser.eval_expr(node.cond)
    body = parser.visit_body(node.body)
    return TraitToyWhileNode(cond=cond, body=body)


def _handle_assert(parser, node):
    cond = parser.eval_expr(node.cond)
    msg = None
    if node.msg is not None:
        msg = parser.eval_expr(node.msg)
    return TraitToyAssertNode(cond=cond, msg=msg)


def _make_parser():
    parser = IRParser(lang_modules={
        "prim_func": _PrimFuncSurface(),
        "launch": _LaunchSurface,  # CLASS — launch() creates instance
        "ir_module": _IRModuleSurface(),
    })
    # Set dialect defaults so bare def/for/if work at top level
    _save_and_set_dialect(parser)
    return parser


# ============================================================================
# Level 0: Decorated func (already working)
# ============================================================================


def test_L0_roundtrip_decorated_func():
    a = TraitToyVar(name="a")
    x = TraitToyVar(name="x")
    f = TraitToyDecoratedFunc(
        name="kernel", params=[a],
        body=[TraitToyAssign(target=x, value=a)],
    )
    text = to_python(f)
    f2 = _make_parser().parse(text)
    assert structural_equal(f, f2)


def test_L0_roundtrip_decorated_func_multi_stmt():
    a, b = TraitToyVar(name="a"), TraitToyVar(name="b")
    x, y, z = TraitToyVar(name="x"), TraitToyVar(name="y"), TraitToyVar(name="z")
    f = TraitToyDecoratedFunc(
        name="kernel", params=[a, b],
        body=[
            TraitToyAssign(target=x, value=a),
            TraitToyAssign(target=y, value=b),
            TraitToyAssign(target=z, value=x),
        ],
    )
    text = to_python(f)
    f2 = _make_parser().parse(text)
    assert structural_equal(f, f2)


# ============================================================================
# Level 2: Bare def handler
# ============================================================================


def test_L2_roundtrip_bare_func():
    a = TraitToyVar(name="a")
    x = TraitToyVar(name="x")
    f = TraitToyFuncNode(
        name="f", params=[a],
        body=[TraitToyAssign(target=x, value=a)], ret=x,
    )
    text = to_python(f)
    f2 = _make_parser().parse(text)
    assert structural_equal(f, f2)


def test_L2_roundtrip_bare_func_binop():
    a, b = TraitToyVar(name="a"), TraitToyVar(name="b")
    x = TraitToyVar(name="x")
    f = TraitToyFuncNode(
        name="my_func", params=[a, b],
        body=[TraitToyAssign(target=x, value=TraitToyAdd(lhs=a, rhs=b))],
        ret=x,
    )
    text = to_python(f)
    f2 = _make_parser().parse(text)
    assert structural_equal(f, f2)


# ============================================================================
# Level 3: For + range
# ============================================================================


def test_L3_roundtrip_for():
    a = TraitToyVar(name="a")
    i = TraitToyVar(name="i")
    x = TraitToyVar(name="x")
    f = TraitToyFuncNode(
        name="fn", params=[a],
        body=[TraitToyForNode(
            loop_var=i, extent=10,
            body=[TraitToyAssign(target=x, value=a)],
        )],
        ret=x,
    )
    text = to_python(f)
    f2 = _make_parser().parse(text)
    assert structural_equal(f, f2)


def test_L3_roundtrip_for_range():
    a = TraitToyVar(name="a")
    i = TraitToyVar(name="i")
    x = TraitToyVar(name="x")
    f = TraitToyFuncNode(
        name="fn", params=[a],
        body=[TraitToyForRangeNode(
            loop_var=i, start=2, end=10, step=1,
            body=[TraitToyAssign(target=x, value=a)],
        )],
        ret=x,
    )
    text = to_python(f)
    f2 = _make_parser().parse(text)
    assert structural_equal(f, f2)


# ============================================================================
# Level 4: SyntaxContext (if/while/assert)
# ============================================================================


def test_L4_roundtrip_if():
    a = TraitToyVar(name="a")
    i = TraitToyVar(name="i")
    n = TraitToyVar(name="n")
    x = TraitToyVar(name="x")
    f = TraitToyFuncNode(
        name="fn", params=[a, i, n],
        body=[TraitToyIfNode(
            cond=TraitToyLt(lhs=i, rhs=n),
            then_body=[TraitToyAssign(target=x, value=a)],
        )],
        ret=x,
    )
    text = to_python(f)
    f2 = _make_parser().parse(text)
    assert structural_equal(f, f2)


def test_L4_roundtrip_if_else():
    a, b = TraitToyVar(name="a"), TraitToyVar(name="b")
    i, n = TraitToyVar(name="i"), TraitToyVar(name="n")
    x = TraitToyVar(name="x")
    f = TraitToyFuncNode(
        name="fn", params=[a, b, i, n],
        body=[TraitToyIfElseNode(
            cond=TraitToyLt(lhs=i, rhs=n),
            then_body=[TraitToyAssign(target=x, value=a)],
            else_body=[TraitToyAssign(target=x, value=b)],
        )],
        ret=x,
    )
    text = to_python(f)
    f2 = _make_parser().parse(text)
    assert structural_equal(f, f2)


def test_L4_roundtrip_while():
    a = TraitToyVar(name="a")
    i, n = TraitToyVar(name="i"), TraitToyVar(name="n")
    x = TraitToyVar(name="x")
    f = TraitToyFuncNode(
        name="fn", params=[a, i, n],
        body=[TraitToyWhileNode(
            cond=TraitToyLt(lhs=i, rhs=n),
            body=[TraitToyAssign(target=x, value=a)],
        )],
        ret=x,
    )
    text = to_python(f)
    f2 = _make_parser().parse(text)
    assert structural_equal(f, f2)


def test_L4_roundtrip_assert():
    a, b = TraitToyVar(name="a"), TraitToyVar(name="b")
    f = TraitToyFuncNode(
        name="fn", params=[a, b],
        body=[TraitToyAssertNode(
            cond=TraitToyLt(lhs=a, rhs=b), msg="oops",
        )],
        ret=a,
    )
    text = to_python(f)
    f2 = _make_parser().parse(text)
    assert structural_equal(f, f2)


# ============================================================================
# Level 5: With + launch surface object
# ============================================================================


def test_L5_roundtrip_with():
    a = TraitToyVar(name="a")
    ctx = TraitToyVar(name="ctx")
    x = TraitToyVar(name="x")
    f = TraitToyFuncNode(
        name="fn", params=[a],
        body=[TraitToyWithNode(
            as_var=ctx,
            body=[TraitToyAssign(target=x, value=a)],
        )],
        ret=x,
    )
    text = to_python(f)
    f2 = _make_parser().parse(text)
    assert structural_equal(f, f2)


# ============================================================================
# Level 6: Subscript store
# ============================================================================


def test_L6_roundtrip_store():
    buf = TraitToyVar(name="A")
    i = TraitToyVar(name="i")
    v = TraitToyVar(name="v")
    f = TraitToyFuncNode(
        name="fn", params=[buf, i, v],
        body=[TraitToyStore(buf=buf, val=v, indices=[i])],
        ret=v,
    )
    text = to_python(f)
    f2 = _make_parser().parse(text)
    assert structural_equal(f, f2)


# ============================================================================
# Level 10: Nested (func + for + if + assign + operators)
# ============================================================================


def test_L10_roundtrip_nested():
    a, b, n = TraitToyVar(name="a"), TraitToyVar(name="b"), TraitToyVar(name="n")
    i = TraitToyVar(name="i")
    temp = TraitToyVar(name="temp")
    result = TraitToyVar(name="result")
    f = TraitToyFuncNode(
        name="compute", params=[a, b, n],
        body=[TraitToyForNode(
            loop_var=i, extent=100,
            body=[TraitToyIfNode(
                cond=TraitToyLt(lhs=i, rhs=n),
                then_body=[
                    TraitToyAssign(target=temp, value=TraitToyAdd(lhs=a, rhs=b)),
                    TraitToyAssign(target=result, value=TraitToyMul(lhs=temp, rhs=i)),
                ],
            )],
        )],
        ret=result,
    )
    text = to_python(f)
    f2 = _make_parser().parse(text)
    assert structural_equal(f, f2)
