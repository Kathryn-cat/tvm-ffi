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
"""Tests for foreign dialect printing through std-kind print builders."""

from __future__ import annotations

from typing import Any, ClassVar

import pytest
from tvm_ffi import Object, method, pyast, std
from tvm_ffi.access_path import AccessPath
from tvm_ffi.dataclasses import body_append, body_prepend, field, py_class

DialectMnemonic = tuple[str, str]


@py_class("testing.foreign_dialect_printer.AddProjected", std_schema=std.Add)
class AddProjected(Object):
    """D0-style foreign add with no std storage inheritance."""

    __ffi_dialect_mnemonic__: ClassVar[DialectMnemonic] = ("tilus", "Add")

    lhs: std.Expr = field(std_field="a")
    rhs: std.Expr = field(std_field="b")


@py_class("testing.foreign_dialect_printer.AddNoMnemonic", std_schema=std.Add)
class AddNoMnemonic(Object):
    """Foreign add that intentionally omits a dialect mnemonic."""

    lhs: std.Expr = field(std_field="a")
    rhs: std.Expr = field(std_field="b")


@py_class("testing.foreign_dialect_printer.AddWithDuplicateStdField", std_schema=std.Add)
class AddWithDuplicateStdField(Object):
    """Invalid projection: two physical fields both claim std field ``a``."""

    __ffi_dialect_mnemonic__: ClassVar[DialectMnemonic] = ("tilus", "BadAdd")

    lhs: std.Expr = field(std_field="a")
    lhs_alias: std.Expr = field(std_field="a")
    rhs: std.Expr = field(std_field="b")


@py_class("testing.foreign_dialect_printer.AddWithExtraField", std_schema=std.Add)
class AddWithExtraField(Object):
    """Invalid projection: an extra reflected field has no print role."""

    __ffi_dialect_mnemonic__: ClassVar[DialectMnemonic] = ("tilus", "AddWithExtra")

    lhs: std.Expr = field(std_field="a")
    rhs: std.Expr = field(std_field="b")
    debug_name: str


@py_class("testing.foreign_dialect_printer.AddWithExactPrinter", std_schema=std.Add)
class AddWithExactPrinter(Object):
    """Exact ``__ffi_text_print__`` must override ``std_schema``."""

    __ffi_dialect_mnemonic__: ClassVar[DialectMnemonic] = ("tilus", "AddExact")

    lhs: std.Expr = field(std_field="a")
    rhs: std.Expr = field(std_field="b")

    def __ffi_text_print__(self, printer: pyast.IRPrinter, path: AccessPath) -> Any:
        return pyast.Id("exact_add")


@py_class("testing.foreign_dialect_printer.TirxAddProjected")
class TirxAddProjected(std.Add, mnemonic="tirx.Add"):
    """D1-style foreign add that inherits std.Add but uses projected field names."""

    operand_a: std.Expr = field(std_field="a")
    operand_b: std.Expr = field(std_field="b")

    def __init__(self, operand_a: std.Expr, operand_b: std.Expr) -> None:
        self.operand_a = operand_a
        self.operand_b = operand_b


@py_class("testing.foreign_dialect_printer.TraceScope", std_schema=std.Scope)
class TraceScope(Object):
    """Scope-like D0 node with body prepend and append fields."""

    __ffi_dialect_mnemonic__: ClassVar[DialectMnemonic] = ("tilus", "TraceScope")

    body: list[std.Stmt] = field(std_field="body")
    before: list[std.Stmt] = field(print=body_prepend("body", order=10))
    after: list[std.Stmt] = field(print=body_append("body", order=20))


@py_class("testing.foreign_dialect_printer.RenderedScope", std_schema=std.Scope)
class RenderedScope(Object):
    """Scope-like D0 node whose prepend syntax is supplied by a render method."""

    __ffi_dialect_mnemonic__: ClassVar[DialectMnemonic] = ("tilus", "RenderedScope")

    body: list[std.Stmt] = field(std_field="body")
    tag: str = field(print=body_prepend("body", order=10, render="print_tag"))

    @method
    def print_tag(self, printer: pyast.IRPrinter, path: AccessPath, tag: str) -> pyast.Stmt:
        del printer, path
        return pyast.ExprStmt(
            pyast.Call(pyast.Attr(pyast.Id("T"), "tag"), [pyast.Literal(tag)], [], [])
        )


@py_class("testing.foreign_dialect_printer.ForeignVar", std_schema=std.Var)
class ForeignVar(Object):
    __ffi_dialect_mnemonic__: ClassVar[DialectMnemonic] = ("tilus", "Var")

    name: str = field(std_field="name")
    ty: Any = field(default=None, std_field="ty")


@py_class("testing.foreign_dialect_printer.ForeignFunc", std_schema=std.Func)
class ForeignFunc(Object):
    __ffi_dialect_mnemonic__: ClassVar[DialectMnemonic] = ("tilus", "Func")

    symbol: str = field(std_field="symbol")
    args: list[Any] = field(std_field="args")
    body: list[Any] = field(std_field="body")
    attrs: Any = field(default=None, std_field="attrs")
    ret_type: Any = field(default=None, std_field="ret_type")


@py_class("testing.foreign_dialect_printer.ForeignModule", std_schema=std.Module)
class ForeignModule(Object):
    __ffi_dialect_mnemonic__: ClassVar[DialectMnemonic] = ("tilus", "Module")

    funcs: list[Any] = field(std_field="funcs")


@py_class("testing.foreign_dialect_printer.ForeignRange", std_schema=std.Range)
class ForeignRange(Object):
    __ffi_dialect_mnemonic__: ClassVar[DialectMnemonic] = ("tilus", "Range")

    start: Any = field(std_field="start")
    stop: Any = field(std_field="stop")
    step: Any = field(std_field="step")


@py_class("testing.foreign_dialect_printer.ForeignAnyTy", std_schema=std.AnyTy)
class ForeignAnyTy(Object):
    __ffi_dialect_mnemonic__: ClassVar[DialectMnemonic] = ("tilus", "Any")


@py_class("testing.foreign_dialect_printer.ForeignPrimTy", std_schema=std.PrimTy)
class ForeignPrimTy(Object):
    __ffi_dialect_mnemonic__: ClassVar[DialectMnemonic] = ("tilus", "Prim")

    dtype: Any = field(std_field="dtype")


@py_class("testing.foreign_dialect_printer.ForeignTupleTy", std_schema=std.TupleTy)
class ForeignTupleTy(Object):
    __ffi_dialect_mnemonic__: ClassVar[DialectMnemonic] = ("tilus", "Tuple")

    fields: list[Any] = field(std_field="fields")


@py_class("testing.foreign_dialect_printer.ForeignTensorTy", std_schema=std.TensorTy)
class ForeignTensorTy(Object):
    __ffi_dialect_mnemonic__: ClassVar[DialectMnemonic] = ("tilus", "Tensor")

    shape: list[Any] = field(std_field="shape")
    dtype: Any = field(std_field="dtype")


@py_class("testing.foreign_dialect_printer.ForeignIntImm", std_schema=std.IntImm)
class ForeignIntImm(Object):
    __ffi_dialect_mnemonic__: ClassVar[DialectMnemonic] = ("tilus", "IntImm")

    value: int = field(std_field="value")


@py_class("testing.foreign_dialect_printer.ForeignFloatImm", std_schema=std.FloatImm)
class ForeignFloatImm(Object):
    __ffi_dialect_mnemonic__: ClassVar[DialectMnemonic] = ("tilus", "FloatImm")

    value: float = field(std_field="value")


@py_class("testing.foreign_dialect_printer.ForeignStringImm", std_schema=std.StringImm)
class ForeignStringImm(Object):
    __ffi_dialect_mnemonic__: ClassVar[DialectMnemonic] = ("tilus", "StringImm")

    value: str = field(std_field="value")


@py_class("testing.foreign_dialect_printer.ForeignNot", std_schema=std.Not)
class ForeignNot(Object):
    __ffi_dialect_mnemonic__: ClassVar[DialectMnemonic] = ("tilus", "Not")

    operand: Any = field(std_field="operand")


@py_class("testing.foreign_dialect_printer.ForeignLoad", std_schema=std.Load)
class ForeignLoad(Object):
    __ffi_dialect_mnemonic__: ClassVar[DialectMnemonic] = ("tilus", "Load")

    lhs: Any = field(std_field="lhs")
    indices: list[Any] = field(std_field="indices")


@py_class("testing.foreign_dialect_printer.ForeignCast", std_schema=std.Cast)
class ForeignCast(Object):
    __ffi_dialect_mnemonic__: ClassVar[DialectMnemonic] = ("tilus", "Cast")

    ty: Any = field(std_field="ty")
    value: Any = field(std_field="value")


@py_class("testing.foreign_dialect_printer.ForeignCall", std_schema=std.Call)
class ForeignCall(Object):
    __ffi_dialect_mnemonic__: ClassVar[DialectMnemonic] = ("tilus", "Call")

    callee: Any = field(std_field="callee")
    args: list[Any] = field(std_field="args")
    attr: Any = field(default=None, std_field="attr")


@py_class("testing.foreign_dialect_printer.ForeignIfStmt", std_schema=std.IfStmt)
class ForeignIfStmt(Object):
    __ffi_dialect_mnemonic__: ClassVar[DialectMnemonic] = ("tilus", "IfStmt")

    cond: Any = field(std_field="cond")
    then_body: list[Any] = field(std_field="then_body")
    else_body: list[Any] = field(std_field="else_body")


@py_class("testing.foreign_dialect_printer.ForeignBindExpr", std_schema=std.BindExpr)
class ForeignBindExpr(Object):
    __ffi_dialect_mnemonic__: ClassVar[DialectMnemonic] = ("tilus", "BindExpr")

    vars: list[Any] = field(std_field="vars")
    expr: Any = field(std_field="expr")
    attrs: Any = field(default=None, std_field="attrs")


@py_class("testing.foreign_dialect_printer.ForeignVarDef", std_schema=std.VarDef)
class ForeignVarDef(Object):
    __ffi_dialect_mnemonic__: ClassVar[DialectMnemonic] = ("tilus", "VarDef")

    vars: list[Any] = field(std_field="vars")
    attrs: Any = field(default=None, std_field="attrs")


@py_class("testing.foreign_dialect_printer.ForeignScope", std_schema=std.Scope)
class ForeignScope(Object):
    __ffi_dialect_mnemonic__: ClassVar[DialectMnemonic] = ("tilus", "Scope")

    binds: list[Any] = field(std_field="binds")
    body: list[Any] = field(std_field="body")
    attrs: Any = field(default=None, std_field="attrs")


@py_class("testing.foreign_dialect_printer.ForeignFor", std_schema=std.For)
class ForeignFor(Object):
    __ffi_dialect_mnemonic__: ClassVar[DialectMnemonic] = ("tilus", "For")

    start: Any = field(std_field="start")
    stop: Any = field(std_field="stop")
    step: Any = field(std_field="step")
    vars: list[Any] = field(std_field="vars")
    body: list[Any] = field(std_field="body")
    attrs: Any = field(default=None, std_field="attrs")


@py_class("testing.foreign_dialect_printer.ForeignWhile", std_schema=std.While)
class ForeignWhile(Object):
    __ffi_dialect_mnemonic__: ClassVar[DialectMnemonic] = ("tilus", "While")

    cond: Any = field(std_field="cond")
    body: list[Any] = field(std_field="body")
    attrs: Any = field(default=None, std_field="attrs")


@py_class("testing.foreign_dialect_printer.ForeignStore", std_schema=std.Store)
class ForeignStore(Object):
    __ffi_dialect_mnemonic__: ClassVar[DialectMnemonic] = ("tilus", "Store")

    lhs: Any = field(std_field="lhs")
    indices: list[Any] = field(std_field="indices")
    rhs: Any = field(std_field="rhs")


@py_class("testing.foreign_dialect_printer.ForeignAssert", std_schema=std.Assert)
class ForeignAssert(Object):
    __ffi_dialect_mnemonic__: ClassVar[DialectMnemonic] = ("tilus", "Assert")

    cond: Any = field(std_field="cond")
    attrs: Any = field(default=None, std_field="attrs")


@py_class("testing.foreign_dialect_printer.ForeignReturn", std_schema=std.Return)
class ForeignReturn(Object):
    __ffi_dialect_mnemonic__: ClassVar[DialectMnemonic] = ("tilus", "Return")

    exprs: list[Any] = field(std_field="exprs")
    attrs: Any = field(default=None, std_field="attrs")


@py_class("testing.foreign_dialect_printer.ForeignYield", std_schema=std.Yield)
class ForeignYield(Object):
    __ffi_dialect_mnemonic__: ClassVar[DialectMnemonic] = ("tilus", "Yield")

    exprs: list[Any] = field(std_field="exprs")
    attrs: Any = field(default=None, std_field="attrs")


@py_class("testing.foreign_dialect_printer.ForeignBreak", std_schema=std.Break)
class ForeignBreak(Object):
    __ffi_dialect_mnemonic__: ClassVar[DialectMnemonic] = ("tilus", "Break")

    attrs: Any = field(default=None, std_field="attrs")


@py_class("testing.foreign_dialect_printer.ForeignContinue", std_schema=std.Continue)
class ForeignContinue(Object):
    __ffi_dialect_mnemonic__: ClassVar[DialectMnemonic] = ("tilus", "Continue")

    attrs: Any = field(default=None, std_field="attrs")


@py_class("testing.foreign_dialect_printer.ForeignDictAttrs", std_schema=std.DictAttrs)
class ForeignDictAttrs(Object):
    __ffi_dialect_mnemonic__: ClassVar[DialectMnemonic] = ("tilus", "DictAttrs")

    values: dict[str, Any] = field(std_field="values")


def _vars() -> tuple[std.Ty, std.Var, std.Var]:
    i32 = std.PrimTy("int32")
    return i32, std.Var(i32, "x"), std.Var(i32, "y")


def test_d0_std_schema_prints_binary_sugar() -> None:
    _, x, y = _vars()

    assert pyast.to_python(AddProjected(x, y)) == "x + y"


def test_d1_inherited_std_kind_prints_projected_fields_without_base_storage_init() -> None:
    _, x, y = _vars()

    assert pyast.to_python(TirxAddProjected(x, y)) == "x + y"


def test_exact_text_print_wins_over_std_schema() -> None:
    _, x, y = _vars()

    assert pyast.to_python(AddWithExactPrinter(x, y)) == "exact_add"


def test_missing_dialect_mnemonic_is_reported() -> None:
    _, x, y = _vars()

    with pytest.raises(ValueError, match=r"No .*__ffi_dialect_mnemonic__.* registered"):
        pyast.to_python(AddNoMnemonic(x, y))


def test_std_schema_requires_registered_std_schema() -> None:
    with pytest.raises(TypeError, match="does not declare __ffi_std_schema__"):

        @py_class("testing.foreign_dialect_printer.BadStdSchema", std_schema=Object)
        class BadStdSchema(Object):
            pass


def test_duplicate_std_field_projection_is_reported() -> None:
    _, x, y = _vars()

    with pytest.raises(ValueError, match=r"Multiple fields.*std field `a`"):
        pyast.to_python(AddWithDuplicateStdField(x, x, y))


def test_unconsumed_foreign_field_is_reported() -> None:
    _, x, y = _vars()

    with pytest.raises(ValueError, match="not consumed by std field resolution or print roles"):
        pyast.to_python(AddWithExtraField(x, y, "debug"))


def test_scope_body_prepend_and_append_roles() -> None:
    _, x, y = _vars()
    cond = x < y
    node = TraceScope(
        body=[std.Return(x)],
        before=[std.Assert(cond)],
        after=[std.Return(y)],
    )

    assert pyast.to_python(node) == "assert x < y\nreturn x\nreturn y"


def test_scope_field_render_method() -> None:
    _, x, _ = _vars()
    node = RenderedScope(body=[std.Return(x)], tag="demo")

    assert pyast.to_python(node) == 'T.tag("demo")\nreturn x'


def test_d0_std_schema_prints_type_literal_and_expr_builders() -> None:
    i32, x, _ = _vars()

    assert pyast.to_python(ForeignAnyTy()) == "tilus.Any"
    assert pyast.to_python(ForeignPrimTy("int32")) == "tilus.i32"
    assert pyast.to_python(ForeignTupleTy([i32])) == "tilus.Tuple[std.i32]"
    assert pyast.to_python(ForeignTensorTy([4], "int32")) == "std.i32[4]"
    assert pyast.to_python(ForeignIntImm(3)) == "3"
    assert pyast.to_python(ForeignFloatImm(1.5)) == "1.5"
    assert pyast.to_python(ForeignStringImm("demo")) == '"demo"'
    assert pyast.to_python(ForeignRange(1, 4, 2)) == "1:4:2"
    assert pyast.to_python(ForeignNot(std.IntImm(i32, 1))) == "not std.i32(1)"
    assert pyast.to_python(ForeignLoad(x, [ForeignRange(1, None, None)])) == "x[1]"
    assert pyast.to_python(ForeignCast(i32, std.IntImm(i32, 1))) == "std.i32(std.i32(1))"
    assert (
        pyast.to_python(ForeignCall("callee", [std.IntImm(i32, 1)], ForeignDictAttrs({"tag": "v"})))
        == 'tilus.Call(callee, std.i32(1), tag="v")'
    )
    assert pyast.to_python(ForeignDictAttrs({"z": 2, "a": 1})) == "tilus.DictAttrs(a=1, z=2)"


def test_d0_std_schema_prints_stmt_scope_and_function_builders() -> None:
    i32, x, y = _vars()
    cond = x < std.IntImm(i32, 2)

    assert pyast.to_python(ForeignIfStmt(cond, [std.Return(x)], [std.Return(y)])) == (
        "if x < std.i32(2):\n  return x\nelse:\n  return y"
    )
    assert pyast.to_python(ForeignBindExpr([x], std.IntImm(i32, 1))) == "x = std.i32(1)"
    assert pyast.to_python(ForeignVarDef([x])) == "x = tilus.VarDef(std.i32)"
    assert (
        pyast.to_python(ForeignStore(x, [ForeignRange(1, None, None)], std.IntImm(i32, 3)))
        == "x[1] = std.i32(3)"
    )
    assert pyast.to_python(ForeignAssert(cond)) == "assert x < std.i32(2)"
    assert pyast.to_python(ForeignReturn([x])) == "return x"
    assert pyast.to_python(ForeignYield([x])) == "yield x"
    assert pyast.to_python(ForeignBreak()) == "tilus.Break()"
    assert pyast.to_python(ForeignContinue()) == "tilus.Continue()"

    scope = ForeignScope(
        [ForeignVarDef([x])],
        [std.Return(x)],
    )
    assert pyast.to_python(scope) == "with tilus.Scope(tilus.VarDef(std.i32)) as x:\n  return x"

    loop = ForeignFor(
        start=0,
        stop=2,
        step=None,
        vars=[x],
        body=[std.Return(x)],
    )
    assert pyast.to_python(loop) == "for x in range(0, 2):\n  return x"

    while_loop = ForeignWhile(cond, [std.Assert(cond)])
    assert pyast.to_python(while_loop) == "while x < std.i32(2):\n  assert x < std.i32(2)"

    foreign_arg = ForeignVar("arg", i32)
    func = ForeignFunc(
        "main",
        [foreign_arg],
        [ForeignReturn([foreign_arg])],
        ret_type=i32,
    )
    assert pyast.to_python(func) == (
        "@tilus.func\ndef main(arg: std.i32) -> std.i32:\n  return arg"
    )
    assert pyast.to_python(ForeignModule([func])) == (
        "@tilus.Module\n"
        "class MyModule:\n"
        "  @tilus.func\n"
        "  def main(arg: std.i32) -> std.i32:\n"
        "    return arg"
    )


def test_field_render_method_must_be_ffi_method() -> None:
    with pytest.raises(NameError, match="must be decorated with @method"):

        @py_class("testing.foreign_dialect_printer.RenderWithoutMethod", std_schema=std.Scope)
        class RenderWithoutMethod(Object):
            __ffi_dialect_mnemonic__: ClassVar[DialectMnemonic] = ("tilus", "RenderWithoutMethod")

            body: list[std.Stmt] = field(std_field="body")
            tag: str = field(print=body_prepend("body", order=10, render="print_tag"))

            def print_tag(self, printer: pyast.IRPrinter, path: AccessPath, tag: str) -> pyast.Stmt:
                del printer, path, tag
                return pyast.ExprStmt(pyast.Id("unregistered"))
