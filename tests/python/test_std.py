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

from typing import Any, cast

import pytest
import tvm_ffi
from tvm_ffi import core, pyast, std
from tvm_ffi.access_path import AccessPath
from tvm_ffi.dataclasses import fields


class TestAnyTy:
    def test_constructor(self) -> None:
        node = std.AnyTy()

        assert isinstance(node, std.AnyTy)
        assert tuple(field.name for field in fields(std.AnyTy)) == ()

    def test_text_format(self) -> None:
        node = std.AnyTy()

        assert node.text() == "Any"

    def test_structural_equality(self) -> None:
        lhs = std.AnyTy()
        rhs = std.AnyTy()
        different = std.PrimTy("int32")

        assert tvm_ffi.structural_equal(lhs, rhs)
        assert tvm_ffi.structural_hash(lhs) == tvm_ffi.structural_hash(rhs)
        assert not tvm_ffi.structural_equal(lhs, different)


class TestPrimTy:
    def test_constructor(self) -> None:
        node = std.PrimTy("int32")

        assert isinstance(node, std.PrimTy)
        assert tuple(field.name for field in fields(std.PrimTy)) == ("dtype",)
        assert node.dtype == tvm_ffi.int32

    def test_text_format(self) -> None:
        node = std.PrimTy("int32")

        assert node.text() == "i32"
        assert std.PrimTy("float16").text() == "f16"
        assert std.PrimTy("bfloat16").text() == "bf16"
        assert std.PrimTy("uint8").text() == "u8"
        assert std.PrimTy("float8_e4m3fn").text() == "f8_e4m3fn"

    def test_structural_equality(self) -> None:
        lhs = std.PrimTy("int32")
        rhs = std.PrimTy("int32")
        different = std.PrimTy("float32")

        assert tvm_ffi.structural_equal(lhs, rhs)
        assert tvm_ffi.structural_hash(lhs) == tvm_ffi.structural_hash(rhs)
        assert not tvm_ffi.structural_equal(lhs, different)


class TestAttrs:
    def test_from_any_converts_none_to_empty_dict_attrs(self) -> None:
        node = core._to_py_class_value(core.TypeSchema.from_annotation(std.Attrs).convert(None))

        assert isinstance(node, std.DictAttrs)
        assert dict(node.values) == {}
        assert node.text() == "std.DictAttrs()"

    def test_from_any_converts_python_dict_to_dict_attrs(self) -> None:
        node = core._to_py_class_value(
            core.TypeSchema.from_annotation(std.Attrs).convert({"tag": "demo"})
        )

        assert isinstance(node, std.DictAttrs)
        assert dict(node.values) == {"tag": "demo"}
        assert node.text() == 'std.DictAttrs(tag="demo")'


def test_std_base_classes_cannot_be_constructed_directly() -> None:
    for cls in [std.Node, std.Ty, std.Stmt, std.Expr, std.Attrs, std.Structure, std.Bind]:
        ctor = cast(Any, cls)
        with pytest.raises(TypeError, match="cannot be constructed directly"):
            ctor()


class TestMnemonic:
    def test_base_classes_do_not_have_ffi_mnemonics(self) -> None:
        for cls in [std.Node, std.Ty, std.Stmt, std.Expr, std.Attrs, std.Structure, std.Bind]:
            cls_any = cast(Any, cls)
            info = cls_any.__tvm_ffi_type_info__

            assert "__ffi_mnemonic__" not in cls.__dict__
            assert not hasattr(cls, "__ffi_mnemonic__")
            assert core._lookup_type_attr(info.type_index, "__ffi_mnemonic__") is None

    def test_concrete_classes_have_exact_ffi_mnemonics(self) -> None:
        cases = [
            (std.AnyTy, "std$AnyTy"),
            (std.PrimTy, "std$PrimTy"),
            (std.TupleType, "std$TupleType"),
            (std.TensorTy, "std$TensorTy"),
            (std.Range, "std$Range"),
            (std.DictAttrs, "std$DictAttrs"),
            (std.Var, "std$Var"),
            (std.Func, "std$Func"),
            (std.Module, "std$Module"),
            (std.IntImm, "std$IntImm"),
            (std.FloatImm, "std$FloatImm"),
            (std.StringImm, "std$StringImm"),
            (std.Add, "std$Add"),
            (std.Sub, "std$Sub"),
            (std.Mul, "std$Mul"),
            (std.FloorDiv, "std$FloorDiv"),
            (std.FloorMod, "std$FloorMod"),
            (std.Min, "std$Min"),
            (std.Max, "std$Max"),
            (std.Eq, "std$Eq"),
            (std.Ne, "std$Ne"),
            (std.Le, "std$Le"),
            (std.Ge, "std$Ge"),
            (std.Gt, "std$Gt"),
            (std.Lt, "std$Lt"),
            (std.And, "std$And"),
            (std.Or, "std$Or"),
            (std.Not, "std$Not"),
            (std.Load, "std$Load"),
            (std.Cast, "std$Cast"),
            (std.Call, "std$Call"),
            (std.IfStmt, "std$IfStmt"),
            (std.Scope, "std$Scope"),
            (std.For, "std$For"),
            (std.While, "std$While"),
            (std.ExprBind, "std$ExprBind"),
            (std.VarDef, "std$VarDef"),
            (std.Store, "std$Store"),
            (std.Return, "std$Return"),
            (std.Yield, "std$Yield"),
            (std.Break, "std$Break"),
            (std.Continue, "std$Continue"),
        ]

        for cls, mnemonic in cases:
            cls_any = cast(Any, cls)
            info = cls_any.__tvm_ffi_type_info__

            assert cls_any.__ffi_mnemonic__ == mnemonic
            assert core._lookup_type_attr(info.type_index, "__ffi_mnemonic__") == mnemonic

    def test_every_exported_concrete_node_has_registered_mnemonic(self) -> None:
        abstract = {std.Node, std.Ty, std.Stmt, std.Expr, std.Attrs, std.Structure, std.Bind}
        concrete_classes = []
        for name in std.__all__:
            cls = getattr(std, name)
            if isinstance(cls, type) and issubclass(cls, std.Node) and cls not in abstract:
                concrete_classes.append(cls)

        assert len(concrete_classes) == 42
        for cls in concrete_classes:
            cls_any = cast(Any, cls)
            info = cls_any.__tvm_ffi_type_info__
            mnemonic = core._lookup_type_attr(info.type_index, "__ffi_mnemonic__")

            assert mnemonic == f"std${cls.__name__}"
            assert cls_any.__ffi_mnemonic__ == mnemonic
            assert info.type_key == f"ffi.std.{cls.__name__}"

    def test_mnemonics_are_unique_and_well_formed(self) -> None:
        abstract = {std.Node, std.Ty, std.Stmt, std.Expr, std.Attrs, std.Structure, std.Bind}
        mnemonics = []
        for name in std.__all__:
            cls = getattr(std, name)
            if isinstance(cls, type) and issubclass(cls, std.Node) and cls not in abstract:
                cls_any = cast(Any, cls)
                info = cls_any.__tvm_ffi_type_info__
                mnemonics.append(core._lookup_type_attr(info.type_index, "__ffi_mnemonic__"))

        assert len(mnemonics) == len(set(mnemonics))
        for mnemonic in mnemonics:
            assert isinstance(mnemonic, str)
            assert mnemonic.count("$") == 1
            dialect, name = mnemonic.split("$")
            assert dialect == "std"
            assert name.isidentifier()

    def test_mnemonics_are_classvars_not_reflected_fields(self) -> None:
        abstract = {std.Node, std.Ty, std.Stmt, std.Expr, std.Attrs, std.Structure, std.Bind}
        for name in std.__all__:
            cls = getattr(std, name)
            if isinstance(cls, type) and issubclass(cls, std.Node) and cls not in abstract:
                assert "__ffi_mnemonic__" in cls.__annotations__
                assert "__ffi_mnemonic__" in cls.__dict__
                assert "__ffi_mnemonic__" not in [field.name for field in fields(cls)]

    def test_text_print_hooks_and_mnemonics_are_independent_type_attrs(self) -> None:
        for cls in [std.Node, std.Expr, std.AnyTy, std.Var, std.Add, std.Func, std.DictAttrs]:
            cls_any = cast(Any, cls)
            info = cls_any.__tvm_ffi_type_info__
            text_print = core._lookup_type_attr(info.type_index, "__ffi_text_print__")
            mnemonic = core._lookup_type_attr(info.type_index, "__ffi_mnemonic__")

            assert callable(text_print)
            if cls in [std.Node, std.Expr]:
                assert mnemonic is None
            else:
                assert mnemonic == f"std${cls.__name__}"

    def test_instance_runtime_types_resolve_to_class_mnemonics(self) -> None:
        i32 = std.PrimTy("int32")
        x = std.Var(i32, "x")
        one = std.IntImm(i32, 1)
        cases = [
            (i32, std.PrimTy),
            (x, std.Var),
            (std.Range(start=0, stop=4), std.Range),
            (std.Add(i32, x, one), std.Add),
            (std.Call(i32, "callee", [x], {"tag": "demo"}), std.Call),
            (std.ExprBind([x], None, one), std.ExprBind),
            (std.VarDef([x], {"tag": "demo"}), std.VarDef),
            (std.Store(x, [], one), std.Store),
            (std.Return([x]), std.Return),
            (std.Yield([x]), std.Yield),
            (std.Break(), std.Break),
            (std.Continue(), std.Continue),
        ]

        for node, cls in cases:
            cls_any = cast(Any, cls)
            info = cls_any.__tvm_ffi_type_info__

            assert isinstance(node, cls)
            assert (
                core._lookup_type_attr(info.type_index, "__ffi_mnemonic__") == f"std${cls.__name__}"
            )


class TestDialectPrintMap:
    def test_dialect_prefix_can_be_dropped(self) -> None:
        node = std.DictAttrs(values={"tag": "demo"})
        cfg = pyast.PrinterConfig(dialect_print_map={"std": "*"})

        assert node.text(cfg) == 'DictAttrs(tag="demo")'

    def test_dialect_prefix_can_be_renamed(self) -> None:
        node = std.DictAttrs(values={"tag": "demo"})
        cfg = pyast.PrinterConfig(dialect_print_map={"std": "core"})

        assert node.text(cfg) == 'core.DictAttrs(tag="demo")'

    def test_full_mnemonic_can_drop_prefix(self) -> None:
        node = std.DictAttrs(values={"tag": "demo"})
        cfg = pyast.PrinterConfig(dialect_print_map={"std$DictAttrs": "*"})

        assert node.text(cfg) == 'DictAttrs(tag="demo")'

    def test_full_mnemonic_can_use_explicit_name(self) -> None:
        node = std.DictAttrs(values={"tag": "demo"})
        cfg = pyast.PrinterConfig(dialect_print_map={"std$DictAttrs": "std.MyAttrs"})

        assert node.text(cfg) == 'std.MyAttrs(tag="demo")'

    def test_full_mnemonic_takes_precedence_over_dialect(self) -> None:
        node = std.DictAttrs(values={"tag": "demo"})
        cfg = pyast.PrinterConfig(dialect_print_map={"std": "core", "std$DictAttrs": "*"})

        assert node.text(cfg) == 'DictAttrs(tag="demo")'

    def test_dialect_prefix_applies_to_func_decorator(self) -> None:
        i32 = std.PrimTy("int32")
        x = std.Var(ty=i32, name="x")
        node = std.Func(
            symbol="main",
            attrs=None,
            args=[x],
            ret_type=i32,
            body=[std.Return(vars=[x])],
        )
        cfg = pyast.PrinterConfig(dialect_print_map={"std": "core"})

        assert node.text(cfg) == "@core.func\ndef main(x: i32) -> i32:\n  return x"

    def test_full_mnemonic_applies_to_func_decorator(self) -> None:
        i32 = std.PrimTy("int32")
        x = std.Var(ty=i32, name="x")
        node = std.Func(
            symbol="main",
            attrs={"tag": "demo"},
            args=[x],
            ret_type=i32,
            body=[std.Return(vars=[x])],
        )
        cfg = pyast.PrinterConfig(dialect_print_map={"std$Func": "ffi_func"})

        assert node.text(cfg) == '@ffi_func(tag="demo")\ndef main(x: i32) -> i32:\n  return x'

    def test_dialect_prefix_applies_to_call_and_cast_helpers(self) -> None:
        i32 = std.PrimTy("int32")
        x = std.Var(ty=i32, name="x")
        cfg = pyast.PrinterConfig(dialect_print_map={"std": "*"})

        assert (
            std.Call(i32, "callee", [x], {"tag": "demo"}).text(cfg) == 'call(callee, x, tag="demo")'
        )
        assert std.Cast(i32, x).text(cfg) == "i32(x)"

    def test_full_mnemonic_applies_to_sugared_cast_helper(self) -> None:
        i32 = std.PrimTy("int32")
        x = std.Var(ty=i32, name="x")
        cfg = pyast.PrinterConfig(dialect_print_map={"std$Cast": "ffi.cast"})

        assert std.Cast(i32, x).text(cfg) == "ffi.cast(x)"

    def test_full_mnemonic_applies_to_call_helper(self) -> None:
        i32 = std.PrimTy("int32")
        x = std.Var(ty=i32, name="x")
        cfg = pyast.PrinterConfig(dialect_print_map={"std$Call": "ffi.call"})

        assert std.Call(i32, "callee", [x], {"tag": "demo"}).text(cfg) == (
            'ffi.call(callee, x, tag="demo")'
        )


class TestStructure:
    def test_range_is_structure(self) -> None:
        node = std.Range(start=1)

        assert isinstance(node, std.Structure)
        assert issubclass(std.Range, std.Structure)
        assert tuple(field.name for field in fields(std.Structure)) == ()


class TestTupleType:
    def test_constructor(self) -> None:
        i32 = std.PrimTy("int32")
        f32 = std.PrimTy("float32")
        node = std.TupleType(fields=[i32, f32])

        assert isinstance(node, std.TupleType)
        assert tuple(field.name for field in fields(std.TupleType)) == ("fields",)
        assert list(node.fields) == [i32, f32]

    def test_text_format(self) -> None:
        node = std.TupleType(fields=[std.PrimTy("int32"), std.PrimTy("float32")])

        assert node.text() == "tuple[i32, f32]"

    def test_structural_equality(self) -> None:
        lhs = std.TupleType(fields=[std.PrimTy("int32"), std.PrimTy("float32")])
        rhs = std.TupleType(fields=[std.PrimTy("int32"), std.PrimTy("float32")])
        different = std.TupleType(fields=[std.PrimTy("int32")])

        assert tvm_ffi.structural_equal(lhs, rhs)
        assert tvm_ffi.structural_hash(lhs) == tvm_ffi.structural_hash(rhs)
        assert not tvm_ffi.structural_equal(lhs, different)


class TestTensorTy:
    def test_constructor(self) -> None:
        node = std.TensorTy(shape=[1, 2], dtype="float32")

        assert isinstance(node, std.TensorTy)
        assert tuple(field.name for field in fields(std.TensorTy)) == ("shape", "dtype")
        shape = list(node.shape)
        assert isinstance(shape[0], std.IntImm)
        assert isinstance(shape[1], std.IntImm)
        assert shape[0].value == 1
        assert shape[1].value == 2
        assert node.dtype == tvm_ffi.float32

    def test_text_format(self) -> None:
        node = std.TensorTy(
            shape=[1, 2],
            dtype="float32",
        )

        assert node.text() == "f32[1, 2]"
        assert (
            std.TensorTy(
                shape=[14, 21],
                dtype="int32",
            ).text()
            == "i32[14, 21]"
        )

    def test_structural_equality(self) -> None:
        lhs = std.TensorTy(
            shape=[1, 2],
            dtype="float32",
        )
        rhs = std.TensorTy(
            shape=[1, 2],
            dtype="float32",
        )
        different = std.TensorTy(shape=[2], dtype="float32")

        assert tvm_ffi.structural_equal(lhs, rhs)
        assert tvm_ffi.structural_hash(lhs) == tvm_ffi.structural_hash(rhs)
        assert not tvm_ffi.structural_equal(lhs, different)


class TestExpr:
    def test_from_any_converts_python_int_to_int_imm(self) -> None:
        node = core._to_py_class_value(core.TypeSchema.from_annotation(std.Expr).convert(7))

        assert isinstance(node, std.IntImm)
        assert isinstance(node.ty, std.AnyTy)
        assert node.value == 7
        assert node.text() == "7"

    def test_from_any_converts_python_float_to_float_imm(self) -> None:
        node = core._to_py_class_value(core.TypeSchema.from_annotation(std.Expr).convert(1.5))

        assert isinstance(node, std.FloatImm)
        assert isinstance(node.ty, std.AnyTy)
        assert node.value == 1.5
        assert node.text() == "1.5"

    def test_from_any_converts_python_string_to_string_imm(self) -> None:
        node = core._to_py_class_value(core.TypeSchema.from_annotation(std.Expr).convert("hello"))

        assert isinstance(node, std.StringImm)
        assert isinstance(node.ty, std.AnyTy)
        assert node.value == "hello"
        assert node.text() == '"hello"'


class TestVar:
    def test_constructor(self) -> None:
        i32 = std.PrimTy("int32")
        node = std.Var(ty=i32, name="x")

        assert isinstance(node, std.Var)
        assert tuple(field.name for field in fields(std.Var)) == ("ty", "name")
        assert node.ty == i32
        assert node.name == "x"

    def test_text_format(self) -> None:
        node = std.Var(ty=std.PrimTy("int32"), name="x")

        assert node.text() == "x"

    def test_structural_equality(self) -> None:
        lhs = std.Var(ty=std.PrimTy("int32"), name="x")
        rhs = std.Var(ty=std.PrimTy("int32"), name="renamed")
        different = std.Var(ty=std.PrimTy("float32"), name="x")

        assert tvm_ffi.structural_equal(lhs, rhs)
        assert tvm_ffi.structural_hash(lhs) == tvm_ffi.structural_hash(rhs)
        assert not tvm_ffi.structural_equal(lhs, different)


class TestFunc:
    def test_constructor(self) -> None:
        i32 = std.PrimTy("int32")
        x = std.Var(ty=i32, name="x")
        node = std.Func(
            symbol="main",
            attrs={"tag": "demo"},
            args=[x],
            ret_type=i32,
            body=[std.Return(vars=[x])],
        )

        assert isinstance(node, std.Func)
        assert tuple(field.name for field in fields(std.Func)) == (
            "symbol",
            "attrs",
            "args",
            "ret_type",
            "body",
        )
        assert node.symbol == "main"
        assert isinstance(node.attrs, std.DictAttrs)
        assert dict(node.attrs.values) == {"tag": "demo"}
        assert list(node.args) == [x]
        assert node.ret_type == i32
        assert len(node.body) == 1

    def test_text_format(self) -> None:
        i32 = std.PrimTy("int32")
        x = std.Var(ty=i32, name="x")
        node = std.Func(
            symbol="main",
            attrs={"tag": "demo"},
            args=[x],
            ret_type=i32,
            body=[std.Return(vars=[x])],
        )

        assert node.text() == ('@std.func(tag="demo")\ndef main(x: i32) -> i32:\n  return x')

    def test_text_format_without_return_type(self) -> None:
        i32 = std.PrimTy("int32")
        x = std.Var(ty=i32, name="x")
        node = std.Func(
            symbol="main",
            attrs=None,
            args=[x],
            ret_type=None,
            body=[std.Return(vars=[x])],
        )

        assert node.text() == "@std.func\ndef main(x: i32):\n  return x"

    def test_structural_equality(self) -> None:
        i32 = std.PrimTy("int32")
        lhs_x = std.Var(ty=i32, name="x")
        rhs_x = std.Var(ty=i32, name="x")
        other_x = std.Var(ty=i32, name="x")
        lhs = std.Func(
            symbol="main",
            attrs={"tag": "demo"},
            args=[lhs_x],
            ret_type=i32,
            body=[std.Return(vars=[lhs_x])],
        )
        rhs = std.Func(
            symbol="main",
            attrs={"tag": "demo"},
            args=[rhs_x],
            ret_type=i32,
            body=[std.Return(vars=[rhs_x])],
        )
        different = std.Func(
            symbol="other",
            attrs={"tag": "demo"},
            args=[other_x],
            ret_type=i32,
            body=[std.Return(vars=[other_x])],
        )

        assert tvm_ffi.structural_equal(lhs, rhs)
        assert tvm_ffi.structural_hash(lhs) == tvm_ffi.structural_hash(rhs)
        assert not tvm_ffi.structural_equal(lhs, different)

    def test_attrs_and_ret_type_are_required(self) -> None:
        i32 = std.PrimTy("int32")
        x = std.Var(ty=i32, name="x")
        func_ctor = cast(Any, std.Func)

        with pytest.raises(TypeError):
            func_ctor(symbol="main", args=[x], ret_type=i32, body=[])

        with pytest.raises(TypeError):
            func_ctor(symbol="main", attrs=None, args=[x], body=[])

        func = std.Func(symbol="main", attrs=None, args=[x], ret_type=None, body=[])
        assert func.attrs is None
        assert func.ret_type is None

    def test_attrs_field_accepts_dict_attrs_and_none(self) -> None:
        i32 = std.PrimTy("int32")
        x = std.Var(ty=i32, name="x")
        attrs = {"tag": "demo"}
        with_attrs = std.Func(
            symbol="main",
            attrs=attrs,
            args=[x],
            ret_type=i32,
            body=[std.Return(vars=[x])],
        )
        with_plain_dict_attrs = std.Func(
            symbol="main",
            attrs={"tag": "demo"},
            args=[x],
            ret_type=i32,
            body=[std.Return(vars=[x])],
        )
        with_empty_attrs = std.Func(
            symbol="main",
            attrs={},
            args=[x],
            ret_type=i32,
            body=[std.Return(vars=[x])],
        )
        without_attrs = std.Func(
            symbol="main",
            attrs=None,
            args=[x],
            ret_type=i32,
            body=[std.Return(vars=[x])],
        )

        assert isinstance(with_attrs.attrs, std.DictAttrs)
        assert dict(with_attrs.attrs.values) == attrs
        assert isinstance(with_plain_dict_attrs.attrs, std.DictAttrs)
        assert dict(with_plain_dict_attrs.attrs.values) == {"tag": "demo"}
        assert isinstance(with_empty_attrs.attrs, std.DictAttrs)
        assert without_attrs.attrs is None
        assert with_attrs.text() == ('@std.func(tag="demo")\ndef main(x: i32) -> i32:\n  return x')
        assert (
            with_plain_dict_attrs.text()
            == '@std.func(tag="demo")\ndef main(x: i32) -> i32:\n  return x'
        )
        assert with_empty_attrs.text() == "@std.func\ndef main(x: i32) -> i32:\n  return x"
        assert without_attrs.text() == "@std.func\ndef main(x: i32) -> i32:\n  return x"


class TestModule:
    def test_constructor(self) -> None:
        i32 = std.PrimTy("int32")
        x = std.Var(ty=i32, name="x")
        func = std.Func(
            symbol="main",
            attrs={"tag": "demo"},
            args=[x],
            ret_type=i32,
            body=[std.Return(vars=[x])],
        )
        node = std.Module(funcs=[func])

        assert isinstance(node, std.Module)
        assert tuple(field.name for field in fields(std.Module)) == ("funcs",)
        assert list(node.funcs) == [func]

    def test_text_format(self) -> None:
        i32 = std.PrimTy("int32")
        x = std.Var(ty=i32, name="x")
        node = std.Module(
            funcs=[
                std.Func(
                    symbol="main",
                    attrs={"tag": "demo"},
                    args=[x],
                    ret_type=i32,
                    body=[std.Return(vars=[x])],
                )
            ]
        )

        assert node.text() == ('@std.func(tag="demo")\ndef main(x: i32) -> i32:\n  return x')

    def test_structural_equality(self) -> None:
        i32 = std.PrimTy("int32")
        lhs_x = std.Var(ty=i32, name="x")
        rhs_x = std.Var(ty=i32, name="x")
        other_x = std.Var(ty=i32, name="x")
        lhs = std.Module(
            funcs=[
                std.Func(
                    symbol="main",
                    attrs={"tag": "demo"},
                    args=[lhs_x],
                    ret_type=i32,
                    body=[std.Return(vars=[lhs_x])],
                )
            ]
        )
        rhs = std.Module(
            funcs=[
                std.Func(
                    symbol="main",
                    attrs={"tag": "demo"},
                    args=[rhs_x],
                    ret_type=i32,
                    body=[std.Return(vars=[rhs_x])],
                )
            ]
        )
        different = std.Module(
            funcs=[
                std.Func(
                    symbol="other",
                    attrs={"tag": "demo"},
                    args=[other_x],
                    ret_type=i32,
                    body=[std.Return(vars=[other_x])],
                )
            ]
        )

        assert tvm_ffi.structural_equal(lhs, rhs)
        assert tvm_ffi.structural_hash(lhs) == tvm_ffi.structural_hash(rhs)
        assert not tvm_ffi.structural_equal(lhs, different)


class TestRange:
    def test_constructor(self) -> None:
        node = std.Range(start=1, stop=2, step=3)

        assert isinstance(node, std.Range)
        assert tuple(field.name for field in fields(std.Range)) == ("start", "stop", "step")
        assert isinstance(node.start, std.IntImm)
        assert isinstance(node.stop, std.IntImm)
        assert isinstance(node.step, std.IntImm)
        assert node.start.value == 1
        assert node.stop.value == 2
        assert node.step.value == 3

    def test_constructor_without_start(self) -> None:
        node = std.Range(start=None, stop=2, step=3)

        assert isinstance(node, std.Range)
        assert node.start is None
        assert isinstance(node.stop, std.IntImm)
        assert isinstance(node.step, std.IntImm)
        assert node.stop.value == 2
        assert node.step.value == 3

    def test_from_any_converts_python_int_to_single_point_range(self) -> None:
        node = core._to_py_class_value(core.TypeSchema.from_annotation(std.Range).convert(7))

        assert isinstance(node, std.Range)
        assert isinstance(node.start, std.IntImm)
        assert node.start.value == 7
        assert node.stop is None
        assert node.step is None
        assert node.text() == "7"

    def test_text_format(self) -> None:
        node = std.Range(start=1, stop=2, step=3)

        assert node.text() == "1:2:3"

    def test_text_format_without_stop_or_step(self) -> None:
        node = std.Range(start=1)

        assert node.text() == "1"

    def test_text_format_without_step(self) -> None:
        node = std.Range(start=1, stop=2)

        assert node.text() == "1:2"

    def test_text_format_without_stop(self) -> None:
        node = std.Range(start=1, step=3)

        assert node.text() == "1::3"

    def test_text_format_without_start(self) -> None:
        node = std.Range(start=None, stop=2, step=3)

        assert node.text() == ":2:3"

    def test_text_format_without_start_or_step(self) -> None:
        node = std.Range(start=None, stop=2)

        assert node.text() == ":2"

    def test_text_format_without_start_or_stop(self) -> None:
        node = std.Range(start=None, step=3)

        assert node.text() == "::3"

    def test_text_format_without_any_part(self) -> None:
        node = std.Range()

        assert node.text() == ":"

    def test_structural_equality(self) -> None:
        lhs = std.Range(start=1, stop=2, step=3)
        rhs = std.Range(start=1, stop=2, step=3)
        different = std.Range(start=1, stop=2)

        assert tvm_ffi.structural_equal(lhs, rhs)
        assert tvm_ffi.structural_hash(lhs) == tvm_ffi.structural_hash(rhs)
        assert not tvm_ffi.structural_equal(lhs, different)

    def test_structural_equality_without_start(self) -> None:
        lhs = std.Range(start=None, stop=2, step=3)
        rhs = std.Range(start=None, stop=2, step=3)
        different = std.Range(start=1, stop=2, step=3)

        assert tvm_ffi.structural_equal(lhs, rhs)
        assert tvm_ffi.structural_hash(lhs) == tvm_ffi.structural_hash(rhs)
        assert not tvm_ffi.structural_equal(lhs, different)

    def test_default_arguments(self) -> None:
        node = std.Range()

        assert node.start is None
        assert node.stop is None
        assert node.step is None


class TestIntImm:
    def test_constructor(self) -> None:
        i32 = std.PrimTy("int32")
        node = std.IntImm(ty=i32, value=1)

        assert isinstance(node, std.IntImm)
        assert tuple(field.name for field in fields(std.IntImm)) == ("ty", "value")
        assert node.ty == i32
        assert node.value == 1

    def test_text_format(self) -> None:
        node = std.IntImm(ty=std.PrimTy("int32"), value=1)

        assert node.text() == "1"

    def test_structural_equality(self) -> None:
        i32 = std.PrimTy("int32")
        lhs = std.IntImm(ty=i32, value=1)
        rhs = std.IntImm(ty=i32, value=1)
        different = std.IntImm(ty=i32, value=2)

        assert tvm_ffi.structural_equal(lhs, rhs)
        assert tvm_ffi.structural_hash(lhs) == tvm_ffi.structural_hash(rhs)
        assert not tvm_ffi.structural_equal(lhs, different)


class TestFloatImm:
    def test_constructor(self) -> None:
        f32 = std.PrimTy("float32")
        node = std.FloatImm(ty=f32, value=1.5)

        assert isinstance(node, std.FloatImm)
        assert tuple(field.name for field in fields(std.FloatImm)) == ("ty", "value")
        assert node.ty == f32
        assert node.value == 1.5

    def test_text_format(self) -> None:
        node = std.FloatImm(ty=std.PrimTy("float32"), value=1.5)

        assert node.text() == "1.5"

    def test_structural_equality(self) -> None:
        f32 = std.PrimTy("float32")
        lhs = std.FloatImm(ty=f32, value=1.5)
        rhs = std.FloatImm(ty=f32, value=1.5)
        different = std.FloatImm(ty=f32, value=2.5)

        assert tvm_ffi.structural_equal(lhs, rhs)
        assert tvm_ffi.structural_hash(lhs) == tvm_ffi.structural_hash(rhs)
        assert not tvm_ffi.structural_equal(lhs, different)


class TestStringImm:
    def test_constructor(self) -> None:
        any_ty = std.AnyTy()
        node = std.StringImm(ty=any_ty, value="hello")

        assert isinstance(node, std.StringImm)
        assert tuple(field.name for field in fields(std.StringImm)) == ("ty", "value")
        assert node.ty == any_ty
        assert node.value == "hello"

    def test_text_format(self) -> None:
        node = std.StringImm(ty=std.AnyTy(), value="hello")

        assert node.text() == '"hello"'

    def test_structural_equality(self) -> None:
        any_ty = std.AnyTy()
        lhs = std.StringImm(ty=any_ty, value="hello")
        rhs = std.StringImm(ty=std.AnyTy(), value="hello")
        different = std.StringImm(ty=std.AnyTy(), value="world")

        assert tvm_ffi.structural_equal(lhs, rhs)
        assert tvm_ffi.structural_hash(lhs) == tvm_ffi.structural_hash(rhs)
        assert not tvm_ffi.structural_equal(lhs, different)


class TestAdd:
    def test_constructor(self) -> None:
        i32 = std.PrimTy("int32")
        one = 1
        two = 2
        node = std.Add(ty=i32, a=one, b=two)

        assert isinstance(node, std.Add)
        assert tuple(field.name for field in fields(std.Add)) == ("ty", "a", "b")
        assert node.ty == i32
        assert isinstance(node.a, std.IntImm)
        assert isinstance(node.b, std.IntImm)
        assert node.a.value == one
        assert node.b.value == two

    def test_text_format(self) -> None:
        i32 = std.PrimTy("int32")
        node = std.Add(ty=i32, a=1, b=2)

        assert node.text() == "1 + 2"

    def test_structural_equality(self) -> None:
        i32 = std.PrimTy("int32")
        lhs = std.Add(ty=i32, a=1, b=2)
        rhs = std.Add(ty=i32, a=1, b=2)
        different = std.Add(
            ty=i32,
            a=1,
            b=3,
        )

        assert tvm_ffi.structural_equal(lhs, rhs)
        assert tvm_ffi.structural_hash(lhs) == tvm_ffi.structural_hash(rhs)
        assert not tvm_ffi.structural_equal(lhs, different)


class TestSub:
    def test_constructor(self) -> None:
        i32 = std.PrimTy("int32")
        one = 1
        two = 2
        node = std.Sub(ty=i32, a=one, b=two)

        assert isinstance(node, std.Sub)
        assert tuple(field.name for field in fields(std.Sub)) == ("ty", "a", "b")
        assert node.ty == i32
        assert isinstance(node.a, std.IntImm)
        assert isinstance(node.b, std.IntImm)
        assert node.a.value == one
        assert node.b.value == two

    def test_text_format(self) -> None:
        i32 = std.PrimTy("int32")
        node = std.Sub(ty=i32, a=1, b=2)

        assert node.text() == "1 - 2"

    def test_structural_equality(self) -> None:
        i32 = std.PrimTy("int32")
        lhs = std.Sub(ty=i32, a=1, b=2)
        rhs = std.Sub(ty=i32, a=1, b=2)
        different = std.Sub(
            ty=i32,
            a=1,
            b=3,
        )

        assert tvm_ffi.structural_equal(lhs, rhs)
        assert tvm_ffi.structural_hash(lhs) == tvm_ffi.structural_hash(rhs)
        assert not tvm_ffi.structural_equal(lhs, different)


class TestMul:
    def test_constructor(self) -> None:
        i32 = std.PrimTy("int32")
        one = 1
        two = 2
        node = std.Mul(ty=i32, a=one, b=two)

        assert isinstance(node, std.Mul)
        assert tuple(field.name for field in fields(std.Mul)) == ("ty", "a", "b")
        assert node.ty == i32
        assert isinstance(node.a, std.IntImm)
        assert isinstance(node.b, std.IntImm)
        assert node.a.value == one
        assert node.b.value == two

    def test_text_format(self) -> None:
        i32 = std.PrimTy("int32")
        node = std.Mul(ty=i32, a=1, b=2)

        assert node.text() == "1 * 2"

    def test_structural_equality(self) -> None:
        i32 = std.PrimTy("int32")
        lhs = std.Mul(ty=i32, a=1, b=2)
        rhs = std.Mul(ty=i32, a=1, b=2)
        different = std.Mul(
            ty=i32,
            a=1,
            b=3,
        )

        assert tvm_ffi.structural_equal(lhs, rhs)
        assert tvm_ffi.structural_hash(lhs) == tvm_ffi.structural_hash(rhs)
        assert not tvm_ffi.structural_equal(lhs, different)


class TestFloorDiv:
    def test_constructor(self) -> None:
        i32 = std.PrimTy("int32")
        one = 1
        two = 2
        node = std.FloorDiv(ty=i32, a=one, b=two)

        assert isinstance(node, std.FloorDiv)
        assert tuple(field.name for field in fields(std.FloorDiv)) == ("ty", "a", "b")
        assert node.ty == i32
        assert isinstance(node.a, std.IntImm)
        assert isinstance(node.b, std.IntImm)
        assert node.a.value == one
        assert node.b.value == two

    def test_text_format(self) -> None:
        i32 = std.PrimTy("int32")
        node = std.FloorDiv(
            ty=i32,
            a=1,
            b=2,
        )

        assert node.text() == "1 // 2"

    def test_structural_equality(self) -> None:
        i32 = std.PrimTy("int32")
        lhs = std.FloorDiv(
            ty=i32,
            a=1,
            b=2,
        )
        rhs = std.FloorDiv(
            ty=i32,
            a=1,
            b=2,
        )
        different = std.FloorDiv(
            ty=i32,
            a=1,
            b=3,
        )

        assert tvm_ffi.structural_equal(lhs, rhs)
        assert tvm_ffi.structural_hash(lhs) == tvm_ffi.structural_hash(rhs)
        assert not tvm_ffi.structural_equal(lhs, different)


class TestFloorMod:
    def test_constructor(self) -> None:
        i32 = std.PrimTy("int32")
        one = 1
        two = 2
        node = std.FloorMod(ty=i32, a=one, b=two)

        assert isinstance(node, std.FloorMod)
        assert tuple(field.name for field in fields(std.FloorMod)) == ("ty", "a", "b")
        assert node.ty == i32
        assert isinstance(node.a, std.IntImm)
        assert isinstance(node.b, std.IntImm)
        assert node.a.value == one
        assert node.b.value == two

    def test_text_format(self) -> None:
        i32 = std.PrimTy("int32")
        node = std.FloorMod(
            ty=i32,
            a=1,
            b=2,
        )

        assert node.text() == "1 % 2"

    def test_structural_equality(self) -> None:
        i32 = std.PrimTy("int32")
        lhs = std.FloorMod(
            ty=i32,
            a=1,
            b=2,
        )
        rhs = std.FloorMod(
            ty=i32,
            a=1,
            b=2,
        )
        different = std.FloorMod(
            ty=i32,
            a=1,
            b=3,
        )

        assert tvm_ffi.structural_equal(lhs, rhs)
        assert tvm_ffi.structural_hash(lhs) == tvm_ffi.structural_hash(rhs)
        assert not tvm_ffi.structural_equal(lhs, different)


class TestMin:
    def test_constructor(self) -> None:
        i32 = std.PrimTy("int32")
        one = 1
        two = 2
        node = std.Min(ty=i32, a=one, b=two)

        assert isinstance(node, std.Min)
        assert tuple(field.name for field in fields(std.Min)) == ("ty", "a", "b")
        assert node.ty == i32
        assert isinstance(node.a, std.IntImm)
        assert isinstance(node.b, std.IntImm)
        assert node.a.value == one
        assert node.b.value == two

    def test_text_format(self) -> None:
        i32 = std.PrimTy("int32")
        node = std.Min(ty=i32, a=1, b=2)

        assert node.text() == "min(1, 2)"

    def test_structural_equality(self) -> None:
        i32 = std.PrimTy("int32")
        lhs = std.Min(ty=i32, a=1, b=2)
        rhs = std.Min(ty=i32, a=1, b=2)
        different = std.Min(
            ty=i32,
            a=1,
            b=3,
        )

        assert tvm_ffi.structural_equal(lhs, rhs)
        assert tvm_ffi.structural_hash(lhs) == tvm_ffi.structural_hash(rhs)
        assert not tvm_ffi.structural_equal(lhs, different)


class TestMax:
    def test_constructor(self) -> None:
        i32 = std.PrimTy("int32")
        one = 1
        two = 2
        node = std.Max(ty=i32, a=one, b=two)

        assert isinstance(node, std.Max)
        assert tuple(field.name for field in fields(std.Max)) == ("ty", "a", "b")
        assert node.ty == i32
        assert isinstance(node.a, std.IntImm)
        assert isinstance(node.b, std.IntImm)
        assert node.a.value == one
        assert node.b.value == two

    def test_text_format(self) -> None:
        i32 = std.PrimTy("int32")
        node = std.Max(ty=i32, a=1, b=2)

        assert node.text() == "max(1, 2)"

    def test_structural_equality(self) -> None:
        i32 = std.PrimTy("int32")
        lhs = std.Max(ty=i32, a=1, b=2)
        rhs = std.Max(ty=i32, a=1, b=2)
        different = std.Max(
            ty=i32,
            a=1,
            b=3,
        )

        assert tvm_ffi.structural_equal(lhs, rhs)
        assert tvm_ffi.structural_hash(lhs) == tvm_ffi.structural_hash(rhs)
        assert not tvm_ffi.structural_equal(lhs, different)


class TestEq:
    def test_constructor(self) -> None:
        i32 = std.PrimTy("int32")
        one = 1
        two = 2
        node = std.Eq(ty=i32, a=one, b=two)

        assert isinstance(node, std.Eq)
        assert tuple(field.name for field in fields(std.Eq)) == ("ty", "a", "b")
        assert node.ty == i32
        assert isinstance(node.a, std.IntImm)
        assert isinstance(node.b, std.IntImm)
        assert node.a.value == one
        assert node.b.value == two

    def test_text_format(self) -> None:
        i32 = std.PrimTy("int32")
        node = std.Eq(ty=i32, a=1, b=2)

        assert node.text() == "1 == 2"

    def test_structural_equality(self) -> None:
        i32 = std.PrimTy("int32")
        lhs = std.Eq(ty=i32, a=1, b=2)
        rhs = std.Eq(ty=i32, a=1, b=2)
        different = std.Eq(
            ty=i32,
            a=1,
            b=3,
        )

        assert tvm_ffi.structural_equal(lhs, rhs)
        assert tvm_ffi.structural_hash(lhs) == tvm_ffi.structural_hash(rhs)
        assert not tvm_ffi.structural_equal(lhs, different)


class TestNe:
    def test_constructor(self) -> None:
        i32 = std.PrimTy("int32")
        one = 1
        two = 2
        node = std.Ne(ty=i32, a=one, b=two)

        assert isinstance(node, std.Ne)
        assert tuple(field.name for field in fields(std.Ne)) == ("ty", "a", "b")
        assert node.ty == i32
        assert isinstance(node.a, std.IntImm)
        assert isinstance(node.b, std.IntImm)
        assert node.a.value == one
        assert node.b.value == two

    def test_text_format(self) -> None:
        i32 = std.PrimTy("int32")
        node = std.Ne(ty=i32, a=1, b=2)

        assert node.text() == "1 != 2"

    def test_structural_equality(self) -> None:
        i32 = std.PrimTy("int32")
        lhs = std.Ne(ty=i32, a=1, b=2)
        rhs = std.Ne(ty=i32, a=1, b=2)
        different = std.Ne(
            ty=i32,
            a=1,
            b=3,
        )

        assert tvm_ffi.structural_equal(lhs, rhs)
        assert tvm_ffi.structural_hash(lhs) == tvm_ffi.structural_hash(rhs)
        assert not tvm_ffi.structural_equal(lhs, different)


class TestLe:
    def test_constructor(self) -> None:
        i32 = std.PrimTy("int32")
        one = 1
        two = 2
        node = std.Le(ty=i32, a=one, b=two)

        assert isinstance(node, std.Le)
        assert tuple(field.name for field in fields(std.Le)) == ("ty", "a", "b")
        assert node.ty == i32
        assert isinstance(node.a, std.IntImm)
        assert isinstance(node.b, std.IntImm)
        assert node.a.value == one
        assert node.b.value == two

    def test_text_format(self) -> None:
        i32 = std.PrimTy("int32")
        node = std.Le(ty=i32, a=1, b=2)

        assert node.text() == "1 <= 2"

    def test_structural_equality(self) -> None:
        i32 = std.PrimTy("int32")
        lhs = std.Le(ty=i32, a=1, b=2)
        rhs = std.Le(ty=i32, a=1, b=2)
        different = std.Le(
            ty=i32,
            a=1,
            b=3,
        )

        assert tvm_ffi.structural_equal(lhs, rhs)
        assert tvm_ffi.structural_hash(lhs) == tvm_ffi.structural_hash(rhs)
        assert not tvm_ffi.structural_equal(lhs, different)


class TestGe:
    def test_constructor(self) -> None:
        i32 = std.PrimTy("int32")
        one = 1
        two = 2
        node = std.Ge(ty=i32, a=one, b=two)

        assert isinstance(node, std.Ge)
        assert tuple(field.name for field in fields(std.Ge)) == ("ty", "a", "b")
        assert node.ty == i32
        assert isinstance(node.a, std.IntImm)
        assert isinstance(node.b, std.IntImm)
        assert node.a.value == one
        assert node.b.value == two

    def test_text_format(self) -> None:
        i32 = std.PrimTy("int32")
        node = std.Ge(ty=i32, a=1, b=2)

        assert node.text() == "1 >= 2"

    def test_structural_equality(self) -> None:
        i32 = std.PrimTy("int32")
        lhs = std.Ge(ty=i32, a=1, b=2)
        rhs = std.Ge(ty=i32, a=1, b=2)
        different = std.Ge(
            ty=i32,
            a=1,
            b=3,
        )

        assert tvm_ffi.structural_equal(lhs, rhs)
        assert tvm_ffi.structural_hash(lhs) == tvm_ffi.structural_hash(rhs)
        assert not tvm_ffi.structural_equal(lhs, different)


class TestGt:
    def test_constructor(self) -> None:
        i32 = std.PrimTy("int32")
        one = 1
        two = 2
        node = std.Gt(ty=i32, a=one, b=two)

        assert isinstance(node, std.Gt)
        assert tuple(field.name for field in fields(std.Gt)) == ("ty", "a", "b")
        assert node.ty == i32
        assert isinstance(node.a, std.IntImm)
        assert isinstance(node.b, std.IntImm)
        assert node.a.value == one
        assert node.b.value == two

    def test_text_format(self) -> None:
        i32 = std.PrimTy("int32")
        node = std.Gt(ty=i32, a=1, b=2)

        assert node.text() == "1 > 2"

    def test_structural_equality(self) -> None:
        i32 = std.PrimTy("int32")
        lhs = std.Gt(ty=i32, a=1, b=2)
        rhs = std.Gt(ty=i32, a=1, b=2)
        different = std.Gt(
            ty=i32,
            a=1,
            b=3,
        )

        assert tvm_ffi.structural_equal(lhs, rhs)
        assert tvm_ffi.structural_hash(lhs) == tvm_ffi.structural_hash(rhs)
        assert not tvm_ffi.structural_equal(lhs, different)


class TestLt:
    def test_constructor(self) -> None:
        i32 = std.PrimTy("int32")
        one = 1
        two = 2
        node = std.Lt(ty=i32, a=one, b=two)

        assert isinstance(node, std.Lt)
        assert tuple(field.name for field in fields(std.Lt)) == ("ty", "a", "b")
        assert node.ty == i32
        assert isinstance(node.a, std.IntImm)
        assert isinstance(node.b, std.IntImm)
        assert node.a.value == one
        assert node.b.value == two

    def test_text_format(self) -> None:
        i32 = std.PrimTy("int32")
        node = std.Lt(ty=i32, a=1, b=2)

        assert node.text() == "1 < 2"

    def test_structural_equality(self) -> None:
        i32 = std.PrimTy("int32")
        lhs = std.Lt(ty=i32, a=1, b=2)
        rhs = std.Lt(ty=i32, a=1, b=2)
        different = std.Lt(
            ty=i32,
            a=1,
            b=3,
        )

        assert tvm_ffi.structural_equal(lhs, rhs)
        assert tvm_ffi.structural_hash(lhs) == tvm_ffi.structural_hash(rhs)
        assert not tvm_ffi.structural_equal(lhs, different)


class TestAnd:
    def test_constructor(self) -> None:
        i32 = std.PrimTy("int32")
        one = 1
        two = 2
        node = std.And(ty=i32, a=one, b=two)

        assert isinstance(node, std.And)
        assert tuple(field.name for field in fields(std.And)) == ("ty", "a", "b")
        assert node.ty == i32
        assert isinstance(node.a, std.IntImm)
        assert isinstance(node.b, std.IntImm)
        assert node.a.value == one
        assert node.b.value == two

    def test_text_format(self) -> None:
        i32 = std.PrimTy("int32")
        node = std.And(ty=i32, a=1, b=2)

        assert node.text() == "1 and 2"

    def test_structural_equality(self) -> None:
        i32 = std.PrimTy("int32")
        lhs = std.And(ty=i32, a=1, b=2)
        rhs = std.And(ty=i32, a=1, b=2)
        different = std.And(
            ty=i32,
            a=1,
            b=3,
        )

        assert tvm_ffi.structural_equal(lhs, rhs)
        assert tvm_ffi.structural_hash(lhs) == tvm_ffi.structural_hash(rhs)
        assert not tvm_ffi.structural_equal(lhs, different)


class TestOr:
    def test_constructor(self) -> None:
        i32 = std.PrimTy("int32")
        one = 1
        two = 2
        node = std.Or(ty=i32, a=one, b=two)

        assert isinstance(node, std.Or)
        assert tuple(field.name for field in fields(std.Or)) == ("ty", "a", "b")
        assert node.ty == i32
        assert isinstance(node.a, std.IntImm)
        assert isinstance(node.b, std.IntImm)
        assert node.a.value == one
        assert node.b.value == two

    def test_text_format(self) -> None:
        i32 = std.PrimTy("int32")
        node = std.Or(ty=i32, a=1, b=2)

        assert node.text() == "1 or 2"

    def test_structural_equality(self) -> None:
        i32 = std.PrimTy("int32")
        lhs = std.Or(ty=i32, a=1, b=2)
        rhs = std.Or(ty=i32, a=1, b=2)
        different = std.Or(
            ty=i32,
            a=1,
            b=3,
        )

        assert tvm_ffi.structural_equal(lhs, rhs)
        assert tvm_ffi.structural_hash(lhs) == tvm_ffi.structural_hash(rhs)
        assert not tvm_ffi.structural_equal(lhs, different)


class TestNot:
    def test_constructor(self) -> None:
        i32 = std.PrimTy("int32")
        operand = 1
        node = std.Not(ty=i32, operand=operand)

        assert isinstance(node, std.Not)
        assert tuple(field.name for field in fields(std.Not)) == ("ty", "operand")
        assert node.ty == i32
        assert isinstance(node.operand, std.IntImm)
        assert node.operand.value == operand

    def test_text_format(self) -> None:
        i32 = std.PrimTy("int32")
        node = std.Not(ty=i32, operand=1)

        assert node.text() == "not 1"

    def test_from_any_converts_python_int_to_not(self) -> None:
        node = core._to_py_class_value(core.TypeSchema.from_annotation(std.Not).convert(1))

        assert isinstance(node, std.Not)
        assert isinstance(node.ty, std.AnyTy)
        assert isinstance(node.operand, std.IntImm)
        assert node.operand.value == 1
        assert node.text() == "not 1"

    def test_structural_equality(self) -> None:
        i32 = std.PrimTy("int32")
        lhs = std.Not(ty=i32, operand=1)
        rhs = std.Not(ty=i32, operand=1)
        different = std.Not(ty=i32, operand=2)

        assert tvm_ffi.structural_equal(lhs, rhs)
        assert tvm_ffi.structural_hash(lhs) == tvm_ffi.structural_hash(rhs)
        assert not tvm_ffi.structural_equal(lhs, different)


class TestLoad:
    def test_constructor(self) -> None:
        i32 = std.PrimTy("int32")
        x = std.Var(ty=i32, name="x")
        first = std.Range(start=1, stop=2)
        second = std.Range(
            start=1,
            stop=2,
            step=3,
        )
        node = std.Load(ty=i32, var=x, indices=[first, second])

        assert isinstance(node, std.Load)
        assert tuple(field.name for field in fields(std.Load)) == ("ty", "var", "indices")
        assert node.ty == i32
        assert node.var == x
        assert list(node.indices) == [first, second]

    def test_text_format(self) -> None:
        i32 = std.PrimTy("int32")
        node = std.Load(
            ty=i32,
            var=std.Var(ty=i32, name="x"),
            indices=[
                std.Range(
                    start=1,
                    stop=2,
                ),
                std.Range(
                    start=1,
                    stop=2,
                    step=3,
                ),
            ],
        )

        assert node.text() == "x[1:2, 1:2:3]"

    def test_text_format_index_variants(self) -> None:
        i32 = std.PrimTy("int32")

        no_indices = std.Load(ty=i32, var=std.Var(ty=i32, name="x"), indices=[])
        assert no_indices.text() == "x"

        point = std.Load(
            ty=i32,
            var=std.Var(ty=i32, name="x"),
            indices=[1],
        )
        assert point.text() == "x[1]"

        slice_without_step = std.Load(
            ty=i32,
            var=std.Var(ty=i32, name="x"),
            indices=[
                std.Range(
                    start=1,
                    stop=2,
                )
            ],
        )
        assert slice_without_step.text() == "x[1:2]"

        slice_without_start = std.Load(
            ty=i32,
            var=std.Var(ty=i32, name="x"),
            indices=[
                std.Range(
                    start=None,
                    stop=2,
                )
            ],
        )
        assert slice_without_start.text() == "x[:2]"

        slice_without_stop = std.Load(
            ty=i32,
            var=std.Var(ty=i32, name="x"),
            indices=[
                std.Range(
                    start=1,
                    step=3,
                )
            ],
        )
        assert slice_without_stop.text() == "x[1::3]"

    def test_structural_equality(self) -> None:
        i32 = std.PrimTy("int32")
        lhs = std.Load(
            ty=i32,
            var=std.Var(ty=i32, name="x"),
            indices=[
                std.Range(
                    start=1,
                    stop=2,
                ),
                std.Range(
                    start=1,
                    stop=2,
                    step=3,
                ),
            ],
        )
        rhs = std.Load(
            ty=i32,
            var=std.Var(ty=i32, name="x"),
            indices=[
                std.Range(
                    start=1,
                    stop=2,
                ),
                std.Range(
                    start=1,
                    stop=2,
                    step=3,
                ),
            ],
        )
        different = std.Load(
            ty=i32,
            var=std.Var(ty=i32, name="x"),
            indices=[2],
        )

        assert tvm_ffi.structural_equal(lhs, rhs)
        assert tvm_ffi.structural_hash(lhs) == tvm_ffi.structural_hash(rhs)
        assert not tvm_ffi.structural_equal(lhs, different)


class TestCast:
    def test_constructor(self) -> None:
        f32 = std.PrimTy("float32")
        value = 1
        node = std.Cast(ty=f32, value=value)

        assert isinstance(node, std.Cast)
        assert tuple(field.name for field in fields(std.Cast)) == ("ty", "value")
        assert node.ty == f32
        assert isinstance(node.value, std.IntImm)
        assert node.value.value == value

    def test_text_format(self) -> None:
        node = std.Cast(
            ty=std.PrimTy("int32"),
            value=1,
        )

        assert node.text() == "std.i32(1)"

    def test_text_format_uses_dtype_abbreviation(self) -> None:
        node = std.Cast(
            ty=std.PrimTy("float32"),
            value=1,
        )

        assert node.text() == "std.f32(1)"

    def test_structural_equality(self) -> None:
        lhs = std.Cast(
            ty=std.PrimTy("float32"),
            value=1,
        )
        rhs = std.Cast(
            ty=std.PrimTy("float32"),
            value=1,
        )
        different = std.Cast(
            ty=std.PrimTy("int32"),
            value=1,
        )

        assert tvm_ffi.structural_equal(lhs, rhs)
        assert tvm_ffi.structural_hash(lhs) == tvm_ffi.structural_hash(rhs)
        assert not tvm_ffi.structural_equal(lhs, different)


class TestCall:
    def test_constructor(self) -> None:
        i32 = std.PrimTy("int32")
        one = 1
        two = 2
        attr = {"tag": "demo"}
        node = std.Call(ty=i32, callee="callee", args=[one, two], attr=attr)

        assert isinstance(node, std.Call)
        assert tuple(field.name for field in fields(std.Call)) == (
            "ty",
            "callee",
            "args",
            "attr",
        )
        assert node.ty == i32
        assert node.callee == "callee"
        args = list(node.args)
        assert isinstance(args[0], std.IntImm)
        assert isinstance(args[1], std.IntImm)
        assert args[0].value == one
        assert args[1].value == two
        assert isinstance(node.attr, std.DictAttrs)
        assert dict(node.attr.values) == attr

    def test_constructor_converts_python_args_and_attr(self) -> None:
        i32 = std.PrimTy("int32")
        node = std.Call(ty=i32, callee="callee", args=[1, 2], attr={"tag": "demo"})

        args = list(node.args)
        assert isinstance(args[0], std.IntImm)
        assert isinstance(args[1], std.IntImm)
        assert isinstance(node.attr, std.DictAttrs)
        assert node.text() == 'std.call(callee, 1, 2, tag="demo")'

    def test_text_format(self) -> None:
        i32 = std.PrimTy("int32")
        node = std.Call(
            ty=i32,
            callee="callee",
            args=[1, 2],
        )

        assert node.text() == "std.call(callee, 1, 2)"

    def test_text_format_without_args(self) -> None:
        i32 = std.PrimTy("int32")
        node = std.Call(ty=i32, callee="callee", args=[])

        assert node.text() == "std.call(callee)"

    def test_text_format_with_func_callee(self) -> None:
        i32 = std.PrimTy("int32")
        x = std.Var(ty=i32, name="x")
        callee = std.Func(
            symbol="helper",
            attrs=None,
            args=[x],
            ret_type=i32,
            body=[std.Return(vars=[x])],
        )
        node = std.Call(
            ty=i32,
            callee=callee,
            args=[1],
        )

        assert node.text() == "std.call(helper, 1)"

    def test_text_format_with_attr(self) -> None:
        i32 = std.PrimTy("int32")
        node = std.Call(
            ty=i32,
            callee="callee",
            args=[1, 2],
            attr={"tag": "demo"},
        )

        assert node.text() == 'std.call(callee, 1, 2, tag="demo")'

    def test_text_format_with_empty_attr(self) -> None:
        i32 = std.PrimTy("int32")
        node = std.Call(
            ty=i32,
            callee="callee",
            args=[],
            attr={},
        )

        assert node.text() == "std.call(callee)"

    def test_text_format_with_sorted_attr_keys(self) -> None:
        i32 = std.PrimTy("int32")
        node = std.Call(
            ty=i32,
            callee="callee",
            args=[],
            attr={"z": 2, "a": 1},
        )

        assert node.text() == "std.call(callee, a=1, z=2)"

    def test_attr_default(self) -> None:
        i32 = std.PrimTy("int32")
        node = std.Call(ty=i32, callee="callee", args=[])

        assert node.attr is None
        assert node.text() == "std.call(callee)"

    def test_attr_field_accepts_dict_attrs_and_none(self) -> None:
        i32 = std.PrimTy("int32")
        attr = {"tag": "demo"}
        with_attr = std.Call(ty=i32, callee="callee", args=[], attr=attr)
        without_attr = std.Call(ty=i32, callee="callee", args=[], attr=None)

        assert isinstance(with_attr.attr, std.DictAttrs)
        assert dict(with_attr.attr.values) == attr
        assert without_attr.attr is None
        assert with_attr.text() == 'std.call(callee, tag="demo")'
        assert without_attr.text() == "std.call(callee)"

    def test_structural_equality(self) -> None:
        i32 = std.PrimTy("int32")
        lhs = std.Call(
            ty=i32,
            callee="callee",
            args=[1, 2],
            attr={"tag": "demo"},
        )
        rhs = std.Call(
            ty=i32,
            callee="callee",
            args=[1, 2],
            attr={"tag": "demo"},
        )
        different = std.Call(
            ty=i32,
            callee="callee",
            args=[1, 2],
            attr={"tag": "other"},
        )

        assert tvm_ffi.structural_equal(lhs, rhs)
        assert tvm_ffi.structural_hash(lhs) == tvm_ffi.structural_hash(rhs)
        assert not tvm_ffi.structural_equal(lhs, different)


class TestIfStmt:
    def test_constructor(self) -> None:
        i32 = std.PrimTy("int32")
        x = std.Var(ty=i32, name="x")
        y = std.Var(ty=i32, name="y")
        two = 2
        cond = std.Lt(ty=i32, a=x, b=two)
        then_body = [std.Return(vars=[x])]
        else_body = [std.Return(vars=[y])]
        node = std.IfStmt(cond=cond, then_body=then_body, else_body=else_body)

        assert isinstance(node, std.IfStmt)
        assert tuple(field.name for field in fields(std.IfStmt)) == (
            "cond",
            "then_body",
            "else_body",
        )
        assert node.cond == cond
        assert list(node.then_body) == then_body
        assert list(node.else_body) == else_body

    def test_text_format(self) -> None:
        i32 = std.PrimTy("int32")
        x = std.Var(ty=i32, name="x")
        y = std.Var(ty=i32, name="y")
        node = std.IfStmt(
            cond=std.Lt(ty=i32, a=x, b=2),
            then_body=[std.Return(vars=[x])],
            else_body=[std.Return(vars=[y])],
        )

        assert node.text() == "if x < 2:\n  return x\nelse:\n  return y"

    def test_text_format_with_empty_else_body(self) -> None:
        i32 = std.PrimTy("int32")
        x = std.Var(ty=i32, name="x")
        node = std.IfStmt(
            cond=std.Lt(ty=i32, a=x, b=2),
            then_body=[std.Return(vars=[x])],
            else_body=[],
        )

        assert node.text() == "if x < 2:\n  return x"

    def test_structural_equality(self) -> None:
        i32 = std.PrimTy("int32")
        lhs = std.IfStmt(
            cond=std.Lt(
                ty=i32,
                a=std.Var(ty=i32, name="x"),
                b=2,
            ),
            then_body=[std.Return(vars=[std.Var(ty=i32, name="x")])],
            else_body=[std.Return(vars=[std.Var(ty=i32, name="y")])],
        )
        rhs = std.IfStmt(
            cond=std.Lt(
                ty=i32,
                a=std.Var(ty=i32, name="x"),
                b=2,
            ),
            then_body=[std.Return(vars=[std.Var(ty=i32, name="x")])],
            else_body=[std.Return(vars=[std.Var(ty=i32, name="y")])],
        )
        different = std.IfStmt(
            cond=std.Gt(
                ty=i32,
                a=std.Var(ty=i32, name="x"),
                b=2,
            ),
            then_body=[std.Return(vars=[std.Var(ty=i32, name="x")])],
            else_body=[std.Return(vars=[std.Var(ty=i32, name="y")])],
        )

        assert tvm_ffi.structural_equal(lhs, rhs)
        assert tvm_ffi.structural_hash(lhs) == tvm_ffi.structural_hash(rhs)
        assert not tvm_ffi.structural_equal(lhs, different)


class TestFor:
    def test_constructor(self) -> None:
        i32 = std.PrimTy("int32")
        x = std.Var(ty=i32, name="x")
        range_node = std.Range(
            start=1,
            stop=2,
        )
        body = [
            std.Store(
                var=x,
                indices=[1],
                rhs=2,
            )
        ]
        node = std.For(
            range_=range_node,
            attrs={"tag": "demo"},
            vars=[x],
            body=body,
        )

        assert isinstance(node, std.For)
        assert isinstance(node, std.Scope)
        assert issubclass(std.For, std.Scope)
        assert tuple(field.name for field in fields(std.For)) == (
            "attrs",
            "vars",
            "body",
            "range_",
        )
        assert node.range_ == range_node
        assert node.attrs is not None
        assert list(node.vars) == [x]
        assert list(node.body) == body

    def test_attrs_field_accepts_dict_attrs_and_none(self) -> None:
        i32 = std.PrimTy("int32")
        x = std.Var(ty=i32, name="x")
        cond_range = std.Range(
            start=1,
            stop=2,
        )
        body = [
            std.Store(
                var=x,
                indices=[1],
                rhs=2,
            )
        ]
        attrs = {"pragma": "unroll"}
        with_attrs = std.For(
            range_=cond_range,
            attrs=attrs,
            vars=[x],
            body=body,
        )
        without_attrs = std.For(
            range_=cond_range,
            attrs=None,
            vars=[x],
            body=body,
        )
        with_empty_attrs = std.For(
            range_=cond_range,
            attrs={},
            vars=[x],
            body=body,
        )

        assert isinstance(with_attrs.attrs, std.DictAttrs)
        assert dict(with_attrs.attrs.values) == attrs
        assert without_attrs.attrs is None
        assert with_attrs.text() == 'for x in range(1, 2, pragma="unroll"):\n  x[1] = 2'
        assert without_attrs.text() == "for x in range(1, 2):\n  x[1] = 2"
        assert with_empty_attrs.text() == "for x in range(1, 2):\n  x[1] = 2"

    def test_text_format(self) -> None:
        i32 = std.PrimTy("int32")
        x = std.Var(ty=i32, name="x")
        node = std.For(
            range_=std.Range(
                start=1,
                stop=2,
            ),
            attrs={"tag": "demo"},
            vars=[x],
            body=[
                std.Store(
                    var=x,
                    indices=[1],
                    rhs=2,
                )
            ],
        )

        assert node.text() == 'for x in range(1, 2, tag="demo"):\n  x[1] = 2'

    def test_text_format_with_sorted_attrs(self) -> None:
        i32 = std.PrimTy("int32")
        x = std.Var(ty=i32, name="x")
        node = std.For(
            range_=std.Range(
                start=1,
                stop=2,
            ),
            attrs={"z": 3, "a": 1},
            vars=[x],
            body=[
                std.Store(
                    var=x,
                    indices=[1],
                    rhs=2,
                )
            ],
        )

        assert node.text() == "for x in range(1, 2, a=1, z=3):\n  x[1] = 2"

    def test_structural_equality(self) -> None:
        i32 = std.PrimTy("int32")
        lhs = std.For(
            range_=std.Range(
                start=1,
                stop=2,
            ),
            attrs={"tag": "demo"},
            vars=[std.Var(ty=i32, name="x")],
            body=[
                std.Store(
                    var=std.Var(ty=i32, name="x"),
                    indices=[1],
                    rhs=2,
                )
            ],
        )
        rhs = std.For(
            range_=std.Range(
                start=1,
                stop=2,
            ),
            attrs={"tag": "demo"},
            vars=[std.Var(ty=i32, name="x")],
            body=[
                std.Store(
                    var=std.Var(ty=i32, name="x"),
                    indices=[1],
                    rhs=2,
                )
            ],
        )
        different = std.For(
            range_=std.Range(
                start=1,
                stop=3,
            ),
            attrs={"tag": "demo"},
            vars=[std.Var(ty=i32, name="x")],
            body=[
                std.Store(
                    var=std.Var(ty=i32, name="x"),
                    indices=[1],
                    rhs=2,
                )
            ],
        )

        assert tvm_ffi.structural_equal(lhs, rhs)
        assert tvm_ffi.structural_hash(lhs) == tvm_ffi.structural_hash(rhs)
        assert not tvm_ffi.structural_equal(lhs, different)


class TestWhile:
    def test_constructor(self) -> None:
        i32 = std.PrimTy("int32")
        x = std.Var(ty=i32, name="x")
        y = std.Var(ty=i32, name="y")
        cond = std.Lt(ty=i32, a=x, b=2)
        body = [std.ExprBind(vars=[y], attrs=None, expr=2)]
        node = std.While(
            cond=cond,
            attrs={"tag": "demo"},
            vars=[x],
            body=body,
        )

        assert isinstance(node, std.While)
        assert isinstance(node, std.Scope)
        assert issubclass(std.While, std.Scope)
        assert tuple(field.name for field in fields(std.While)) == (
            "attrs",
            "vars",
            "body",
            "cond",
        )
        assert node.cond == cond
        assert node.attrs is not None
        assert list(node.vars) == [x]
        assert list(node.body) == body

    def test_attrs_field_accepts_dict_attrs_and_none(self) -> None:
        i32 = std.PrimTy("int32")
        x = std.Var(ty=i32, name="x")
        y = std.Var(ty=i32, name="y")
        cond = std.Lt(ty=i32, a=x, b=2)
        body = [
            std.ExprBind(
                expr=2,
                attrs=None,
                vars=[y],
            )
        ]
        attrs = {"pragma": "pipeline"}
        with_attrs = std.While(
            cond=cond,
            attrs=attrs,
            vars=[x],
            body=body,
        )
        without_attrs = std.While(
            cond=cond,
            attrs=None,
            vars=[x],
            body=body,
        )
        with_empty_attrs = std.While(
            cond=cond,
            attrs={},
            vars=[],
            body=body,
        )

        assert isinstance(with_attrs.attrs, std.DictAttrs)
        assert dict(with_attrs.attrs.values) == attrs
        assert without_attrs.attrs is None
        assert with_attrs.text() == 'with std.While(x < 2, pragma="pipeline") as x:\n  y = 2'
        assert without_attrs.text() == "with std.While(x < 2) as x:\n  y = 2"
        assert with_empty_attrs.text() == "while x < 2:\n  y = 2"

    def test_text_format(self) -> None:
        i32 = std.PrimTy("int32")
        x = std.Var(ty=i32, name="x")
        node = std.While(
            cond=std.Lt(ty=i32, a=x, b=2),
            attrs={"tag": "demo"},
            vars=[x],
            body=[
                std.ExprBind(
                    expr=2,
                    attrs=None,
                    vars=[std.Var(ty=i32, name="y")],
                )
            ],
        )

        assert node.text() == 'with std.While(x < 2, tag="demo") as x:\n  y = 2'

    def test_text_format_with_simple_while(self) -> None:
        i32 = std.PrimTy("int32")
        x = std.Var(ty=i32, name="x")
        node = std.While(
            cond=std.Lt(ty=i32, a=x, b=2),
            attrs=None,
            vars=[],
            body=[
                std.ExprBind(
                    expr=2,
                    attrs=None,
                    vars=[std.Var(ty=i32, name="y")],
                )
            ],
        )

        assert node.text() == "while x < 2:\n  y = 2"

    def test_text_format_with_sorted_attrs_and_vars(self) -> None:
        i32 = std.PrimTy("int32")
        x = std.Var(ty=i32, name="x")
        state = std.Var(ty=i32, name="state")
        node = std.While(
            cond=std.Lt(ty=i32, a=x, b=2),
            attrs={"z": 3, "a": 1},
            vars=[x, state],
            body=[
                std.ExprBind(
                    expr=2,
                    attrs=None,
                    vars=[std.Var(ty=i32, name="y")],
                )
            ],
        )

        assert node.text() == "with std.While(x < 2, a=1, z=3) as x, state:\n  y = 2"

    def test_structural_equality(self) -> None:
        i32 = std.PrimTy("int32")
        lhs = std.While(
            cond=std.Lt(
                ty=i32,
                a=std.Var(ty=i32, name="x"),
                b=2,
            ),
            attrs={"tag": "demo"},
            vars=[std.Var(ty=i32, name="x")],
            body=[
                std.ExprBind(
                    expr=2,
                    attrs=None,
                    vars=[std.Var(ty=i32, name="y")],
                )
            ],
        )
        rhs = std.While(
            cond=std.Lt(
                ty=i32,
                a=std.Var(ty=i32, name="x"),
                b=2,
            ),
            attrs={"tag": "demo"},
            vars=[std.Var(ty=i32, name="x")],
            body=[
                std.ExprBind(
                    expr=2,
                    attrs=None,
                    vars=[std.Var(ty=i32, name="y")],
                )
            ],
        )
        different = std.While(
            cond=std.Gt(
                ty=i32,
                a=std.Var(ty=i32, name="x"),
                b=2,
            ),
            attrs={"tag": "demo"},
            vars=[std.Var(ty=i32, name="x")],
            body=[
                std.ExprBind(
                    expr=2,
                    attrs=None,
                    vars=[std.Var(ty=i32, name="y")],
                )
            ],
        )

        assert tvm_ffi.structural_equal(lhs, rhs)
        assert tvm_ffi.structural_hash(lhs) == tvm_ffi.structural_hash(rhs)
        assert not tvm_ffi.structural_equal(lhs, different)


class TestScope:
    def test_constructor(self) -> None:
        i32 = std.PrimTy("int32")
        x = std.Var(ty=i32, name="x")
        body = [std.Return(vars=[x])]
        node = std.Scope(
            attrs={"tag": "demo"},
            vars=[x],
            body=body,
        )

        assert isinstance(node, std.Scope)
        assert tuple(field.name for field in fields(std.Scope)) == (
            "attrs",
            "vars",
            "body",
        )
        assert node.attrs is not None
        assert list(node.vars) == [x]
        assert list(node.body) == body

    def test_attrs_field_accepts_dict_attrs_and_none(self) -> None:
        i32 = std.PrimTy("int32")
        x = std.Var(ty=i32, name="x")
        body = [std.Return(vars=[x])]
        attrs = {"pragma": "scope"}
        with_attrs = std.Scope(attrs=attrs, vars=[], body=body)
        without_attrs = std.Scope(attrs=None, vars=[], body=body)
        with_empty_attrs = std.Scope(
            attrs={},
            vars=[],
            body=body,
        )

        assert isinstance(with_attrs.attrs, std.DictAttrs)
        assert dict(with_attrs.attrs.values) == attrs
        assert without_attrs.attrs is None
        assert with_attrs.text() == 'with std.Scope(pragma="scope"):\n  return x'
        assert without_attrs.text() == "return x"
        assert with_empty_attrs.text() == "return x"

    def test_text_format(self) -> None:
        i32 = std.PrimTy("int32")
        x = std.Var(ty=i32, name="x")
        node = std.Scope(attrs=None, vars=[x], body=[std.Return(vars=[x])])

        assert node.text() == "with std.Scope() as x:\n  return x"

    def test_text_format_with_simple_scope(self) -> None:
        i32 = std.PrimTy("int32")
        x = std.Var(ty=i32, name="x")
        node = std.Scope(attrs=None, vars=[], body=[std.Return(vars=[x])])

        assert node.text() == "return x"

    def test_text_format_with_sorted_attrs_and_vars(self) -> None:
        i32 = std.PrimTy("int32")
        x = std.Var(ty=i32, name="x")
        state = std.Var(ty=i32, name="state")
        node = std.Scope(
            attrs={"z": 3, "a": 1},
            vars=[x, state],
            body=[std.Return(vars=[x])],
        )

        assert node.text() == "with std.Scope(a=1, z=3) as x, state:\n  return x"

    def test_structural_equality(self) -> None:
        i32 = std.PrimTy("int32")
        lhs = std.Scope(
            attrs={"tag": "demo"},
            vars=[std.Var(ty=i32, name="x")],
            body=[std.Return(vars=[std.Var(ty=i32, name="x")])],
        )
        rhs = std.Scope(
            attrs={"tag": "demo"},
            vars=[std.Var(ty=i32, name="x")],
            body=[std.Return(vars=[std.Var(ty=i32, name="x")])],
        )
        different = std.Scope(
            attrs={"tag": "other"},
            vars=[std.Var(ty=i32, name="x")],
            body=[std.Return(vars=[std.Var(ty=i32, name="x")])],
        )

        assert tvm_ffi.structural_equal(lhs, rhs)
        assert tvm_ffi.structural_hash(lhs) == tvm_ffi.structural_hash(rhs)
        assert not tvm_ffi.structural_equal(lhs, different)


class TestBind:
    def test_constructor_is_disabled(self) -> None:
        ctor = cast(Any, std.Bind)

        with pytest.raises(TypeError, match="cannot be constructed directly"):
            ctor()
        with pytest.raises(TypeError, match="cannot be constructed directly"):
            ctor(vars=[], attrs=None)

    def test_fields(self) -> None:
        assert tuple(field.name for field in fields(std.Bind)) == ("vars", "attrs")


class TestExprBind:
    def test_constructor(self) -> None:
        i32 = std.PrimTy("int32")
        expr = 1
        attrs = {"tag": "demo"}
        var = std.Var(ty=i32, name="y")
        node = std.ExprBind(vars=[var], attrs=attrs, expr=expr)

        assert isinstance(node, std.ExprBind)
        assert tuple(field.name for field in fields(std.ExprBind)) == ("vars", "attrs", "expr")
        assert list(node.vars) == [var]
        assert isinstance(node.attrs, std.DictAttrs)
        assert dict(node.attrs.values) == attrs
        assert isinstance(node.expr, std.IntImm)
        assert node.expr.value == expr

    def test_attrs_field_accepts_dict_attrs_and_none(self) -> None:
        i32 = std.PrimTy("int32")
        var = std.Var(ty=i32, name="y")
        attrs = {"tag": "demo"}
        with_attrs = std.ExprBind(vars=[var], attrs=attrs, expr=1)
        with_empty_attrs = std.ExprBind(vars=[var], attrs={}, expr=1)
        without_attrs = std.ExprBind(vars=[var], attrs=None, expr=1)

        assert isinstance(with_attrs.attrs, std.DictAttrs)
        assert dict(with_attrs.attrs.values) == attrs
        assert isinstance(with_empty_attrs.attrs, std.DictAttrs)
        assert without_attrs.attrs is None
        assert with_attrs.text() == 'y = std.bind(1, tag="demo")'
        assert with_empty_attrs.text() == "y = 1"
        assert without_attrs.text() == "y = 1"

    def test_text_format_without_vars(self) -> None:
        assert (
            std.ExprBind(vars=[], attrs={"tag": "demo"}, expr=1).text() == 'std.bind(1, tag="demo")'
        )
        assert std.ExprBind(vars=[], attrs=None, expr=1).text() == "1"

    def test_text_format_with_single_var(self) -> None:
        i32 = std.PrimTy("int32")
        with_attrs = std.ExprBind(
            vars=[std.Var(ty=i32, name="y")],
            attrs={"tag": "demo"},
            expr=1,
        )
        without_attrs = std.ExprBind(
            vars=[std.Var(ty=i32, name="y")],
            attrs=None,
            expr=1,
        )

        assert with_attrs.text() == 'y = std.bind(1, tag="demo")'
        assert without_attrs.text() == "y = 1"

    def test_text_format_with_multiple_vars(self) -> None:
        i32 = std.PrimTy("int32")
        with_attrs = std.ExprBind(
            vars=[std.Var(ty=i32, name="y"), std.Var(ty=i32, name="z")],
            attrs={"z": 2, "a": 1},
            expr=1,
        )
        without_attrs = std.ExprBind(
            vars=[std.Var(ty=i32, name="y"), std.Var(ty=i32, name="z")],
            attrs=None,
            expr=1,
        )

        assert with_attrs.text() == "y, z = std.bind(1, a=1, z=2)"
        assert without_attrs.text() == "y, z = 1"

    def test_structural_equality(self) -> None:
        i32 = std.PrimTy("int32")
        lhs = std.ExprBind(
            vars=[std.Var(ty=i32, name="y")],
            attrs={"tag": "demo"},
            expr=1,
        )
        rhs = std.ExprBind(
            vars=[std.Var(ty=i32, name="renamed")],
            attrs={"tag": "demo"},
            expr=1,
        )
        different = std.ExprBind(
            vars=[std.Var(ty=i32, name="y")],
            attrs={"tag": "demo"},
            expr=2,
        )

        assert tvm_ffi.structural_equal(lhs, rhs)
        assert tvm_ffi.structural_hash(lhs) == tvm_ffi.structural_hash(rhs)
        assert not tvm_ffi.structural_equal(lhs, different)


class TestVarDef:
    def test_constructor(self) -> None:
        i32 = std.PrimTy("int32")
        attrs = {"tag": "demo"}
        var = std.Var(ty=i32, name="y")
        node = std.VarDef(vars=[var], attrs=attrs)

        assert isinstance(node, std.VarDef)
        assert tuple(field.name for field in fields(std.VarDef)) == ("vars", "attrs")
        assert list(node.vars) == [var]
        assert isinstance(node.attrs, std.DictAttrs)
        assert dict(node.attrs.values) == attrs

    def test_text_format_without_attrs(self) -> None:
        i32 = std.PrimTy("int32")
        node = std.VarDef(vars=[std.Var(ty=i32, name="y")], attrs=None)

        assert node.text() == "y = std.var_def(i32)"

    def test_text_format_with_attrs(self) -> None:
        i32 = std.PrimTy("int32")
        node = std.VarDef(vars=[std.Var(ty=i32, name="y")], attrs={"tag": "demo"})

        assert node.text() == 'y = std.var_def(i32, tag="demo")'
        assert (
            std.VarDef(vars=[std.Var(ty=i32, name="y")], attrs={}).text() == "y = std.var_def(i32)"
        )

    def test_text_format_with_multiple_vars(self) -> None:
        i32 = std.PrimTy("int32")
        bf16_3x12 = std.TensorTy(shape=[3, 12], dtype="bfloat16")
        with_attrs = std.VarDef(
            vars=[std.Var(ty=i32, name="y"), std.Var(ty=bf16_3x12, name="z")],
            attrs={"tag": "demo"},
        )
        without_attrs = std.VarDef(
            vars=[std.Var(ty=i32, name="y"), std.Var(ty=bf16_3x12, name="z")],
            attrs=None,
        )

        assert with_attrs.text() == 'y, z = std.var_def(i32, bf16[3, 12], tag="demo")'
        assert without_attrs.text() == "y, z = std.var_def(i32, bf16[3, 12])"

    def test_text_format_without_vars(self) -> None:
        assert std.VarDef(vars=[], attrs=None).text() == "pass"
        assert std.VarDef(vars=[], attrs={"tag": "demo"}).text() == 'std.var_def(tag="demo")'

    def test_structural_equality(self) -> None:
        i32 = std.PrimTy("int32")
        lhs = std.VarDef(vars=[std.Var(ty=i32, name="y")], attrs={"tag": "demo"})
        rhs = std.VarDef(vars=[std.Var(ty=i32, name="renamed")], attrs={"tag": "demo"})
        different = std.VarDef(vars=[std.Var(ty=i32, name="y")], attrs={"tag": "other"})

        assert tvm_ffi.structural_equal(lhs, rhs)
        assert tvm_ffi.structural_hash(lhs) == tvm_ffi.structural_hash(rhs)
        assert not tvm_ffi.structural_equal(lhs, different)


class TestStore:
    def test_constructor(self) -> None:
        i32 = std.PrimTy("int32")
        x = std.Var(ty=i32, name="x")
        index = 1
        rhs = 2
        node = std.Store(var=x, indices=[index], rhs=rhs)

        assert isinstance(node, std.Store)
        assert tuple(field.name for field in fields(std.Store)) == ("var", "indices", "rhs")
        assert node.var == x
        indices = list(node.indices)
        assert isinstance(indices[0], std.Range)
        assert isinstance(indices[0].start, std.IntImm)
        assert indices[0].start.value == index
        assert isinstance(node.rhs, std.IntImm)
        assert node.rhs.value == rhs

    def test_text_format(self) -> None:
        i32 = std.PrimTy("int32")
        node = std.Store(
            var=std.Var(ty=i32, name="x"),
            indices=[1],
            rhs=2,
        )

        assert node.text() == "x[1] = 2"

    def test_text_format_index_variants(self) -> None:
        i32 = std.PrimTy("int32")

        no_indices = std.Store(
            var=std.Var(ty=i32, name="x"),
            indices=[],
            rhs=2,
        )
        assert no_indices.text() == "x[()] = 2"

        slice_without_step = std.Store(
            var=std.Var(ty=i32, name="x"),
            indices=[
                std.Range(
                    start=1,
                    stop=2,
                )
            ],
            rhs=3,
        )
        assert slice_without_step.text() == "x[1:2] = 3"

        slice_without_start = std.Store(
            var=std.Var(ty=i32, name="x"),
            indices=[
                std.Range(
                    start=None,
                    stop=2,
                )
            ],
            rhs=3,
        )
        assert slice_without_start.text() == "x[:2] = 3"

        mixed_indices = std.Store(
            var=std.Var(ty=i32, name="x"),
            indices=[
                std.Range(
                    start=1,
                    stop=2,
                ),
                3,
            ],
            rhs=4,
        )
        assert mixed_indices.text() == "x[1:2, 3] = 4"

    def test_constructor_converts_python_indices_and_rhs(self) -> None:
        i32 = std.PrimTy("int32")
        node = std.Store(
            var=std.Var(ty=i32, name="x"),
            indices=[1],
            rhs=2,
        )

        index = next(iter(node.indices))
        assert isinstance(index, std.Range)
        assert isinstance(index.start, std.IntImm)
        assert isinstance(node.rhs, std.IntImm)
        assert node.text() == "x[1] = 2"

    def test_structural_equality(self) -> None:
        i32 = std.PrimTy("int32")
        lhs = std.Store(
            var=std.Var(ty=i32, name="x"),
            indices=[1],
            rhs=2,
        )
        rhs = std.Store(
            var=std.Var(ty=i32, name="x"),
            indices=[1],
            rhs=2,
        )
        different = std.Store(
            var=std.Var(ty=i32, name="x"),
            indices=[1],
            rhs=3,
        )

        assert tvm_ffi.structural_equal(lhs, rhs)
        assert tvm_ffi.structural_hash(lhs) == tvm_ffi.structural_hash(rhs)
        assert not tvm_ffi.structural_equal(lhs, different)


class TestReturn:
    def test_constructor(self) -> None:
        i32 = std.PrimTy("int32")
        x = std.Var(ty=i32, name="x")
        y = std.Var(ty=i32, name="y")
        node = std.Return(vars=[x, y])

        assert isinstance(node, std.Return)
        assert tuple(field.name for field in fields(std.Return)) == ("vars",)
        assert list(node.vars) == [x, y]

    def test_text_format(self) -> None:
        i32 = std.PrimTy("int32")
        node = std.Return(vars=[std.Var(ty=i32, name="x"), std.Var(ty=i32, name="y")])

        assert node.text() == "return (x, y)"

    def test_structural_equality(self) -> None:
        i32 = std.PrimTy("int32")
        lhs = std.Return(vars=[std.Var(ty=i32, name="x"), std.Var(ty=i32, name="y")])
        rhs = std.Return(
            vars=[std.Var(ty=i32, name="renamed_x"), std.Var(ty=i32, name="renamed_y")]
        )
        different = std.Return(vars=[std.Var(ty=i32, name="x")])

        assert tvm_ffi.structural_equal(lhs, rhs)
        assert tvm_ffi.structural_hash(lhs) == tvm_ffi.structural_hash(rhs)
        assert not tvm_ffi.structural_equal(lhs, different)


class TestYield:
    def test_constructor(self) -> None:
        x = std.Var(ty=std.PrimTy("int32"), name="x")
        y = std.Var(ty=std.PrimTy("int32"), name="y")
        node = std.Yield(vars=[x, y])

        assert isinstance(node, std.Yield)
        assert tuple(field.name for field in fields(std.Yield)) == ("vars",)
        assert list(node.vars) == [x, y]

    def test_text_format(self) -> None:
        node = std.Yield(vars=[std.Var(ty=std.PrimTy("int32"), name="x")])

        assert node.text() == "yield x"

    def test_text_format_with_multiple_vars(self) -> None:
        i32 = std.PrimTy("int32")
        node = std.Yield(vars=[std.Var(ty=i32, name="x"), std.Var(ty=i32, name="y")])

        assert node.text() == "yield (x, y)"

    def test_text_format_without_vars(self) -> None:
        node = std.Yield(vars=[])

        assert node.text() == "yield"

    def test_structural_equality(self) -> None:
        i32 = std.PrimTy("int32")
        lhs = std.Yield(vars=[std.Var(ty=i32, name="x"), std.Var(ty=i32, name="y")])
        rhs = std.Yield(vars=[std.Var(ty=i32, name="renamed_x"), std.Var(ty=i32, name="renamed_y")])
        different = std.Yield(vars=[std.Var(ty=i32, name="x")])

        assert tvm_ffi.structural_equal(lhs, rhs)
        assert tvm_ffi.structural_hash(lhs) == tvm_ffi.structural_hash(rhs)
        assert not tvm_ffi.structural_equal(lhs, different)


class TestBreak:
    def test_constructor(self) -> None:
        node = std.Break()

        assert isinstance(node, std.Break)
        assert tuple(field.name for field in fields(std.Break)) == ()

    def test_text_format(self) -> None:
        node = std.Break()

        assert node.text() == "break"

    def test_structural_equality(self) -> None:
        lhs = std.Break()
        rhs = std.Break()
        different = std.Continue()

        assert tvm_ffi.structural_equal(lhs, rhs)
        assert tvm_ffi.structural_hash(lhs) == tvm_ffi.structural_hash(rhs)
        assert not tvm_ffi.structural_equal(lhs, different)


class TestContinue:
    def test_constructor(self) -> None:
        node = std.Continue()

        assert isinstance(node, std.Continue)
        assert tuple(field.name for field in fields(std.Continue)) == ()

    def test_text_format(self) -> None:
        node = std.Continue()

        assert node.text() == "continue"

    def test_structural_equality(self) -> None:
        lhs = std.Continue()
        rhs = std.Continue()
        different = std.Break()

        assert tvm_ffi.structural_equal(lhs, rhs)
        assert tvm_ffi.structural_hash(lhs) == tvm_ffi.structural_hash(rhs)
        assert not tvm_ffi.structural_equal(lhs, different)


class TestDictAttrs:
    def test_constructor(self) -> None:
        node = std.DictAttrs(values={"tag": "demo"})

        assert isinstance(node, std.DictAttrs)
        assert tuple(field.name for field in fields(std.DictAttrs)) == ("values",)
        assert dict(node.values) == {"tag": "demo"}

    def test_text_format(self) -> None:
        node = std.DictAttrs(values={"tag": "demo"})

        assert node.text() == 'std.DictAttrs(tag="demo")'

    def test_text_format_sorts_keys(self) -> None:
        node = std.DictAttrs(values={"z": 2, "a": 1})

        assert node.text() == "std.DictAttrs(a=1, z=2)"

    def test_text_format_empty(self) -> None:
        node = std.DictAttrs(values={})

        assert node.text() == "std.DictAttrs()"

    def test_text_print_hook_returns_keyword_only_call_ast(self) -> None:
        node = std.DictAttrs(values={"z": 2, "a": 1})
        ast_node = pyast.IRPrinter()(node, AccessPath.root())

        assert isinstance(ast_node, pyast.Call)
        assert list(ast_node.args) == []
        assert list(ast_node.kwargs_keys) == ["a", "z"]
        assert [value.to_python() for value in ast_node.kwargs_values] == ["1", "2"]

    def test_structural_equality(self) -> None:
        lhs = std.DictAttrs(values={"tag": "demo"})
        rhs = std.DictAttrs(values={"tag": "demo"})
        different = std.DictAttrs(values={"tag": "other"})

        assert tvm_ffi.structural_equal(lhs, rhs)
        assert tvm_ffi.structural_hash(lhs) == tvm_ffi.structural_hash(rhs)
        assert not tvm_ffi.structural_equal(lhs, different)
