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
# ruff: noqa: D100, D101, D102, D103, D106
from __future__ import annotations

import dataclasses as dc
from numbers import Number
from typing import Any, Callable

from tvm_ffi import dataclasses as tdc
from tvm_ffi import ir_traits, pyast, std
from tvm_ffi._pyast_parser import MISSING, Parser, Source

_FIELD_PREFIX = "$field:"


def _strip_field(ref: str) -> str:
    if not ref.startswith(_FIELD_PREFIX):
        raise ValueError(
            f"FuncFrame: trait field reference {ref!r} does not start with "
            f"{_FIELD_PREFIX!r}; only direct field references are supported."
        )
    return ref[len(_FIELD_PREFIX) :]


@tdc.py_class("T.AllocTensor")
class AllocTensor(std.SingleBinding):
    shape: list[std.Expr]
    dtype: str

    def __ffi_text_print__(self, printer: pyast.IRPrinter, path: Any) -> pyast.Assign:
        lhs = printer.var_def(self.value.name, self.value, None)
        annotation = printer(self.value.ty, path.attr("value").attr("ty"))
        shape = pyast.List(
            [printer(dim, path.attr("shape").array_item(i)) for i, dim in enumerate(self.shape)]
        )
        dtype = printer(self.dtype, path.attr("dtype"))
        rhs = pyast.Call(pyast.Attr(pyast.Id("T"), "alloc_tensor"), [shape, dtype], [], [])
        return pyast.Assign(lhs, rhs, annotation)


@dc.dataclass
class FuncFrame:
    ir_cls: type
    attrs: dict[str, Any]
    symbol: str = ""
    args: list[std.Value] = dc.field(default_factory=list)
    body: list[std.Stmt] = dc.field(default_factory=list)
    ret_type: Any = None

    def __init__(self, ir_cls: type, attrs: dict[str, Any]) -> None:
        self.ir_cls = ir_cls
        self.attrs = attrs
        self.symbol = ""
        self.args = []
        self.body = []
        self.ret_type = None

    def to_dialect(self) -> Any:
        trait = getattr(self.ir_cls, "__ffi_ir_traits__", None)
        if not isinstance(trait, ir_traits.FuncTraits):
            raise TypeError(
                f"FuncFrame.to_dialect: {self.ir_cls.__name__} has no "
                f"FuncTraits; got {type(trait).__name__!r}."
            )

        region = trait.region
        fields: dict[str, Any] = {_strip_field(trait.symbol): self.symbol}
        if trait.attrs is not None:
            fields[_strip_field(trait.attrs)] = std.DictAttrs(values=self.attrs)
        if region.def_values is not None:
            fields[_strip_field(region.def_values)] = self.args
        fields[_strip_field(region.body)] = self.body
        if region.ret is not None:
            fields[_strip_field(region.ret)] = self.ret_type
        return self.ir_cls(**fields)


@dc.dataclass
class ForFrame:
    ir_cls: type
    range_: std.Range
    value: std.Value
    attrs: std.Attrs
    body: list[std.Stmt] = dc.field(default_factory=list)
    carry_inits: list[std.Expr] = dc.field(default_factory=list)

    def __init__(
        self,
        ir_cls: type,
        range_: std.Range,
        value: std.Value,
        attrs: std.Attrs,
    ) -> None:
        self.ir_cls = ir_cls
        self.range_ = range_
        self.value = value
        self.attrs = attrs
        self.body = []
        self.carry_inits = []

    def to_dialect(self) -> Any:
        trait = getattr(self.ir_cls, "__ffi_ir_traits__", None)
        if not isinstance(trait, ir_traits.ForTraits):
            raise TypeError(
                f"ForFrame.to_dialect: {self.ir_cls.__name__} has no "
                f"ForTraits; got {type(trait).__name__!r}."
            )

        region = trait.region
        fields: dict[str, Any] = {_strip_field(region.body): self.body}
        if region.def_values is not None:
            fields[_strip_field(region.def_values)] = [self.value]
        if region.def_expr is not None:
            fields[_strip_field(region.def_expr)] = self.range_
        if trait.attrs is not None:
            fields[_strip_field(trait.attrs)] = self.attrs
        if trait.carry_init is not None:
            fields[_strip_field(trait.carry_init)] = self.carry_inits
        return self.ir_cls(**fields)


@dc.dataclass
class IFThenElseFrame:
    ir_cls: type
    cond: std.Expr
    then_body: list[std.Stmt]
    else_body: list[std.Stmt]

    def to_dialect(self) -> Any:
        trait = getattr(self.ir_cls, "__ffi_ir_traits__", None)
        if not isinstance(trait, ir_traits.IfTraits):
            raise TypeError(
                f"IFThenElseFrame.to_dialect: {self.ir_cls.__name__} has no "
                f"IfTraits; got {type(trait).__name__!r}."
            )
        fields: dict[str, Any] = {
            _strip_field(trait.cond): self.cond,
            _strip_field(trait.then_region.body): self.then_body,
        }
        if trait.else_region is not None:
            fields[_strip_field(trait.else_region.body)] = self.else_body
        return self.ir_cls(**fields)


class T:
    class Expr:
        pass

    class Tensor:
        # TODO: add __getitem__ for buffer loading
        # TODO: add __setitem__ for buffer storing
        pass

    class f32(Expr):
        def __class_getitem__(cls, *shape: int) -> T.Tensor:
            return std.TensorTy(shape=shape, dtype="float32")

    class i32(Expr):
        def __class_getitem__(cls, *shape: int) -> T.Tensor:
            return std.TensorTy(shape=shape, dtype="int32")

    def prim_func(**kwargs: Any) -> Callable[..., Any]:
        def decorator(func: Any) -> Any:
            return func

        decorator.__ffi_parse__ = FuncFrame(
            ir_cls=std.Func,
            attrs=kwargs,
        )
        return decorator

    def range(
        start: int,
        stop: std.Expr | int | object = MISSING,
        step: std.Expr | int = 1,
        /,
    ) -> ForFrame:
        if MISSING.is_(stop):
            start, stop = 0, start
        if isinstance(start, int):
            start = std.IntImm(std.AnyTy(), start)
        if isinstance(stop, int):
            stop = std.IntImm(std.AnyTy(), stop)
        if isinstance(step, int):
            if step == 1:
                step = None
            else:
                step = std.IntImm(std.AnyTy(), step)

        return ForFrame(
            ir_cls=std.For,
            range_=std.Range(start=start, stop=stop, step=step),
            value=std.Value(name="_", ty=std.PrimTy(dtype="int32")),
            attrs=std.Attrs(),
        )

    def alloc_tensor(shape: tuple[int, ...], dtype: str) -> T.Tensor:
        ty = std.TensorTy(shape=shape, dtype=dtype)
        return AllocTensor(
            shape=shape,
            dtype=dtype,
            value=std.Value(ty, "_"),
        )

    def Cast(dtype: str, value: std.Expr) -> std.Expr:
        return std.Cast(std.PrimTy(dtype), value=value)


def _tensor_load(tensor: std.Value, *indices: std.Expr) -> std.Expr:
    return std.Load(
        value=tensor,
        indices=indices,
        ty=std.PrimTy(tensor.ty.dtype),
    )


def _tensor_store(tensor: std.Value, value: std.Expr | Number, *indices: std.Expr) -> std.Stmt:
    if isinstance(value, Number):
        value = std.Expr._make(value, ty=std.PrimTy(tensor.ty.dtype))
    assert value.ty.dtype == tensor.ty.dtype
    return std.Store(
        value=tensor,
        indices=indices,
        rhs=value,
    )


def _if_stmt(cond: std.Expr, then_body: list[std.Stmt], else_body: list[std.Stmt]) -> std.Stmt:
    return IFThenElseFrame(
        ir_cls=std.IfStmt,
        cond=cond,
        then_body=then_body,
        else_body=else_body,
    ).to_dialect()


_MORE_GENERICS: dict[tuple[str, type[std.Ty] | str], Callable[..., Any]] = {
    ("__load__", std.TensorTy): _tensor_load,
    ("__store__", std.TensorTy): _tensor_store,
    ("__add__", std.PrimTy): std.Add._make,
    ("__sub__", std.PrimTy): std.Sub._make,
    ("__lt__", std.PrimTy): std.Lt._make,
    ("__mul__", std.PrimTy): std.Mul._make,
    ("__if_stmt__", "T"): _if_stmt,
}


def main() -> None:
    parser = Parser(
        source=Source(
            program="""
@T.prim_func(private=True, my_attr="hello")
def main_func(p0: T.f32[30], p1: T.i32[1], hybrid_nms: T.f32[30]):
    argsort_nms_cpu = T.alloc_tensor((5,), "int32")
    for i in range(1):
        nkeep = T.alloc_tensor((1,), "int32")
        if 0 < p1[i]:
            nkeep[0] = p1[i]
            if 2 < nkeep[0]:
                nkeep[0] = 2
            for j in range(nkeep[0]):
                for k in range(6):
                    hybrid_nms[i * 30 + j * 6 + k] = p0[i * 30 + argsort_nms_cpu[i * 5 + j] * 6 + k]
                hybrid_nms[i * 5 + j] = T.Cast("float32", argsort_nms_cpu[i * 5 + j])
            if 2 < p1[i]:
                for j in T.range(p1[i] - nkeep[0]): # TODO: replace with T.parallel
                    for k in range(6):
                        hybrid_nms[i * 30 + j * 6 + nkeep[0] * 6 + k] = -1.0
                    hybrid_nms[i * 5 + j + nkeep[0]] = -1.0
""",
            feature_version=(3, 14),
        ),
        extra_vars={
            "T": T,
            "range": T.range,
        },
    )
    for key, handler in _MORE_GENERICS.items():
        parser.generics[key] = handler
    parser.dialect_stack.append("T")
    (func,) = parser.run()
    # print(funcs)
    print(pyast.to_python(func))


if __name__ == "__main__":
    main()
