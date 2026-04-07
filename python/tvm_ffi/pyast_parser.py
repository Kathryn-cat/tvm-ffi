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
"""Minimal parser: TVM-FFI AST → user IR via value-driven dispatch."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from tvm_ffi import pyast


class ParseError(Exception):
    pass


# ---- SurfaceObject base class ----


class SurfaceObject:
    """Base class for parser dispatch targets.

    Any object with __ffi_text_parse__(parser, node) participates in
    value-driven dispatch. This base class routes to position-specific
    methods based on the AST node type.
    """

    def __ffi_text_parse__(self, parser: IRParser, node: pyast.Node) -> Any:
        if isinstance(node, pyast.Function):
            return self.parse_function(parser, node)
        if isinstance(node, pyast.For):
            return self.parse_for(parser, node)
        if isinstance(node, pyast.With):
            return self.parse_with(parser, node)
        if isinstance(node, pyast.Assign):
            return self.parse_assign(parser, node)
        raise ParseError(
            f"{type(self).__name__} cannot appear in "
            f"{type(node).__name__} position"
        )

    def parse_function(self, parser, node):
        raise ParseError(f"{type(self).__name__}: parse_function not implemented")

    def parse_for(self, parser, node):
        raise ParseError(f"{type(self).__name__}: parse_for not implemented")

    def parse_with(self, parser, node):
        raise ParseError(f"{type(self).__name__}: parse_with not implemented")

    def parse_assign(self, parser, node):
        raise ParseError(f"{type(self).__name__}: parse_assign not implemented")


# ---- VarTable ----


class VarTable:
    """Frame-based scoped variable bindings."""

    def __init__(self):
        self.frames: list[dict[str, Any]] = [{}]

    def define(self, name: str, value: Any) -> None:
        self.frames[-1][name] = value

    def get(self, name: str) -> Any | None:
        for frame in reversed(self.frames):
            if name in frame:
                return frame[name]
        return None

    @contextmanager
    def frame(self):
        self.frames.append({})
        try:
            yield
        finally:
            self.frames.pop()


# ---- IRParser ----


class IRParser:
    """Stateful parser: TVM-FFI AST → user IR.

    Uses value-driven dispatch: evaluate an expression, let the
    returned value handle parsing via __ffi_text_parse__.
    """

    def __init__(self, lang_modules: dict[str, object] | None = None):
        self.var_table = VarTable()
        self.lang_modules = lang_modules or {}

    def parse(self, source: str | pyast.Node) -> Any:
        if isinstance(source, str):
            node = pyast.from_py(source)
        else:
            node = source
        # from_py wraps in StmtBlock; unwrap single-statement blocks
        if isinstance(node, pyast.StmtBlock) and len(node.stmts) == 1:
            node = node.stmts[0]
        return self.visit_stmt(node)

    # ---- eval_expr ----

    def eval_expr(self, node) -> Any:
        if isinstance(node, pyast.Literal):
            return node.value

        if isinstance(node, pyast.Id):
            return self._resolve_name(node.name)

        if isinstance(node, pyast.Attr):
            base = self.eval_expr(node.obj)
            return getattr(base, node.name)

        if isinstance(node, pyast.Call):
            callee = self.eval_expr(node.callee)
            args = [self.eval_expr(a) for a in node.args]
            kwargs = {}
            for k, v in zip(node.kwargs_keys, node.kwargs_values):
                kwargs[str(k)] = self.eval_expr(v)
            return callee(*args, **kwargs)

        if isinstance(node, pyast.Operation):
            operands = node.operands
            if len(operands) == 2:
                lhs = self.eval_expr(operands[0])
                rhs = self.eval_expr(operands[1])
                return _apply_binary_op(node.op, lhs, rhs)
            if len(operands) == 1:
                return _apply_unary_op(node.op, self.eval_expr(operands[0]))

        if isinstance(node, pyast.Index):
            base = self.eval_expr(node.obj)
            indices = [self.eval_expr(i) for i in node.idx]
            if len(indices) == 1:
                return base[indices[0]]
            return base[tuple(indices)]

        if isinstance(node, pyast.Tuple):
            return tuple(self.eval_expr(e) for e in node.elements)

        if isinstance(node, pyast.List):
            return [self.eval_expr(e) for e in node.elements]

        raise ParseError(f"Cannot evaluate {type(node).__name__}")

    def _resolve_name(self, name: str) -> Any:
        v = self.var_table.get(name)
        if v is not None:
            return v
        if name in self.lang_modules:
            return self.lang_modules[name]
        import builtins

        if hasattr(builtins, name):
            return getattr(builtins, name)
        raise ParseError(f"Undefined name: {name}")

    # ---- visit_stmt ----

    def visit_stmt(self, node) -> Any:
        if isinstance(node, pyast.Function):
            for dec in node.decorators:
                dec_val = self.eval_expr(dec)
                r = self._try_dispatch(dec_val, node)
                if r is not None:
                    return r
            raise ParseError("Function has no recognized decorator")

        if isinstance(node, pyast.Assign):
            if node.rhs is not None:
                rhs_val = self.eval_expr(node.rhs)
                r = self._try_dispatch(rhs_val, node)
                if r is not None:
                    return r
                # Default: bind to var_table
                name = node.lhs.name
                self.var_table.define(name, rhs_val)
                return rhs_val
            return None

        if isinstance(node, pyast.Return):
            if node.value is not None:
                return self.eval_expr(node.value)
            return None

        raise ParseError(f"Unhandled statement: {type(node).__name__}")

    def visit_body(self, stmts) -> list:
        results = []
        for stmt in stmts:
            result = self.visit_stmt(stmt)
            if result is not None:
                results.append(result)
        return results

    # ---- dispatch helpers ----

    def _try_dispatch(self, val, node) -> Any | None:
        if hasattr(val, "__ffi_text_parse__"):
            return val.__ffi_text_parse__(self, node)
        return None


# ---- Operator dispatch (Python semantics) ----


def _apply_binary_op(kind: int, lhs, rhs):
    K = pyast.OperationKind
    ops = {
        K.Add: lambda a, b: a + b,
        K.Sub: lambda a, b: a - b,
        K.Mult: lambda a, b: a * b,
        K.Div: lambda a, b: a / b,
        K.FloorDiv: lambda a, b: a // b,
        K.Mod: lambda a, b: a % b,
        K.Lt: lambda a, b: a < b,
        K.LtE: lambda a, b: a <= b,
        K.Gt: lambda a, b: a > b,
        K.GtE: lambda a, b: a >= b,
        K.Eq: lambda a, b: a == b,
        K.NotEq: lambda a, b: a != b,
    }
    if kind in ops:
        return ops[kind](lhs, rhs)
    raise ParseError(f"Unknown binary op kind: {kind}")


def _apply_unary_op(kind: int, operand):
    K = pyast.OperationKind
    ops = {
        K.USub: lambda a: -a,
        K.Invert: lambda a: ~a,
        K.Not: lambda a: not a,
    }
    if kind in ops:
        return ops[kind](operand)
    raise ParseError(f"Unknown unary op kind: {kind}")
