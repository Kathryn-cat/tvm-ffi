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
from typing import Any, Callable

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
        if isinstance(node, pyast.Class):
            return self.parse_class(parser, node)
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

    def parse_class(self, parser, node):
        raise ParseError(f"{type(self).__name__}: parse_class not implemented")

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

    The surface object can set ``create_var`` and ``make_assign``
    on the parser before calling ``visit_body``, so that body
    statements are handled with dialect-specific logic.
    """

    def __init__(self, lang_modules: dict[str, object] | None = None):
        self.var_table = VarTable()
        self.lang_modules = lang_modules or {}
        # Dialect callbacks — set by surface objects before visit_body
        self.create_var: Callable = lambda name, ann=None: name
        self.make_assign: Callable | None = None
        self.make_func: Callable | None = None
        self.make_for: Callable | None = None
        # SyntaxContext — scope-determined statement handlers
        self.make_store: Callable | None = None
        # SyntaxContext — scope-determined statement handlers
        self.handle_if: Callable | None = None
        self.handle_while: Callable | None = None
        self.handle_assert: Callable | None = None
        self.handle_return: Callable | None = None

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

        if isinstance(node, pyast.Slice):
            start = self.eval_expr(node.start) if node.start is not None else None
            stop = self.eval_expr(node.stop) if node.stop is not None else None
            step = self.eval_expr(node.step) if node.step is not None else None
            return slice(start, stop, step)

        if isinstance(node, pyast.Tuple):
            return tuple(self.eval_expr(e) for e in node.values)

        if isinstance(node, pyast.List):
            return [self.eval_expr(e) for e in node.values]

        if isinstance(node, pyast.Dict):
            return {
                self.eval_expr(k): self.eval_expr(v)
                for k, v in zip(node.keys, node.values)
            }

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
            # Bare def (no decorator matched) — use make_func if set
            if self.make_func is not None:
                return self._parse_bare_function(node)
            raise ParseError("Function has no recognized decorator")

        if isinstance(node, pyast.Class):
            for dec in node.decorators:
                dec_val = self.eval_expr(dec)
                r = self._try_dispatch(dec_val, node)
                if r is not None:
                    return r
            raise ParseError("Class has no recognized decorator")

        if isinstance(node, pyast.For):
            # Check if iter is range() call — handle specially to avoid
            # calling builtins.range with IR values
            if self.make_for is not None and self._is_range_call(node.rhs):
                return self._parse_range_for_from_ast(node)
            iter_val = self.eval_expr(node.rhs)
            r = self._try_dispatch(iter_val, node)
            if r is not None:
                return r
            if self.make_for is not None:
                return self._parse_range_for(node, iter_val)
            raise ParseError("For loop iter has no dispatch and no make_for")

        if isinstance(node, pyast.Assign):
            if node.rhs is not None:
                rhs_val = self.eval_expr(node.rhs)
                r = self._try_dispatch(rhs_val, node)
                if r is not None:
                    return r
                # Subscript store: A[i] = val
                if isinstance(node.lhs, pyast.Index):
                    return self._handle_subscript_store(node, rhs_val)
                return self._default_assign(node, rhs_val)
            return None

        if isinstance(node, pyast.With):
            ctx_val = self.eval_expr(node.rhs)
            r = self._try_dispatch(ctx_val, node)
            if r is not None:
                return r
            raise ParseError("With context has no dispatch")

        if isinstance(node, pyast.If):
            if self.handle_if is not None:
                return self.handle_if(self, node)
            raise ParseError("If statement but no handle_if set")

        if isinstance(node, pyast.While):
            if self.handle_while is not None:
                return self.handle_while(self, node)
            raise ParseError("While statement but no handle_while set")

        if isinstance(node, pyast.Assert):
            if self.handle_assert is not None:
                return self.handle_assert(self, node)
            raise ParseError("Assert statement but no handle_assert set")

        if isinstance(node, pyast.Return):
            if self.handle_return is not None:
                return self.handle_return(self, node)
            if node.value is not None:
                return self.eval_expr(node.value)
            return None

        if isinstance(node, pyast.ExprStmt):
            return self.eval_expr(node.expr)

        raise ParseError(f"Unhandled statement: {type(node).__name__}")

    def _default_assign(self, node, rhs_val) -> Any:
        """Handle assignment when RHS has no __ffi_text_parse__.

        If the surface object set ``make_assign``, use it to construct
        a dialect-specific IR assign node. Otherwise just bind to var_table.
        ``make_assign`` receives ``(parser, node, rhs_val)`` so it can
        access the full AST node and parser state.
        """
        if self.make_assign is not None:
            return self.make_assign(self, node, rhs_val)
        name = node.lhs.name
        self.var_table.define(name, rhs_val)
        return rhs_val

    def _handle_subscript_store(self, node, rhs_val) -> Any:
        """Handle A[i] = val."""
        target = self.eval_expr(node.lhs.obj)
        indices = [self.eval_expr(i) for i in node.lhs.idx]
        if self.make_store is not None:
            return self.make_store(target, rhs_val, indices)
        # Fallback: use __setitem__
        if len(indices) == 1:
            target[indices[0]] = rhs_val
        else:
            target[tuple(indices)] = rhs_val
        return None

    def _is_range_call(self, node) -> bool:
        """Check if node is Call(Id("range"), ...)."""
        return (
            isinstance(node, pyast.Call)
            and isinstance(node.callee, pyast.Id)
            and node.callee.name == "range"
        )

    def _parse_range_for_from_ast(self, node) -> Any:
        """Handle for-loop with range() by evaluating args individually."""
        call = node.rhs  # Call(Id("range"), args)
        args = [self.eval_expr(a) for a in call.args]
        if len(args) == 1:
            start, end, step = 0, args[0], 1
        elif len(args) == 2:
            start, end, step = args[0], args[1], 1
        elif len(args) == 3:
            start, end, step = args[0], args[1], args[2]
        else:
            raise ParseError(f"range() expects 1-3 args, got {len(args)}")
        # Pass parser so make_for can create loop var with correct dtype
        # and define it in var_table
        return self.make_for(
            parser=self, var_name=node.lhs.name,
            start=start, end=end, step=step,
            body_node=node.body,
        )

    def _parse_range_for(self, node, iter_val) -> Any:
        """Handle for-loop with Python range object (plain int args)."""
        if isinstance(iter_val, range):
            return self.make_for(
                parser=self, var_name=node.lhs.name,
                start=iter_val.start, end=iter_val.stop,
                step=iter_val.step, body_node=node.body,
            )
        raise ParseError(f"Expected range, got {type(iter_val)}")

    def _parse_bare_function(self, node) -> Any:
        """Handle bare def (no decorator) using make_func callback."""
        with self.var_table.frame():
            params = []
            for arg in node.args:
                name = arg.lhs.name
                var = self.create_var(name)
                self.var_table.define(name, var)
                params.append(var)
            body_stmts = self.visit_body(node.body)
            # Separate return from body
            ret = None
            stmts_only = []
            for i, stmt in enumerate(node.body):
                if isinstance(stmt, pyast.Return):
                    ret = body_stmts[i]
                else:
                    stmts_only.append(body_stmts[i])
            return self.make_func(
                name=node.name.name,
                params=params,
                body=stmts_only,
                ret=ret,
            )

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
        K.BitAnd: lambda a, b: a & b,
        K.BitOr: lambda a, b: a | b,
        K.BitXor: lambda a, b: a ^ b,
        K.LShift: lambda a, b: a << b,
        K.RShift: lambda a, b: a >> b,
    }
    if kind in ops:
        # If one side is a plain Python int/float and the other has dtype
        # (e.g. PrimExpr), convert the plain value to match. This prevents
        # Python reflected operators from flipping operand order
        # (e.g. int.__lt__ unknown → PrimExpr.__gt__ which flips LT→GT).
        if isinstance(lhs, (int, float)) and hasattr(rhs, "dtype"):
            try:
                import tvm.tirx

                if isinstance(lhs, float):
                    lhs = tvm.tirx.FloatImm(str(rhs.dtype), lhs)
                else:
                    lhs = tvm.tirx.IntImm(str(rhs.dtype), lhs)
            except Exception:
                pass
        elif isinstance(rhs, (int, float)) and hasattr(lhs, "dtype"):
            try:
                import tvm.tirx

                if isinstance(rhs, float):
                    rhs = tvm.tirx.FloatImm(str(lhs.dtype), rhs)
                else:
                    rhs = tvm.tirx.IntImm(str(lhs.dtype), rhs)
            except Exception:
                pass
        return ops[kind](lhs, rhs)
    # and / or cannot use Python operators on IR values
    # Try calling _logical_op_handler if set
    if kind == K.And:
        return _logical_and(lhs, rhs)
    if kind == K.Or:
        return _logical_or(lhs, rhs)
    raise ParseError(f"Unknown binary op kind: {kind}")


def _logical_and(a, b):
    """Handle `and` for both Python bools and IR values."""
    try:
        return a and b
    except Exception:
        pass
    # IR values: use tvm.tirx.And
    import tvm.tirx
    return tvm.tirx.And(a, b)


def _logical_or(a, b):
    """Handle `or` for both Python bools and IR values."""
    try:
        return a or b
    except Exception:
        pass
    import tvm.tirx
    return tvm.tirx.Or(a, b)


def _apply_unary_op(kind: int, operand):
    K = pyast.OperationKind
    ops = {
        K.USub: lambda a: -a,
        K.Invert: lambda a: ~a,
    }
    if kind in ops:
        return ops[kind](operand)
    if kind == K.Not:
        if isinstance(operand, bool):
            return not operand
        try:
            import tvm.tirx
            return tvm.tirx.Not(operand)
        except Exception:
            return not operand
    raise ParseError(f"Unknown unary op kind: {kind}")
