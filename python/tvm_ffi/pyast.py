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
"""Python-style AST node definitions, printer classes, and rendering utilities.

This module defines the abstract syntax tree (AST) used by the text printer to
represent Python-style source code. The hierarchy is:

* ``Node`` -- base class for all AST nodes, providing ``to_python()`` and
  ``print_python()`` methods as well as source-path tracking.
* ``Expr(Node)`` -- base class for expression nodes (literals, identifiers,
  attribute access, indexing, calls, operations, etc.). Supports Python
  operator overloading so that AST fragments can be composed with ``+``,
  ``-``, ``*``, comparison operators, and more.
* ``Stmt(Node)`` -- base class for statement nodes (assignments, control flow,
  function/class definitions, comments, etc.).

Concrete node types correspond closely to the Python AST: ``Literal``, ``Id``,
``Attr``, ``Index``, ``Call``, ``Operation``, ``Lambda``, ``Tuple``, ``List``,
``Dict``, ``Slice`` (expressions) and ``StmtBlock``, ``Assign``, ``If``,
``While``, ``For``, ``With``, ``ExprStmt``, ``Assert``, ``Return``,
``Function``, ``Class``, ``Comment``, ``DocString`` (statements).
"""

# ruff: noqa: D102
# tvm-ffi-stubgen(begin): import-section
# fmt: off
# isort: off
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from collections.abc import MutableMapping, MutableSequence
    from tvm_ffi import Object
    from tvm_ffi.access_path import AccessPath
    from typing import Any, Callable, ClassVar
# isort: on
# fmt: on
# tvm-ffi-stubgen(end)

import contextlib
import sys
from collections.abc import Callable, Generator, Iterator, Sequence
from typing import Any, TypeVar

from tvm_ffi import Object
from tvm_ffi.access_path import AccessPath
from tvm_ffi.dataclasses import c_class


@c_class("ffi.pyast.PrinterConfig", init=False)
class PrinterConfig(Object):
    """Configuration for the Python-style text printer.

    Controls formatting behavior such as indentation, line numbering, and
    how free variables and duplicate variable names are handled.

    Attributes
    ----------
    def_free_var
        Whether to automatically define free variables that
        appear in the output. Default ``True``.
    indent_spaces
        Number of spaces per indentation level. Default ``2``.
    print_line_numbers
        If greater than zero, prefix each output line with
        its line number. ``0`` disables line numbers (default).
    num_context_lines
        Number of context lines to show around underlined
        regions when ``path_to_underline`` is set. ``-1`` means show all
        lines (default).
    print_addr_on_dup_var
        When ``True``, append an object address suffix
        to disambiguate variables that share the same name. Default
        ``False``.
    path_to_underline
        A list of ``AccessPath`` instances identifying
        sub-expressions to underline in the printed output. Default
        is an empty list.

    Examples
    --------
    .. code-block:: python

        from tvm_ffi import pyast

        cfg = ast.PrinterConfig(indent_spaces=4, print_line_numbers=1)
        node = ast.Id(name="x")
        print(node.to_python(cfg))

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.PrinterConfig
    # fmt: off
    def_free_var: bool
    indent_spaces: int
    print_line_numbers: int
    num_context_lines: int
    print_addr_on_dup_var: bool
    path_to_underline: MutableSequence[AccessPath]
    if TYPE_CHECKING:
        def __init__(self, def_free_var: bool, indent_spaces: int, print_line_numbers: int, num_context_lines: int, print_addr_on_dup_var: bool, path_to_underline: MutableSequence[AccessPath]) -> None: ...
        def __ffi_init__(self, _0: bool, _1: int, _2: int, _3: int, _4: bool, _5: MutableSequence[AccessPath], /) -> None: ...  # ty: ignore[invalid-method-override]
    # fmt: on
    # tvm-ffi-stubgen(end)

    def __init__(
        self,
        def_free_var: bool = True,
        indent_spaces: int = 2,
        print_line_numbers: int = 0,
        num_context_lines: int = -1,
        print_addr_on_dup_var: bool = False,
        path_to_underline: list[AccessPath] | None = None,
    ) -> None:
        """Initialize a PrinterConfig.

        Parameters
        ----------
        def_free_var
            Whether to automatically define free variables. Default ``True``.
        indent_spaces
            Number of spaces per indentation level. Default ``2``.
        print_line_numbers
            If greater than zero, prefix each output line with its line
            number. Default ``0``.
        num_context_lines
            Number of context lines to show around underlined regions.
            ``-1`` means show all lines. Default ``-1``.
        print_addr_on_dup_var
            When ``True``, append an object address suffix to disambiguate
            variables that share the same name. Default ``False``.
        path_to_underline
            A list of ``AccessPath`` instances identifying sub-expressions
            to underline. Default ``None`` (empty list).

        """
        if path_to_underline is None:
            path_to_underline = []
        self.__ffi_init__(
            def_free_var,
            indent_spaces,
            print_line_numbers,
            num_context_lines,
            print_addr_on_dup_var,
            path_to_underline,
        )


@c_class("ffi.pyast.Node", init=False)
class Node(Object):
    """Base class for all text-printer AST nodes.

    Every AST node carries an optional list of ``source_paths`` that trace
    the node back to the original IR object it was derived from. The two
    main entry points for rendering are ``to_python()`` (returns a string)
    and ``print_python()`` (prints to stdout).

    Attributes
    ----------
    source_paths
        Access paths linking this node to the original IR
        objects it represents. Default is an empty list.

    Examples
    --------
    .. code-block:: python

        from tvm_ffi import pyast

        node = ast.Id(name="x")
        source = node.to_python()  # "x"
        node.print_python()  # prints "x" to stdout

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.Node
    # fmt: off
    source_paths: MutableSequence[AccessPath]
    lineno: int
    col_offset: int
    end_lineno: int
    end_col_offset: int
    if TYPE_CHECKING:
        def _to_python(self, _1: PrinterConfig, /) -> str: ...
    # fmt: on
    # tvm-ffi-stubgen(end)

    def to_python(self, config: PrinterConfig | None = None) -> str:
        """Render this AST node as Python-style source code.

        Parameters
        ----------
        config
            Printer configuration. Uses default settings when ``None``.

        Returns
        -------
        source
            The rendered source code as a string.

        Examples
        --------
        .. code-block:: python

            from tvm_ffi import pyast

            node = ast.Id(name="my_var")
            node.to_python()  # "my_var"

        """
        if config is None:
            config = PrinterConfig()
        return self._to_python(config)

    def print_python(
        self,
        config: PrinterConfig | None = None,
        style: str | None = None,
    ) -> None:
        """Print this AST node as Python-style source code to stdout.

        Uses Pygments syntax highlighting when available.

        Parameters
        ----------
        config
            Printer configuration. Uses default settings when ``None``.
        style
            Pygments style name or one of ``"light"``, ``"dark"``,
            ``"ansi"``. Defaults to ``"light"`` in notebooks, ``"ansi"``
            in terminals.

        """
        from ._pyast_colored_print import cprint  # noqa: PLC0415

        cprint(self.to_python(config), style=style)

    def add_path(self, path: Any) -> Node:
        """Append a source path to this node and return the node itself.

        This allows chaining, e.g. ``node.add_path(p1).add_path(p2)``.

        Parameters
        ----------
        path
            The access path to append.

        Returns
        -------
        self
            ``self``, for fluent chaining.

        """
        self.source_paths.append(path)
        return self


@c_class("ffi.pyast.Expr", init=False)
class Expr(Node):
    """Base class for expression AST nodes.

    ``Expr`` extends ``Node`` with Python operator overloading and builder
    methods so that AST fragments can be composed using natural syntax:

    * **Arithmetic**: ``+``, ``-``, ``*``, ``/``, ``//``, ``%``, ``**``
    * **Bitwise**: ``&``, ``|``, ``^``, ``~``, ``<<``, ``>>``
    * **Comparison**: ``<``, ``<=``, ``>``, ``>=``, ``.eq()``, ``.ne()``
    * **Logical**: ``.logical_and()``, ``.logical_or()``
    * **Ternary**: ``.if_then_else(then, else_)``
    * **Access**: ``.attr("name")``, ``.index([...])``, ``[...]``
    * **Calls**: ``.call(*args)``, ``.call_kw(args, keys, values)``

    Examples
    --------
    .. code-block:: python

        from tvm_ffi import pyast

        x = ast.Id(name="x")
        y = ast.Id(name="y")
        expr = (x + y).attr("shape").call(ast.Literal(0))
        expr.print_python()  # (x + y).shape(0)

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.Expr
    # fmt: off
    source_paths: MutableSequence[AccessPath]
    # fmt: on
    # tvm-ffi-stubgen(end)

    def __neg__(self) -> Expr:
        return Operation(OperationKind.USub, [self])

    def __invert__(self) -> Expr:
        return Operation(OperationKind.Invert, [self])

    def __add__(self, other: Expr) -> Expr:
        return Operation(OperationKind.Add, [self, other])

    def __sub__(self, other: Expr) -> Expr:
        return Operation(OperationKind.Sub, [self, other])

    def __mul__(self, other: Expr) -> Expr:
        return Operation(OperationKind.Mult, [self, other])

    def __truediv__(self, other: Expr) -> Expr:
        return Operation(OperationKind.Div, [self, other])

    def __floordiv__(self, other: Expr) -> Expr:
        return Operation(OperationKind.FloorDiv, [self, other])

    def __mod__(self, other: Expr) -> Expr:
        return Operation(OperationKind.Mod, [self, other])

    def __pow__(self, other: Expr) -> Expr:
        return Operation(OperationKind.Pow, [self, other])

    def __lshift__(self, other: Expr) -> Expr:
        return Operation(OperationKind.LShift, [self, other])

    def __rshift__(self, other: Expr) -> Expr:
        return Operation(OperationKind.RShift, [self, other])

    def __and__(self, other: Expr) -> Expr:
        return Operation(OperationKind.BitAnd, [self, other])

    def __or__(self, other: Expr) -> Expr:
        return Operation(OperationKind.BitOr, [self, other])

    def __xor__(self, other: Expr) -> Expr:
        return Operation(OperationKind.BitXor, [self, other])

    def __lt__(self, other: Expr) -> Expr:
        return Operation(OperationKind.Lt, [self, other])

    def __le__(self, other: Expr) -> Expr:
        return Operation(OperationKind.LtE, [self, other])

    def __gt__(self, other: Expr) -> Expr:
        return Operation(OperationKind.Gt, [self, other])

    def __ge__(self, other: Expr) -> Expr:
        return Operation(OperationKind.GtE, [self, other])

    def logical_and(self, other: Expr) -> Expr:
        """Build a logical ``and`` operation (``self and other``).

        Python's ``and`` operator cannot be overloaded, so this explicit
        method is provided instead.

        Parameters
        ----------
        other
            The right-hand operand.

        Returns
        -------
        result
            An ``Operation`` node representing ``self and other``.

        """
        return Operation(OperationKind.And, [self, other])

    def logical_or(self, other: Expr) -> Expr:
        """Build a logical ``or`` operation (``self or other``).

        Python's ``or`` operator cannot be overloaded, so this explicit
        method is provided instead.

        Parameters
        ----------
        other
            The right-hand operand.

        Returns
        -------
        result
            An ``Operation`` node representing ``self or other``.

        """
        return Operation(OperationKind.Or, [self, other])

    def if_then_else(self, then: Expr, else_: Expr) -> Expr:
        """Build a ternary conditional expression (``then if self else else_``).

        Parameters
        ----------
        then
            The value when the condition is true.
        else_
            The value when the condition is false.

        Returns
        -------
        result
            An ``Operation`` node representing the ternary expression.

        Examples
        --------
        .. code-block:: python

            from tvm_ffi import pyast

            cond = ast.Id(name="flag")
            result = cond.if_then_else(ast.Literal(1), ast.Literal(0))
            result.print_python()  # 1 if flag else 0

        """
        return Operation(OperationKind.IfThenElse, [self, then, else_])

    def eq(self, other: Expr) -> Expr:
        """Build an equality comparison (``self == other``).

        Python's ``__eq__`` is not overloaded to preserve standard object
        identity semantics, so this explicit method is provided instead.

        Parameters
        ----------
        other
            The right-hand operand.

        Returns
        -------
        result
            An ``Operation`` node representing ``self == other``.

        """
        return Operation(OperationKind.Eq, [self, other])

    def ne(self, other: Expr) -> Expr:
        """Build an inequality comparison (``self != other``).

        Python's ``__ne__`` is not overloaded to preserve standard object
        identity semantics, so this explicit method is provided instead.

        Parameters
        ----------
        other
            The right-hand operand.

        Returns
        -------
        result
            An ``Operation`` node representing ``self != other``.

        """
        return Operation(OperationKind.NotEq, [self, other])

    def attr(self, name: str) -> Expr:
        """Build an attribute access expression (``self.name``).

        Parameters
        ----------
        name
            The attribute name.

        Returns
        -------
        result
            An ``Attr`` node representing ``self.name``.

        Examples
        --------
        .. code-block:: python

            from tvm_ffi import pyast

            obj = ast.Id(name="module")
            obj.attr("forward").print_python()  # module.forward

        """
        return Attr(self, name)

    def index(self, indices: MutableSequence[Expr]) -> Expr:
        """Build a subscript/index expression (``self[indices]``).

        Parameters
        ----------
        indices
            A sequence of index expressions.

        Returns
        -------
        result
            An ``Index`` node representing ``self[indices]``.

        Examples
        --------
        .. code-block:: python

            from tvm_ffi import pyast

            arr = ast.Id(name="arr")
            arr.index([ast.Literal(0)]).print_python()  # arr[0]

        """
        return Index(self, indices)

    def call(self, *args: Expr) -> Expr:
        """Build a positional-only call expression (``self(args...)``).

        Parameters
        ----------
        *args
            Positional argument expressions.

        Returns
        -------
        result
            A ``Call`` node representing ``self(*args)``.

        Examples
        --------
        .. code-block:: python

            from tvm_ffi import pyast

            fn = ast.Id(name="relu")
            fn.call(ast.Id(name="x")).print_python()  # relu(x)

        """
        return Call(
            self,
            args,  # ty: ignore[invalid-argument-type]
            [],
            [],
        )

    def call_kw(
        self,
        args: Sequence[Expr],
        kwargs_keys: Sequence[str],
        kwargs_values: Sequence[Expr],
    ) -> Expr:
        """Build a call expression with keyword arguments.

        Renders as ``self(*args, key0=val0, key1=val1, ...)``.

        Parameters
        ----------
        args
            Positional argument expressions.
        kwargs_keys
            Keyword argument names.
        kwargs_values
            Keyword argument value expressions, in the same
            order as *kwargs_keys*.

        Returns
        -------
        result
            A ``Call`` node representing the keyword call.

        Examples
        --------
        .. code-block:: python

            from tvm_ffi import pyast

            fn = ast.Id(name="conv2d")
            fn.call_kw(
                args=[ast.Id(name="x")],
                kwargs_keys=["stride"],
                kwargs_values=[ast.Literal(2)],
            ).print_python()  # conv2d(x, stride=2)

        """
        if not isinstance(args, Sequence):
            args = (args,)
        if not isinstance(kwargs_keys, Sequence):
            kwargs_keys = (kwargs_keys,)
        if not isinstance(kwargs_values, Sequence):
            kwargs_values = (kwargs_values,)
        return Call(
            self,
            args,  # ty: ignore[invalid-argument-type]
            kwargs_keys,  # ty: ignore[invalid-argument-type]
            kwargs_values,  # ty: ignore[invalid-argument-type]
        )

    def __getitem__(self, indices: Expr | Sequence[Expr]) -> Expr:
        """Build a subscript expression via Python's ``[]`` syntax.

        Delegates to ``self.index()``, wrapping a single index in a tuple
        if necessary.

        Parameters
        ----------
        indices
            One or more index expressions.

        Returns
        -------
        result
            An ``Index`` node representing ``self[indices]``.

        """
        if isinstance(indices, Sequence):
            return self.index(indices)  # ty: ignore[invalid-argument-type]
        return self.index([indices])


@c_class("ffi.pyast.Stmt", init=False)
class Stmt(Node):
    """Base class for statement AST nodes.

    Statements represent executable constructs (assignments, loops,
    conditionals, etc.). Every statement may carry an optional trailing
    ``comment`` that is rendered as ``# comment`` after the statement.

    Attributes
    ----------
    comment
        An optional inline comment string. Default ``None``.

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.Stmt
    # fmt: off
    source_paths: MutableSequence[AccessPath]
    comment: str | None
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("ffi.pyast.StmtBlock")
class StmtBlock(Stmt):
    """A sequence of statements rendered as a block.

    Represents a group of statements that are printed together, typically
    as the body of a function, class, loop, or conditional.

    Attributes
    ----------
    stmts
        The list of statements in this block.

    Examples
    --------
    .. code-block:: python

        from tvm_ffi import pyast

        block = ast.StmtBlock(stmts=[ast.ExprStmt(expr=ast.Id(name="x"))])
        block.print_python()  # x

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.StmtBlock
    # fmt: off
    stmts: MutableSequence[Stmt]
    if TYPE_CHECKING:
        def __init__(self, stmts: MutableSequence[Stmt], *, comment: str | None = ...) -> None: ...
        def __ffi_init__(self, stmts: MutableSequence[Stmt], *, comment: str | None = ...) -> None: ...  # ty: ignore[invalid-method-override]
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("ffi.pyast.Literal")
class Literal(Expr):
    """A literal value expression (``42``, ``3.14``, ``"hello"``, ``True``, ``None``).

    Wraps an arbitrary Python value and renders it using its ``repr()``.

    Attributes
    ----------
    value
        The literal Python value.

    Examples
    --------
    .. code-block:: python

        from tvm_ffi import pyast

        ast.Literal(42).print_python()  # 42
        ast.Literal("hello").print_python()  # "hello"

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.Literal
    # fmt: off
    value: Any
    kind: str | None
    if TYPE_CHECKING:
        def __init__(self, value: Any, kind: str | None = ...) -> None: ...
        def __ffi_init__(self, value: Any, kind: str | None = ...) -> None: ...  # ty: ignore[invalid-method-override]
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("ffi.pyast.Id")
class Id(Expr):
    """An identifier / variable name expression.

    Renders as the bare name string.

    Attributes
    ----------
    name
        The identifier string.

    Examples
    --------
    .. code-block:: python

        from tvm_ffi import pyast

        ast.Id(name="x").print_python()  # x

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.Id
    # fmt: off
    name: str
    if TYPE_CHECKING:
        def __init__(self, name: str) -> None: ...
        def __ffi_init__(self, name: str) -> None: ...  # ty: ignore[invalid-method-override]
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("ffi.pyast.Attr")
class Attr(Expr):
    """An attribute access expression (``obj.name``).

    Attributes
    ----------
    obj
        The object expression being accessed.
    name
        The attribute name.

    Examples
    --------
    .. code-block:: python

        from tvm_ffi import pyast

        ast.Attr(obj=ast.Id(name="self"), name="weight").print_python()  # self.weight

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.Attr
    # fmt: off
    obj: Expr
    name: str
    if TYPE_CHECKING:
        def __init__(self, obj: Expr, name: str) -> None: ...
        def __ffi_init__(self, obj: Expr, name: str) -> None: ...  # ty: ignore[invalid-method-override]
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("ffi.pyast.Index")
class Index(Expr):
    """A subscript / index expression (``obj[idx0, idx1, ...]``).

    Attributes
    ----------
    obj
        The object expression being indexed.
    idx
        A list of index expressions.

    Examples
    --------
    .. code-block:: python

        from tvm_ffi import pyast

        ast.Index(obj=ast.Id(name="x"), idx=[ast.Literal(0)]).print_python()  # x[0]

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.Index
    # fmt: off
    obj: Expr
    idx: MutableSequence[Expr]
    if TYPE_CHECKING:
        def __init__(self, obj: Expr, idx: MutableSequence[Expr]) -> None: ...
        def __ffi_init__(self, obj: Expr, idx: MutableSequence[Expr]) -> None: ...  # ty: ignore[invalid-method-override]
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("ffi.pyast.Call")
class Call(Expr):
    """A function call expression (``callee(args..., key=val, ...)``).

    Attributes
    ----------
    callee
        The callable expression.
    args
        Positional argument expressions.
    kwargs_keys
        Keyword argument names.
    kwargs_values
        Keyword argument value expressions, aligned with
        *kwargs_keys*.

    Examples
    --------
    .. code-block:: python

        from tvm_ffi import pyast

        ast.Call(
            callee=ast.Id(name="f"),
            args=[ast.Literal(1)],
            kwargs_keys=["dim"],
            kwargs_values=[ast.Literal(0)],
        ).print_python()  # f(1, dim=0)

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.Call
    # fmt: off
    callee: Expr
    args: MutableSequence[Expr]
    kwargs_keys: MutableSequence[str]
    kwargs_values: MutableSequence[Expr]
    if TYPE_CHECKING:
        def __init__(self, callee: Expr, args: MutableSequence[Expr], kwargs_keys: MutableSequence[str], kwargs_values: MutableSequence[Expr]) -> None: ...
        def __ffi_init__(self, callee: Expr, args: MutableSequence[Expr], kwargs_keys: MutableSequence[str], kwargs_values: MutableSequence[Expr]) -> None: ...  # ty: ignore[invalid-method-override]
    # fmt: on
    # tvm-ffi-stubgen(end)


class OperationKind:
    """Enum-like class defining operation kinds for ``Operation`` nodes.

    Integer constants are grouped into three ranges:

    * **Unary** (``_UnaryStart`` .. ``_UnaryEnd``): ``USub`` (``-x``),
      ``Invert`` (``~x``), ``Not`` (``not x``).
    * **Binary** (``_BinaryStart`` .. ``_BinaryEnd``): arithmetic
      (``Add``, ``Sub``, ``Mult``, ``Div``, ``FloorDiv``, ``Mod``,
      ``Pow``), bitwise (``LShift``, ``RShift``, ``BitAnd``, ``BitOr``,
      ``BitXor``), comparison (``Lt``, ``LtE``, ``Eq``, ``NotEq``,
      ``Gt``, ``GtE``), and logical (``And``, ``Or``).
    * **Special** (``_SpecialStart`` .. ``SpecialEnd``): ``IfThenElse``
      (ternary conditional ``a if cond else b``).

    These constants are used as the ``op`` field of ``Operation`` nodes.

    Examples
    --------
    .. code-block:: python

        from tvm_ffi import pyast

        x = ast.Id(name="x")
        y = ast.Id(name="y")
        add_op = ast.Operation(ast.OperationKind.Add, [x, y])
        add_op.print_python()  # x + y

    """

    Undefined = -1
    _UnaryStart = 0
    USub = 1
    Invert = 2
    Not = 3
    UAdd = 4
    _UnaryEnd = 5
    _BinaryStart = 5
    Add = 6
    Sub = 7
    Mult = 8
    Div = 9
    FloorDiv = 10
    Mod = 11
    Pow = 12
    LShift = 13
    RShift = 14
    BitAnd = 15
    BitOr = 16
    BitXor = 17
    Lt = 18
    LtE = 19
    Eq = 20
    NotEq = 21
    Gt = 22
    GtE = 23
    And = 24
    Or = 25
    MatMult = 26
    Is = 27
    IsNot = 28
    In = 29
    NotIn = 30
    _BinaryEnd = 31
    _SpecialStart = 32
    IfThenElse = 33
    ChainedCompare = 34
    Parens = 35
    SpecialEnd = 36


@c_class("ffi.pyast.Operation")
class Operation(Expr):
    """A unary, binary, or special operation expression.

    The ``op`` field is one of the integer constants defined in
    ``OperationKind``. The ``operands`` list contains one element for
    unary ops, two for binary ops, or three for ``IfThenElse``.

    Attributes
    ----------
    op
        The operation kind (an ``OperationKind`` constant).
    operands
        The operand expressions.

    Examples
    --------
    .. code-block:: python

        from tvm_ffi import pyast

        x = ast.Id(name="x")
        y = ast.Id(name="y")
        expr = ast.Operation(ast.OperationKind.Add, [x, y])
        expr.print_python()  # x + y

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.Operation
    # fmt: off
    op: int
    operands: MutableSequence[Expr]
    if TYPE_CHECKING:
        def __init__(self, op: int, operands: MutableSequence[Expr]) -> None: ...
        def __ffi_init__(self, op: int, operands: MutableSequence[Expr]) -> None: ...  # ty: ignore[invalid-method-override]
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("ffi.pyast.Lambda")
class Lambda(Expr):
    """A lambda expression (``lambda args: body``).

    Attributes
    ----------
    args
        The parameter identifiers.
    body
        The body expression.

    Examples
    --------
    .. code-block:: python

        from tvm_ffi import pyast

        ast.Lambda(args=[ast.Id(name="x")], body=ast.Id(name="x")).print_python()
        # lambda x: x

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.Lambda
    # fmt: off
    args: MutableSequence[Expr]
    body: Expr
    if TYPE_CHECKING:
        def __init__(self, args: MutableSequence[Expr], body: Expr) -> None: ...
        def __ffi_init__(self, args: MutableSequence[Expr], body: Expr) -> None: ...  # ty: ignore[invalid-method-override]
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("ffi.pyast.Tuple")
class Tuple(Expr):
    """A tuple expression (``(a, b, c)``).

    Attributes
    ----------
    values
        The element expressions.

    Examples
    --------
    .. code-block:: python

        from tvm_ffi import pyast

        ast.Tuple(values=[ast.Literal(1), ast.Literal(2)]).print_python()  # (1, 2)

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.Tuple
    # fmt: off
    values: MutableSequence[Expr]
    if TYPE_CHECKING:
        def __init__(self, values: MutableSequence[Expr]) -> None: ...
        def __ffi_init__(self, values: MutableSequence[Expr]) -> None: ...  # ty: ignore[invalid-method-override]
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("ffi.pyast.List")
class List(Expr):
    """A list expression (``[a, b, c]``).

    Attributes
    ----------
    values
        The element expressions.

    Examples
    --------
    .. code-block:: python

        from tvm_ffi import pyast

        ast.List(values=[ast.Literal(1), ast.Literal(2)]).print_python()  # [1, 2]

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.List
    # fmt: off
    values: MutableSequence[Expr]
    if TYPE_CHECKING:
        def __init__(self, values: MutableSequence[Expr]) -> None: ...
        def __ffi_init__(self, values: MutableSequence[Expr]) -> None: ...  # ty: ignore[invalid-method-override]
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("ffi.pyast.Dict")
class Dict(Expr):
    """A dictionary expression (``{k0: v0, k1: v1, ...}``).

    Attributes
    ----------
    keys
        The key expressions.
    values
        The value expressions, aligned with *keys*.

    Examples
    --------
    .. code-block:: python

        from tvm_ffi import pyast

        ast.Dict(
            keys=[ast.Literal("a")],
            values=[ast.Literal(1)],
        ).print_python()  # {"a": 1}

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.Dict
    # fmt: off
    keys: MutableSequence[Expr]
    values: MutableSequence[Expr]
    if TYPE_CHECKING:
        def __init__(self, keys: MutableSequence[Expr], values: MutableSequence[Expr]) -> None: ...
        def __ffi_init__(self, keys: MutableSequence[Expr], values: MutableSequence[Expr]) -> None: ...  # ty: ignore[invalid-method-override]
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("ffi.pyast.Slice")
class Slice(Expr):
    """A slice expression (``start:stop:step``).

    All three components are optional. A ``None`` component is omitted
    from the rendered output.

    Attributes
    ----------
    start
        The start expression, or ``None``.
    stop
        The stop expression, or ``None``.
    step
        The step expression, or ``None``.

    Examples
    --------
    .. code-block:: python

        from tvm_ffi import pyast

        ast.Slice(start=ast.Literal(0), stop=ast.Literal(10)).print_python()  # 0:10

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.Slice
    # fmt: off
    start: Expr | None
    stop: Expr | None
    step: Expr | None
    if TYPE_CHECKING:
        def __init__(self, start: Expr | None = ..., stop: Expr | None = ..., step: Expr | None = ...) -> None: ...
        def __ffi_init__(self, start: Expr | None = ..., stop: Expr | None = ..., step: Expr | None = ...) -> None: ...  # ty: ignore[invalid-method-override]
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("ffi.pyast.Assign")
class Assign(Stmt):
    """An assignment statement (``lhs = rhs`` or ``lhs: annotation = rhs``).

    When ``rhs`` is ``None`` the statement renders as a bare declaration
    (``lhs: annotation``). When ``annotation`` is ``None`` the type
    annotation is omitted.

    Attributes
    ----------
    lhs
        The left-hand-side target expression.
    rhs
        The right-hand-side value expression, or ``None``.
    annotation
        An optional type annotation expression.

    Examples
    --------
    .. code-block:: python

        from tvm_ffi import pyast

        ast.Assign(lhs=ast.Id(name="x"), rhs=ast.Literal(42)).print_python()  # x = 42

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.Assign
    # fmt: off
    lhs: Expr
    rhs: Expr | None
    annotation: Expr | None
    aug_op: int
    if TYPE_CHECKING:
        def __init__(self, lhs: Expr, rhs: Expr | None = ..., annotation: Expr | None = ..., aug_op: int = ..., *, comment: str | None = ...) -> None: ...
        def __ffi_init__(self, lhs: Expr, rhs: Expr | None = ..., annotation: Expr | None = ..., aug_op: int = ..., *, comment: str | None = ...) -> None: ...  # ty: ignore[invalid-method-override]
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("ffi.pyast.If")
class If(Stmt):
    """An ``if / elif / else`` conditional statement.

    Attributes
    ----------
    cond
        The condition expression.
    then_branch
        Statements executed when the condition is true.
    else_branch
        Statements executed when the condition is false
        (may be empty).

    Examples
    --------
    .. code-block:: python

        from tvm_ffi import pyast

        ast.If(
            cond=ast.Id(name="flag"),
            then_branch=[ast.ExprStmt(expr=ast.Id(name="a"))],
            else_branch=[ast.ExprStmt(expr=ast.Id(name="b"))],
        ).print_python()
        # if flag:
        #   a
        # else:
        #   b

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.If
    # fmt: off
    cond: Expr
    then_branch: MutableSequence[Stmt]
    else_branch: MutableSequence[Stmt]
    if TYPE_CHECKING:
        def __init__(self, cond: Expr, then_branch: MutableSequence[Stmt], else_branch: MutableSequence[Stmt], *, comment: str | None = ...) -> None: ...
        def __ffi_init__(self, cond: Expr, then_branch: MutableSequence[Stmt], else_branch: MutableSequence[Stmt], *, comment: str | None = ...) -> None: ...  # ty: ignore[invalid-method-override]
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("ffi.pyast.While")
class While(Stmt):
    """A ``while`` loop statement.

    Attributes
    ----------
    cond
        The loop condition expression.
    body
        The loop body statements.

    Examples
    --------
    .. code-block:: python

        from tvm_ffi import pyast

        ast.While(
            cond=ast.Id(name="running"),
            body=[ast.ExprStmt(expr=ast.Id(name="step"))],
        ).print_python()
        # while running:
        #   step

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.While
    # fmt: off
    cond: Expr
    body: MutableSequence[Stmt]
    orelse: MutableSequence[Stmt]
    if TYPE_CHECKING:
        def __init__(self, cond: Expr, body: MutableSequence[Stmt], orelse: MutableSequence[Stmt] = ..., *, comment: str | None = ...) -> None: ...
        def __ffi_init__(self, cond: Expr, body: MutableSequence[Stmt], orelse: MutableSequence[Stmt] = ..., *, comment: str | None = ...) -> None: ...  # ty: ignore[invalid-method-override]
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("ffi.pyast.For")
class For(Stmt):
    """A ``for`` loop statement (``for lhs in rhs: body``).

    Attributes
    ----------
    lhs
        The loop variable expression.
    rhs
        The iterable expression.
    body
        The loop body statements.

    Examples
    --------
    .. code-block:: python

        from tvm_ffi import pyast

        ast.For(
            lhs=ast.Id(name="i"),
            rhs=ast.Id(name="items"),
            body=[ast.ExprStmt(expr=ast.Id(name="process"))],
        ).print_python()
        # for i in items:
        #   process

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.For
    # fmt: off
    lhs: Expr
    rhs: Expr
    body: MutableSequence[Stmt]
    is_async: bool
    orelse: MutableSequence[Stmt]
    if TYPE_CHECKING:
        def __init__(self, lhs: Expr, rhs: Expr, body: MutableSequence[Stmt], is_async: bool = ..., orelse: MutableSequence[Stmt] = ..., *, comment: str | None = ...) -> None: ...
        def __ffi_init__(self, lhs: Expr, rhs: Expr, body: MutableSequence[Stmt], is_async: bool = ..., orelse: MutableSequence[Stmt] = ..., *, comment: str | None = ...) -> None: ...  # ty: ignore[invalid-method-override]
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("ffi.pyast.With")
class With(Stmt):
    """A ``with`` context-manager statement (``with rhs as lhs: body``).

    When ``lhs`` is ``None``, the ``as lhs`` clause is omitted.

    Attributes
    ----------
    lhs
        The optional target expression (``as`` variable), or ``None``.
    rhs
        The context-manager expression.
    body
        The body statements.

    Examples
    --------
    .. code-block:: python

        from tvm_ffi import pyast

        ast.With(
            lhs=ast.Id(name="f"),
            rhs=ast.Id(name="open_file"),
            body=[ast.ExprStmt(expr=ast.Id(name="read"))],
        ).print_python()
        # with open_file as f:
        #   read

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.With
    # fmt: off
    lhs: Expr | None
    rhs: Expr
    body: MutableSequence[Stmt]
    is_async: bool
    if TYPE_CHECKING:
        def __init__(self, lhs: Expr | None, rhs: Expr, body: MutableSequence[Stmt], is_async: bool = ..., *, comment: str | None = ...) -> None: ...
        def __ffi_init__(self, lhs: Expr | None, rhs: Expr, body: MutableSequence[Stmt], is_async: bool = ..., *, comment: str | None = ...) -> None: ...  # ty: ignore[invalid-method-override]
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("ffi.pyast.ExprStmt")
class ExprStmt(Stmt):
    """An expression used as a statement (e.g. a bare function call).

    Attributes
    ----------
    expr
        The expression to evaluate as a statement.

    Examples
    --------
    .. code-block:: python

        from tvm_ffi import pyast

        ast.ExprStmt(expr=ast.Id(name="do_something")).print_python()  # do_something

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.ExprStmt
    # fmt: off
    expr: Expr
    if TYPE_CHECKING:
        def __init__(self, expr: Expr, *, comment: str | None = ...) -> None: ...
        def __ffi_init__(self, expr: Expr, *, comment: str | None = ...) -> None: ...  # ty: ignore[invalid-method-override]
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("ffi.pyast.Assert")
class Assert(Stmt):
    """An ``assert`` statement (``assert cond, msg``).

    When ``msg`` is ``None``, only the condition is rendered.

    Attributes
    ----------
    cond
        The condition expression.
    msg
        An optional message expression, or ``None``.

    Examples
    --------
    .. code-block:: python

        from tvm_ffi import pyast

        ast.Assert(
            cond=ast.Id(name="x"), msg=ast.Literal("x must be set")
        ).print_python()
        # assert x, "x must be set"

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.Assert
    # fmt: off
    cond: Expr
    msg: Expr | None
    if TYPE_CHECKING:
        def __init__(self, cond: Expr, msg: Expr | None = ..., *, comment: str | None = ...) -> None: ...
        def __ffi_init__(self, cond: Expr, msg: Expr | None = ..., *, comment: str | None = ...) -> None: ...  # ty: ignore[invalid-method-override]
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("ffi.pyast.Return")
class Return(Stmt):
    """A ``return`` statement.

    When ``value`` is ``None``, renders as a bare ``return``.

    Attributes
    ----------
    value
        The return value expression, or ``None``.

    Examples
    --------
    .. code-block:: python

        from tvm_ffi import pyast

        ast.Return(value=ast.Literal(42)).print_python()  # return 42

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.Return
    # fmt: off
    value: Expr | None
    if TYPE_CHECKING:
        def __init__(self, value: Expr | None = ..., *, comment: str | None = ...) -> None: ...
        def __ffi_init__(self, value: Expr | None = ..., *, comment: str | None = ...) -> None: ...  # ty: ignore[invalid-method-override]
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("ffi.pyast.Function", init=False)
class Function(Stmt):
    """A ``def`` function definition statement.

    Attributes
    ----------
    name
        The function name identifier.
    args
        The parameter list, each represented as an ``Assign`` node
        (the ``lhs`` is the parameter name; ``annotation`` and ``rhs``
        provide type hints and default values).
    decorators
        Decorator expressions applied above the function.
    return_type
        An optional return-type annotation expression.
    body
        The function body statements.

    Examples
    --------
    .. code-block:: python

        from tvm_ffi import pyast

        ast.Function(
            name=ast.Id(name="add"),
            args=[
                ast.Assign(lhs=ast.Id(name="a")),
                ast.Assign(lhs=ast.Id(name="b")),
            ],
            decorators=[],
            return_type=None,
            body=[ast.Return(value=ast.Id(name="a") + ast.Id(name="b"))],
        ).print_python()
        # def add(a, b):
        #   return a + b

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.Function
    # fmt: off
    name: Id
    args: MutableSequence[Assign]
    decorators: MutableSequence[Expr]
    return_type: Expr | None
    body: MutableSequence[Stmt]
    is_async: bool
    if TYPE_CHECKING:
        def __init__(self, name: Id, args: MutableSequence[Assign], decorators: MutableSequence[Expr], return_type: Expr | None, body: MutableSequence[Stmt], is_async: bool = ..., *, comment: str | None = ...) -> None: ...
        def __ffi_init__(self, _0: Id, _1: MutableSequence[Assign], _2: MutableSequence[Expr], _3: Expr | None, _4: MutableSequence[Stmt], _5: bool, /) -> None: ...  # ty: ignore[invalid-method-override]
    # fmt: on
    # tvm-ffi-stubgen(end)

    def __init__(
        self,
        name: Id,
        args: MutableSequence[Assign],
        decorators: MutableSequence[Expr],
        return_type: Expr | None,
        body: MutableSequence[Stmt],
        is_async: bool = False,
        *,
        comment: str | None = None,
    ) -> None:
        self.__ffi_init__(name, args, decorators, return_type, body, is_async)
        if comment is not None:
            self.comment = comment


@c_class("ffi.pyast.Class", init=False)
class Class(Stmt):
    """A ``class`` definition statement.

    Attributes
    ----------
    name
        The class name identifier.
    decorators
        Decorator expressions applied above the class.
    body
        The class body statements.

    Examples
    --------
    .. code-block:: python

        from tvm_ffi import pyast

        ast.Class(
            name=ast.Id(name="MyClass"),
            decorators=[],
            body=[ast.ExprStmt(expr=ast.Id(name="pass"))],
        ).print_python()
        # class MyClass:
        #   pass

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.Class
    # fmt: off
    name: Id
    bases: MutableSequence[Expr]
    decorators: MutableSequence[Expr]
    body: MutableSequence[Stmt]
    kwargs_keys: MutableSequence[str]
    kwargs_values: MutableSequence[Expr]
    if TYPE_CHECKING:
        def __init__(self, name: Id, bases: MutableSequence[Expr] = ..., decorators: MutableSequence[Expr] = ..., body: MutableSequence[Stmt] = ..., kwargs_keys: MutableSequence[str] = ..., kwargs_values: MutableSequence[Expr] = ..., *, comment: str | None = ...) -> None: ...
        def __ffi_init__(self, _0: Id, _1: MutableSequence[Expr], _2: MutableSequence[Expr], _3: MutableSequence[Stmt], _4: MutableSequence[str], _5: MutableSequence[Expr], /) -> None: ...  # ty: ignore[invalid-method-override]
    # fmt: on
    # tvm-ffi-stubgen(end)

    def __init__(
        self,
        name: Id,
        bases: MutableSequence[Expr] | None = None,
        decorators: MutableSequence[Expr] | None = None,
        body: MutableSequence[Stmt] | None = None,
        kwargs_keys: MutableSequence[str] | None = None,
        kwargs_values: MutableSequence[Expr] | None = None,
        *,
        comment: str | None = None,
    ) -> None:
        if bases is None:
            bases = []
        if decorators is None:
            decorators = []
        if body is None:
            body = []
        if kwargs_keys is None:
            kwargs_keys = []
        if kwargs_values is None:
            kwargs_values = []
        self.__ffi_init__(name, bases, decorators, body, kwargs_keys, kwargs_values)
        if comment is not None:
            self.comment = comment


@c_class("ffi.pyast.Comment", init=False)
class Comment(Stmt):
    """A standalone ``# comment`` line.

    The ``comment`` field (inherited from ``Stmt``) holds the comment text.
    It is rendered as a full-line comment rather than an inline comment.

    Examples
    --------
    .. code-block:: python

        from tvm_ffi import pyast

        ast.Comment("TODO: refactor this").print_python()  # # TODO: refactor this

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.Comment
    # fmt: off
    if TYPE_CHECKING:
        def __init__(self, *, comment: str | None = ...) -> None: ...
        def __ffi_init__(self, *, comment: str | None = ...) -> None: ...  # ty: ignore[invalid-method-override]
    # fmt: on
    # tvm-ffi-stubgen(end)

    def __init__(self, comment: str | None) -> None:
        self.__ffi_init__(comment=comment)


@c_class("ffi.pyast.DocString", init=False)
class DocString(Stmt):
    r"""A triple-quoted docstring statement.

    Renders as a ``\"\"\"...\"\"\"``. The ``comment`` field (inherited from
    ``Stmt``) holds the docstring text.

    Examples
    --------
    .. code-block:: python

        from tvm_ffi import pyast

        ast.DocString("This is a docstring.").print_python()

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.DocString
    # fmt: off
    if TYPE_CHECKING:
        def __init__(self, *, comment: str | None = ...) -> None: ...
        def __ffi_init__(self, *, comment: str | None = ...) -> None: ...  # ty: ignore[invalid-method-override]
    # fmt: on
    # tvm-ffi-stubgen(end)

    def __init__(self, comment: str | None) -> None:
        self.__ffi_init__(comment=comment)


@c_class("ffi.pyast.Set")
class Set(Expr):
    """A set expression (``{a, b, c}``).

    Attributes
    ----------
    values
        The element expressions.

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.Set
    # fmt: off
    values: MutableSequence[Expr]
    if TYPE_CHECKING:
        def __init__(self, values: MutableSequence[Expr]) -> None: ...
        def __ffi_init__(self, values: MutableSequence[Expr]) -> None: ...  # ty: ignore[invalid-method-override]
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("ffi.pyast.ComprehensionIter")
class ComprehensionIter(Node):
    """One ``for target in iter [if cond]...`` clause in a comprehension.

    Attributes
    ----------
    target
        The loop variable expression.
    iter
        The iterable expression.
    ifs
        Zero or more filter-condition expressions.

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.ComprehensionIter
    # fmt: off
    target: Expr
    iter: Expr
    ifs: MutableSequence[Expr]
    is_async: bool
    if TYPE_CHECKING:
        def __init__(self, target: Expr, iter: Expr, ifs: MutableSequence[Expr], is_async: bool) -> None: ...
        def __ffi_init__(self, target: Expr, iter: Expr, ifs: MutableSequence[Expr], is_async: bool) -> None: ...  # ty: ignore[invalid-method-override]
    # fmt: on
    # tvm-ffi-stubgen(end)


class ComprehensionKind:
    """Enum-like class for comprehension kinds."""

    List = 0
    Set = 1
    Dict = 2
    Generator = 3


@c_class("ffi.pyast.Comprehension")
class Comprehension(Expr):
    """A comprehension expression.

    Covers list comprehensions (``[elt for ...]``), set comprehensions
    (``{elt for ...}``), dict comprehensions (``{key: value for ...}``),
    and generator expressions (``(elt for ...)``).

    Attributes
    ----------
    kind
        The comprehension kind (a ``ComprehensionKind`` constant).
    elt
        The element expression (or key for dict comprehensions).
    value
        The value expression (only for dict comprehensions; ``None`` otherwise).
    iters
        The list of ``ComprehensionIter`` clauses.

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.Comprehension
    # fmt: off
    kind: int
    elt: Expr
    value: Expr | None
    iters: MutableSequence[ComprehensionIter]
    if TYPE_CHECKING:
        def __init__(self, kind: int, elt: Expr, value: Expr | None, iters: MutableSequence[ComprehensionIter]) -> None: ...
        def __ffi_init__(self, kind: int, elt: Expr, value: Expr | None, iters: MutableSequence[ComprehensionIter]) -> None: ...  # ty: ignore[invalid-method-override]
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("ffi.pyast.Yield")
class Yield(Expr):
    """A yield expression (``yield value``).

    Attributes
    ----------
    value
        The yielded value, or ``None`` for bare ``yield``.

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.Yield
    # fmt: off
    value: Expr | None
    if TYPE_CHECKING:
        def __init__(self, value: Expr | None = ...) -> None: ...
        def __ffi_init__(self, value: Expr | None = ...) -> None: ...  # ty: ignore[invalid-method-override]
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("ffi.pyast.YieldFrom")
class YieldFrom(Expr):
    """A yield-from expression (``yield from iterable``).

    Attributes
    ----------
    value
        The iterable to yield from.

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.YieldFrom
    # fmt: off
    value: Expr
    if TYPE_CHECKING:
        def __init__(self, value: Expr) -> None: ...
        def __ffi_init__(self, value: Expr) -> None: ...  # ty: ignore[invalid-method-override]
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("ffi.pyast.StarredExpr")
class StarredExpr(Expr):
    """A starred expression (``*value``)."""

    # tvm-ffi-stubgen(begin): object/ffi.pyast.StarredExpr
    # fmt: off
    value: Expr
    if TYPE_CHECKING:
        def __init__(self, value: Expr) -> None: ...
        def __ffi_init__(self, value: Expr) -> None: ...  # ty: ignore[invalid-method-override]
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("ffi.pyast.Await")
class AwaitExpr(Expr):
    """An await expression (``await value``)."""

    # tvm-ffi-stubgen(begin): object/ffi.pyast.Await
    # fmt: off
    value: Expr
    if TYPE_CHECKING:
        def __init__(self, value: Expr) -> None: ...
        def __ffi_init__(self, value: Expr) -> None: ...  # ty: ignore[invalid-method-override]
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("ffi.pyast.WalrusExpr")
class WalrusExpr(Expr):
    """A walrus / named expression (``target := value``)."""

    # tvm-ffi-stubgen(begin): object/ffi.pyast.WalrusExpr
    # fmt: off
    target: Expr
    value: Expr
    if TYPE_CHECKING:
        def __init__(self, target: Expr, value: Expr) -> None: ...
        def __ffi_init__(self, target: Expr, value: Expr) -> None: ...  # ty: ignore[invalid-method-override]
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("ffi.pyast.FStr")
class FStr(Expr):
    """An f-string expression (``f"...{x}..."``).

    ``values`` is a list of ``Literal(str)`` for text parts and
    ``FStrValue`` for interpolated expressions.
    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.FStr
    # fmt: off
    values: MutableSequence[Expr]
    if TYPE_CHECKING:
        def __init__(self, values: MutableSequence[Expr]) -> None: ...
        def __ffi_init__(self, values: MutableSequence[Expr]) -> None: ...  # ty: ignore[invalid-method-override]
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("ffi.pyast.FStrValue")
class FStrValue(Expr):
    """A formatted value inside an f-string (``{value!r:.2f}``)."""

    # tvm-ffi-stubgen(begin): object/ffi.pyast.FStrValue
    # fmt: off
    value: Expr
    conversion: int
    format_spec: Expr | None
    if TYPE_CHECKING:
        def __init__(self, value: Expr, conversion: int = ..., format_spec: Expr | None = ...) -> None: ...
        def __ffi_init__(self, value: Expr, conversion: int = ..., format_spec: Expr | None = ...) -> None: ...  # ty: ignore[invalid-method-override]
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("ffi.pyast.ExceptHandler")
class ExceptHandler(Node):
    """One ``except [Type [as name]]:`` clause in a try statement."""

    # tvm-ffi-stubgen(begin): object/ffi.pyast.ExceptHandler
    # fmt: off
    type: Expr | None
    name: str | None
    body: MutableSequence[Stmt]
    if TYPE_CHECKING:
        def __init__(self, type: Expr | None, name: str | None, body: MutableSequence[Stmt]) -> None: ...
        def __ffi_init__(self, type: Expr | None, name: str | None, body: MutableSequence[Stmt]) -> None: ...  # ty: ignore[invalid-method-override]
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("ffi.pyast.Try")
class Try(Stmt):
    """A ``try / except / else / finally`` statement."""

    # tvm-ffi-stubgen(begin): object/ffi.pyast.Try
    # fmt: off
    body: MutableSequence[Stmt]
    handlers: MutableSequence[ExceptHandler]
    orelse: MutableSequence[Stmt]
    finalbody: MutableSequence[Stmt]
    is_star: bool
    if TYPE_CHECKING:
        def __init__(self, body: MutableSequence[Stmt], handlers: MutableSequence[ExceptHandler], orelse: MutableSequence[Stmt] = ..., finalbody: MutableSequence[Stmt] = ..., is_star: bool = ..., *, comment: str | None = ...) -> None: ...
        def __ffi_init__(self, body: MutableSequence[Stmt], handlers: MutableSequence[ExceptHandler], orelse: MutableSequence[Stmt] = ..., finalbody: MutableSequence[Stmt] = ..., is_star: bool = ..., *, comment: str | None = ...) -> None: ...  # ty: ignore[invalid-method-override]
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("ffi.pyast.MatchCase")
class MatchCase(Node):
    """One ``case pattern [if guard]:`` clause in a match statement."""

    # tvm-ffi-stubgen(begin): object/ffi.pyast.MatchCase
    # fmt: off
    pattern: Expr
    guard: Expr | None
    body: MutableSequence[Stmt]
    if TYPE_CHECKING:
        def __init__(self, pattern: Expr, guard: Expr | None, body: MutableSequence[Stmt]) -> None: ...
        def __ffi_init__(self, pattern: Expr, guard: Expr | None, body: MutableSequence[Stmt]) -> None: ...  # ty: ignore[invalid-method-override]
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("ffi.pyast.Match")
class Match(Stmt):
    """A ``match / case`` statement."""

    # tvm-ffi-stubgen(begin): object/ffi.pyast.Match
    # fmt: off
    subject: Expr
    cases: MutableSequence[MatchCase]
    if TYPE_CHECKING:
        def __init__(self, subject: Expr, cases: MutableSequence[MatchCase], *, comment: str | None = ...) -> None: ...
        def __ffi_init__(self, subject: Expr, cases: MutableSequence[MatchCase], *, comment: str | None = ...) -> None: ...  # ty: ignore[invalid-method-override]
    # fmt: on
    # tvm-ffi-stubgen(end)


def from_py(source: Any) -> Node:
    """Convert a Python source string or ``ast.AST`` node to a TVM-FFI text AST node.

    Parameters
    ----------
    source : str | ast.AST
        Either a Python source string or a Python standard-library ``ast``
        node to convert.

    Returns
    -------
    Node
        The corresponding TVM-FFI text AST node.

    Examples
    --------
    .. code-block:: python

        from tvm_ffi import pyast

        node = ast.from_py("x + 1")
        node.print_python()  # x + 1

    """
    from ._pyast_translator import ast_translate  # noqa: PLC0415

    return ast_translate(source)


@c_class("ffi.pyast.VarInfo")
class VarInfo(Object):
    """Metadata for a variable tracked by ``IRPrinter``.

    Attributes
    ----------
    name
        The display name assigned to the variable, or ``None`` if
        a name has not yet been chosen (see ``var_def_no_name``).
    creator
        A ``Function`` callable that, when invoked by the printer,
        produces the definition site AST for this variable.

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.VarInfo
    # fmt: off
    name: str | None
    creator: Callable[..., Any]
    if TYPE_CHECKING:
        def __init__(self, name: str | None, creator: Callable[..., Any]) -> None: ...
        def __ffi_init__(self, _0: str | None, _1: Callable[..., Any], /) -> None: ...  # ty: ignore[invalid-method-override]
    # fmt: on
    # tvm-ffi-stubgen(end)


FrameType = TypeVar("FrameType", bound=Object)


@c_class("ffi.pyast.DefaultFrame", init=False)
class DefaultFrame(Object):
    """The default scoping frame used by ``IRPrinter``.

    A frame collects statements emitted while it is active on the printer's
    frame stack. ``DefaultFrame`` is the simplest frame type and simply
    holds a mutable list of ``Stmt`` nodes.

    Attributes
    ----------
    stmts
        The list of statements accumulated in this frame.

    Examples
    --------
    .. code-block:: python

        printer = IRPrinter()
        with printer.with_frame(DefaultFrame()) as frame:
            # ... emit statements ...
            pass
        print(frame.stmts)

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.DefaultFrame
    # fmt: off
    stmts: MutableSequence[Stmt]
    if TYPE_CHECKING:
        def __init__(self, stmts: MutableSequence[Stmt]) -> None: ...
        def __ffi_init__(self, _0: MutableSequence[Stmt], /) -> None: ...  # ty: ignore[invalid-method-override]
    # fmt: on
    # tvm-ffi-stubgen(end)

    def __init__(self, stmts: list[Stmt] | None = None) -> None:
        if stmts is None:
            stmts = []
        self.__ffi_init__(stmts)


@c_class("ffi.pyast.IRPrinter", init=False)
class IRPrinter(Object):
    """Stateful printer that converts TVM FFI objects into text-printer AST nodes.

    ``IRPrinter`` manages variable bindings and a stack of scoping frames.
    When called on an object, it dispatches to the object's registered
    printer handler to produce AST nodes, automatically defining and
    referencing variables as needed.

    Attributes
    ----------
    cfg
        The ``PrinterConfig`` controlling output formatting.
    obj2info
        Mapping from IR objects to their ``VarInfo`` metadata.
    defined_names
        Mapping from variable name strings to usage counts
        (used for de-duplication).
    frames
        The current stack of scoping frames.
    frame_vars
        Mapping from frame objects to the set of variables
        defined within that frame.

    Examples
    --------
    .. code-block:: python

        from tvm_ffi.pyast import IRPrinter
        from tvm_ffi.pyast import PrinterConfig
        from tvm_ffi.access_path import AccessPath

        printer = IRPrinter(PrinterConfig(indent_spaces=4))
        node = printer(my_obj, AccessPath.root())
        print(node.to_python())

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.IRPrinter
    # fmt: off
    cfg: PrinterConfig
    obj2info: MutableMapping[Any, VarInfo]
    defined_names: MutableMapping[str, int]
    frames: MutableSequence[Any]
    frame_vars: MutableMapping[Any, Any]
    if TYPE_CHECKING:
        def __init__(self, cfg: PrinterConfig, obj2info: MutableMapping[Any, VarInfo], defined_names: MutableMapping[str, int], frames: MutableSequence[Any], frame_vars: MutableMapping[Any, Any]) -> None: ...
        def __ffi_init__(self, _0: PrinterConfig, _1: MutableMapping[Any, VarInfo], _2: MutableMapping[str, int], _3: MutableSequence[Any], _4: MutableMapping[Any, Any], /) -> None: ...  # ty: ignore[invalid-method-override]
        def var_is_defined(self, _1: Object, /) -> bool: ...
        def var_def(self, _1: str, _2: Object, _3: Object | None, /) -> Id: ...
        def var_def_no_name(self, _1: Callable[..., Any], _2: Object, _3: Object | None, /) -> None: ...
        def var_remove(self, _1: Object, /) -> None: ...
        def var_get(self, _1: Object, /) -> Expr | None: ...
        def frame_push(self, _1: Object, /) -> None: ...
        def frame_pop(self, /) -> None: ...
        def __call__(self, _1: Any, _2: AccessPath, /) -> Any: ...
    # fmt: on
    # tvm-ffi-stubgen(end)

    def __init__(self, cfg: PrinterConfig | None = None) -> None:
        if cfg is None:
            cfg = PrinterConfig()
        self.__ffi_init__(cfg, {}, {}, [], {})

    def __call__(self, obj: Any, path: AccessPath) -> Any:
        """Convert *obj* to a text format AST node using this printer's state.

        Parameters
        ----------
        obj
            The TVM FFI object to convert.
        path
            The access path describing how *obj* was reached.

        Returns
        -------
        Any
            The resulting AST node.

        Examples
        --------
        .. code-block:: python

            from tvm_ffi.access_path import AccessPath

            printer = IRPrinter()
            node = printer(my_obj, AccessPath.root())

        """
        info = type(self).__tvm_ffi_type_info__  # type: ignore[attr-defined]
        call_fn = next(m.func for m in info.methods if m.name == "__call__")
        return call_fn(self, obj, path)

    @contextlib.contextmanager
    def with_frame(self, frame: FrameType) -> Generator[FrameType, None, None]:
        """Context manager that pushes *frame* and pops it on exit.

        Any variables defined while the frame is active are associated with
        it and cleaned up when the frame is popped.

        Parameters
        ----------
        frame
            The frame object to activate.

        Yields
        ------
        FrameType
            The same *frame* object, for convenience.

        Examples
        --------
        .. code-block:: python

            printer = IRPrinter()
            with printer.with_frame(DefaultFrame()) as f:
                # statements emitted here go into f.stmts
                pass

        """
        self.frame_push(frame)
        try:
            yield frame
        finally:
            self.frame_pop()


def to_python(obj: Any, cfg: PrinterConfig | None = None) -> str:
    """Convert any TVM FFI object to Python-style source code."""
    if cfg is None:
        cfg = PrinterConfig()
    # If the object is already a pyast Node, print it directly
    if isinstance(obj, Node):
        return obj.to_python(cfg)
    printer = IRPrinter(cfg)
    with printer.with_frame(DefaultFrame()) as frame:
        ret = printer(obj, AccessPath.root())
    if not frame.stmts:
        return ret.to_python(cfg)
    if isinstance(ret, StmtBlock):
        frame.stmts.extend(ret.stmts)
    elif isinstance(ret, Expr):
        frame.stmts.append(ExprStmt(ret))
    elif isinstance(ret, Stmt):
        frame.stmts.append(ret)
    return StmtBlock(frame.stmts).to_python(cfg)


#: Default set of field names hidden by :func:`dump_ast`.
#:
#: These are source-location and commentary fields inherited from
#: :class:`Node` / :class:`Stmt` that are structurally uninteresting when
#: debugging parser input shape. Pass a custom ``skip_fields`` to surface
#: them.
_DUMP_AST_DEFAULT_SKIP: frozenset[str] = frozenset(
    {
        "source_paths",
        "lineno",
        "col_offset",
        "end_lineno",
        "end_col_offset",
        "comment",
    }
)


def dump_ast(  # noqa: PLR0912
    node: Any,
    *,
    indent: int = 0,
    max_depth: int = 16,
    file: Any = None,
    skip_fields: frozenset[str] = _DUMP_AST_DEFAULT_SKIP,
) -> None:
    """Recursively print the structure of a pyast :class:`Node` tree.

    Counterpart to ``_dump_ir_node`` in ``step1_all_loops_trace.py`` tuned
    for tvm-ffi :class:`Node` instances. Uses :func:`iter_fields` (the
    FFI-reflection-backed field iterator) instead of ad-hoc ``dir()``
    scraping, so exactly the declared fields are shown — in registration
    order, inherited fields included.

    Intended for parser debugging: given source text or a :class:`Node`,
    print the structural shape the :class:`IRParser` is about to consume.

    Output format (example, for ``def func(a: T.int32): b = a + 1``)::

        Function
          .name =
            Id
              .name = 'func'
          .args = [1]
              Assign
                .lhs =
                  Id
                    .name = 'a'
                .annotation =
                  Attr
                    .obj =
                      Id
                        .name = 'T'
                    .name = 'int32'
                .aug_op = 0
          .body = [1]
              Assign
                .lhs =
                  Id
                    .name = 'b'
                .rhs =
                  Operation
                    .kind = 0
                    .operands = [2] ...
                .aug_op = 0

    Parameters
    ----------
    node
        The pyast :class:`Node` (or a primitive, list, ``None``) to dump.
        Accepts raw Python source strings via the caller's :func:`from_py`
        preprocessing — not done here to keep the function pure.
    indent
        Current indentation level (used for recursive calls). Callers
        typically leave this at ``0``.
    max_depth
        Stop recursing past this depth; deeper nodes are rendered as
        ``ClassName ...``. Default ``16`` is deep enough for realistic
        parser inputs while bounding pathological cases.
    file
        Stream to write to. Defaults to :data:`sys.stdout`.
    skip_fields
        Field names to omit from the output. Default hides Node's source-
        location metadata (``source_paths``, ``lineno``, ``col_offset``,
        ``end_lineno``, ``end_col_offset``) and ``comment``. Pass
        ``frozenset()`` to surface everything.

    Examples
    --------
    .. code-block:: python

        from tvm_ffi import pyast

        src = pyast.from_py("def f(a: int): return a + 1")
        pyast.dump_ast(src)

    """
    import sys  # noqa: PLC0415

    if file is None:
        file = sys.stdout
    pad = "  " * indent

    # ---- None and scalars ----------------------------------------------------
    if node is None:
        print(f"{pad}None", file=file)
        return
    if isinstance(node, (int, float, bool, str)):
        print(f"{pad}{type(node).__name__}({node!r})", file=file)
        return

    # ---- Depth guard ---------------------------------------------------------
    if indent >= max_depth:
        print(f"{pad}{type(node).__name__} ...", file=file)
        return

    # ---- List / tuple-like top-level containers ------------------------------
    if isinstance(node, (list, tuple)):
        items = list(node)
        if not items:
            print(f"{pad}[]", file=file)
            return
        print(f"{pad}[{len(items)}]", file=file)
        for item in items:
            dump_ast(
                item,
                indent=indent + 1,
                max_depth=max_depth,
                file=file,
                skip_fields=skip_fields,
            )
        return

    # ---- Node (FFI-reflected object) -----------------------------------------
    if isinstance(node, Node):
        print(f"{pad}{type(node).__name__}", file=file)
        for fname, value in iter_fields(node):
            if fname in skip_fields:
                continue
            if value is None:
                continue
            if isinstance(value, (int, float, bool, str)):
                print(f"{pad}  .{fname} = {value!r}", file=file)
                continue
            if isinstance(value, Node):
                print(f"{pad}  .{fname} =", file=file)
                dump_ast(
                    value,
                    indent=indent + 2,
                    max_depth=max_depth,
                    file=file,
                    skip_fields=skip_fields,
                )
                continue
            try:
                items = list(value)
            except TypeError:
                print(f"{pad}  .{fname} = {value!r}", file=file)
                continue
            if not items:
                print(f"{pad}  .{fname} = []", file=file)
            else:
                print(f"{pad}  .{fname} = [{len(items)}]", file=file)
                for item in items:
                    dump_ast(
                        item,
                        indent=indent + 2,
                        max_depth=max_depth,
                        file=file,
                        skip_fields=skip_fields,
                    )
        return

    # ---- Fallback: unknown object type ---------------------------------------
    print(f"{pad}{type(node).__name__}({node!r})", file=file)


# ============================================================================
# Parser frames — cross-dialect dispatch stack
# ============================================================================


class Frame:
    """Generic parser frame — dispatch contribution + IR-construction state."""

    __slots__ = ("dialects", "__dict__")

    def __init__(self, *, dialects: Sequence[Any] | None = None, **data: Any) -> None:
        self.dialects: list[Any] = list(dialects) if dialects else []
        if data:
            self.__dict__.update(data)


class FuncFrame(Frame):
    """Marker frame for a function-definition body (``def f(...): ...``)."""


class ForFrame(Frame):
    """Marker frame for a for-loop body. Pushed by every dialect's
    ``__ffi_for_handler__`` so ``break`` / ``continue`` validation and
    loop-carried-var tracking can query the enclosing loop context
    without caring which dialect owns it."""


class IfFrame(Frame):
    """Marker frame for an if/else branch body."""


class WhileFrame(Frame):
    """Marker frame for a while-loop body."""


class WithFrame(Frame):
    """Marker frame for a with-block body. Pushed by every dialect's
    ``__ffi_with_handler__``."""


# ============================================================================
# Parse-hook factories — build frame-mutating callables for
# ``__ffi_parse_hooks__`` declarations. See design_docs/parser_frame_hooks.md.
# ============================================================================


def frame_setter(
    field: str,
    frame_cls: type[Frame] = FuncFrame,
) -> Callable[..., None]:
    """Return a parse hook that overwrites ``frame.<field>`` with its
    single positional argument.

    Used for prologue calls like ``T.func_ret(i32)`` that set a scalar
    property on the enclosing function. Raises :class:`TypeError` at
    parse time if the hook receives anything other than exactly one
    positional arg.
    """

    def _hook(parser: "IRParser", *args: Any, **kwargs: Any) -> None:
        if kwargs or len(args) != 1:
            raise TypeError(
                f"frame_setter({field!r}): expected exactly one positional "
                f"arg, got args={args!r} kwargs={kwargs!r}",
            )
        frame = parser.find_frame(frame_cls, origin=f"frame_setter({field!r})")
        setattr(frame, field, args[0])

    _hook.__ffi_parse_hook__ = True  # type: ignore[attr-defined]
    _hook.__name__ = f"_frame_setter_{field}"
    return _hook


def frame_merger(
    field: str,
    frame_cls: type[Frame] = FuncFrame,
) -> Callable[..., None]:
    """Return a parse hook that shallow-merges a dict arg into
    ``frame.<field>``.

    Equivalent to ``frame.<field> = {**(frame.<field> or {}), **value}``.
    Used for prologue calls like ``T.func_attr({"noalias": True})``;
    multiple such calls in the same body accumulate rather than
    overwriting each other.
    """

    def _hook(parser: "IRParser", *args: Any, **kwargs: Any) -> None:
        if kwargs or len(args) != 1:
            raise TypeError(
                f"frame_merger({field!r}): expected exactly one positional "
                f"dict arg, got args={args!r} kwargs={kwargs!r}",
            )
        value = args[0]
        if not isinstance(value, dict):
            raise TypeError(
                f"frame_merger({field!r}): expected dict arg, got "
                f"{type(value).__name__}",
            )
        frame = parser.find_frame(frame_cls, origin=f"frame_merger({field!r})")
        current = getattr(frame, field, None)
        setattr(frame, field, {**(current or {}), **value})

    _hook.__ffi_parse_hook__ = True  # type: ignore[attr-defined]
    _hook.__name__ = f"_frame_merger_{field}"
    return _hook


# ============================================================================
# IRParser — trait-driven parser (PyAST → IR objects)
# ============================================================================


class IRParser:
    """Trait-driven IR parser: converts PyAST nodes into IR objects.

    Counterpart to :class:`IRPrinter`, which converts IR objects into PyAST.
    Uses the same ``__ffi_ir_traits__`` metadata to drive the parse dispatch.

    Examples
    --------
    .. code-block:: python

        parser = IRParser()
        ast_node = pyast.from_py(source_text)
        ir_obj = parser.parse(ast_node)

    """

    def __init__(
        self,
        lang_modules: dict[str, Any] | None = None,
        *,
        var_factory: Callable[[str, Any], Any] | None = None,
    ) -> None:
        """Construct an :class:`IRParser`.

        Parameters
        ----------
        lang_modules
            Prefix → language-module registry — maps identifier strings
            (``"T"``, ``"I"``, …) to Python objects that resolve during
            ``eval_expr(Id("T"))``. Also becomes the initial registered
            dialect list for frame-based dispatch.
        var_factory
            Optional legacy ``(name, ty) → Var`` factory, retained for
            back-compat with tests that still wire it explicitly.
        """
        self._scopes: list[dict[str, Any]] = [{}]
        self._lang_modules: dict[str, Any] = dict(lang_modules) if lang_modules else {}
        self.var_factory = var_factory

        # --- Frame-based dispatch state ---
        self._frames: list[Frame] = []
        self._registered_dialects: list[Any] = []
        for mod in self._lang_modules.values():
            if mod not in self._registered_dialects:
                self._registered_dialects.append(mod)

    # ------------------------------------------------------------------
    # Frame-based dispatch API
    # ------------------------------------------------------------------

    def register_dialect(self, *dialects: Any) -> None:
        """Add dialects to the parser's base registry."""
        for d in dialects:
            if d not in self._registered_dialects:
                self._registered_dialects.append(d)

    @contextlib.contextmanager
    def push_frame(self, frame: Frame) -> Iterator[Frame]:
        """Push ``frame`` onto the dispatch stack for the ``with`` block."""
        self._frames.append(frame)
        try:
            yield frame
        finally:
            popped = self._frames.pop()
            assert popped is frame, (
                "IRParser frame-stack corruption: the frame popped at exit "
                f"({type(popped).__name__}) is not the frame that was pushed "
                f"({type(frame).__name__}). A handler probably mutated "
                "``_frames`` without using ``push_frame``."
            )

    @contextlib.contextmanager
    def with_dialects(self, *dialects: Any) -> Iterator[None]:
        """Sugar: push a bare :class:`Frame` that only contributes dialects."""
        with self.push_frame(Frame(dialects=dialects)):
            yield

    def active_dialects(self) -> Iterator[Any]:
        """Yield dialects in dispatch order: innermost frame first, then
        outer frames, then the base registry."""
        seen: set[int] = set()
        for frame in reversed(self._frames):
            for d in frame.dialects:
                key = id(d)
                if key in seen:
                    continue
                seen.add(key)
                yield d
        for d in self._registered_dialects:
            key = id(d)
            if key in seen:
                continue
            seen.add(key)
            yield d

    def find_frame(
        self,
        frame_cls: type[Frame],
        *,
        origin: str | None = None,
    ) -> Frame:
        """Return the innermost active frame of ``frame_cls``.

        Used by parse hooks that need to mutate enclosing-construct state
        (``FuncFrame.attrs`` from ``T.func_attr({...})`` etc.). Raises
        :class:`RuntimeError` if no such frame is on the dispatch stack —
        the error is deliberately loud so missing / mis-ordered frames
        show up at the site that needs them rather than silently corrupting
        downstream IR construction.
        """
        for frame in reversed(self._frames):
            if isinstance(frame, frame_cls):
                return frame
        origin_s = f" (needed by {origin})" if origin else ""
        raise RuntimeError(
            f"find_frame({frame_cls.__name__}): no ancestor frame of that "
            f"type on the dispatch stack{origin_s}. Active frames: "
            + ", ".join(type(f).__name__ for f in self._frames),
        )

    def _lookup_hook(self, name: str) -> Any:
        """Find ``name`` as an attribute on the nearest active dialect."""
        for dialect in self.active_dialects():
            val = getattr(dialect, name, None)
            if val is not None:
                return val
        return None

    # ------------------------------------------------------------------
    # Var-table
    # ------------------------------------------------------------------

    def push_scope(self) -> None:
        """Open a new innermost scope."""
        self._scopes.append({})

    def pop_scope(self) -> None:
        """Close the innermost scope."""
        if len(self._scopes) <= 1:
            raise RuntimeError(
                "pop_scope called with only the global scope on the stack — "
                "indicates an unbalanced push/pop pair. Prefer "
                "IRParser.scoped_frame() to keep them paired automatically.",
            )
        self._scopes.pop()

    @contextlib.contextmanager
    def scoped_frame(self) -> Iterator[None]:
        """Push a fresh scope for the duration of the ``with`` block."""
        self.push_scope()
        try:
            yield
        finally:
            self.pop_scope()

    def define(self, name: str, var: Any) -> None:
        """Register ``name → var`` in the innermost scope."""
        innermost = self._scopes[-1]
        if name in innermost:
            raise NameError(
                f"variable {name!r} is already defined in the innermost "
                f"scope (depth {self.scope_depth - 1}); this is usually a "
                f"parser bug (duplicate parameter / loop var / bind). "
                f"Call IRParser.redefine to overwrite intentionally. "
                f"Active scope chain: {self._format_scope_chain()}",
            )
        innermost[name] = var

    def redefine(self, name: str, var: Any) -> None:
        """Overwrite ``name`` in the innermost scope without checking."""
        self._scopes[-1][name] = var

    def lookup(self, name: str) -> Any | None:
        """Resolve ``name`` against the scope chain (innermost first)."""
        for scope in reversed(self._scopes):
            if name in scope:
                return scope[name]
        return None

    def lookup_required(self, name: str, *, role: str = "variable") -> Any:
        """Like :meth:`lookup` but raises :class:`NameError` on miss."""
        val = self.lookup(name)
        if val is None:
            raise NameError(
                f"{role} {name!r} is not defined; active scope chain "
                f"(innermost first): {self._format_scope_chain()}",
            )
        return val

    @property
    def scope_depth(self) -> int:
        """Total number of active scopes, including the global one."""
        return len(self._scopes)

    def _format_scope_chain(self) -> str:
        """Render the scope chain for inclusion in error messages."""
        parts: list[str] = []
        n = len(self._scopes)
        for offset, scope in enumerate(reversed(self._scopes)):
            depth = n - 1 - offset
            keys = ", ".join(sorted(scope.keys())) or ""
            parts.append(f"depth {depth}: {{{keys}}}")
        return "[" + "; ".join(parts) + "]"

    # ---- Public entry point ----

    def parse(self, source: str | Node) -> Any:
        """Parse Python source text or a PyAST node into IR objects.

        Parameters
        ----------
        source
            Either a Python source string (passed through :func:`from_py`
            first) or a PyAST :class:`Node` directly.

        Returns
        -------
        Any
            The parsed IR object(s).  For a :class:`StmtBlock` input,
            returns a list of IR objects.

        """
        if isinstance(source, str):
            source = from_py(source)
            import sys

            dump_ast(source, file=sys.stdout)
        return self.visit(source)

    # ---- Unified dispatch ----

    def visit(self, node: Node) -> Any:
        """Dispatch a PyAST node to its ``visit_*`` method."""
        if isinstance(node, StmtBlock):
            return self.visit_stmt_block(node)
        if isinstance(node, Function):
            return self.visit_function(node)
        if isinstance(node, Class):
            return self.visit_class(node)
        if isinstance(node, Assign):
            return self.visit_assign(node)
        if isinstance(node, ExprStmt):
            return self.visit_expr_stmt(node)
        if isinstance(node, Return):
            return self.visit_return(node)
        if isinstance(node, Assert):
            return self.visit_assert(node)
        if isinstance(node, If):
            return self.visit_if(node)
        if isinstance(node, While):
            return self.visit_while(node)
        if isinstance(node, For):
            return self.visit_for(node)
        if isinstance(node, With):
            return self.visit_with(node)
        if isinstance(node, Id):
            return self.visit_id(node)
        raise NotImplementedError(f"visit: unhandled {type(node).__name__}")

    def visit_body(self, stmts: Sequence[Stmt]) -> list[Any]:
        """Visit a sequence of statements and return their IR objects.

        A ``None`` result from ``visit`` is treated as a signal that the
        statement was consumed by a frame-mutating parse hook (see
        ``__ffi_parse_hook__`` — e.g. ``T.func_attr({...})`` prologue
        calls). The ``None`` is dropped so no residue appears in the
        constructed body.
        """
        out: list[Any] = []
        for s in stmts:
            if isinstance(s, ExprStmt) and isinstance(s.expr, Id) and s.expr.name == "pass":
                continue
            result = self.visit(s)
            if result is None:
                continue
            out.append(result)
        return out

    def eval_expr(self, node: Expr) -> Any:
        """Evaluate an expression to a Python value.

        Resolution order for :class:`Id`:
        1. Local scope (variables registered via :meth:`define`).
        2. Language-module registry passed at construction.
        3. :meth:`resolve_module` hook for subclass extension.
        """
        if isinstance(node, Literal):
            return node.value
        if isinstance(node, Id):
            var = self.lookup(node.name)
            if var is not None:
                return var
            if node.name in self._lang_modules:
                return self._lang_modules[node.name]
            return self.resolve_module(node.name)
        if isinstance(node, Attr):
            qualified = self._maybe_qualified_class(node)
            if qualified is not None:
                return qualified
            return getattr(self.eval_expr(node.obj), node.name)
        if isinstance(node, Operation):
            return self.visit_operation(node)
        if isinstance(node, Call):
            return self.visit_call(node)
        if isinstance(node, Index):
            return self.visit_index(node)
        if isinstance(node, List):
            return [self.eval_expr(v) for v in node.values]
        if isinstance(node, Tuple):
            return tuple(self.eval_expr(v) for v in node.values)
        if isinstance(node, Dict):
            return {
                self.eval_expr(k): self.eval_expr(v)
                for k, v in zip(node.keys, node.values)
            }
        raise NotImplementedError(f"eval_expr: unhandled {type(node).__name__}")

    def resolve_module(self, name: str) -> Any:
        """Subclass hook for resolving identifiers not in scope or lang modules."""
        raise NameError(f"Unknown identifier: {name!r}")

    def _maybe_qualified_class(self, node: "Attr") -> Any:
        """Attempt to resolve an ``Attr`` chain as a ``@py_class`` type key.

        The default printer emits leaf IR classes (Tier-3) as their
        fully-qualified type key — e.g. ``mini.tir.Cast(target=...)``.
        When the leftmost :class:`Id` isn't a registered language
        module, flatten the ``Attr`` chain into a dotted string and
        look it up in the type-key registry. Returns the class on hit,
        :data:`None` on miss (callers should fall back to the normal
        attribute walk).
        """
        parts: list[str] = []
        current: Any = node
        while isinstance(current, Attr):
            parts.append(current.name)
            current = current.obj
        if not isinstance(current, Id):
            return None
        root = current.name
        # If the leftmost Id is a registered lang module or bound
        # variable, use the normal attribute walk — the type-key
        # fallback is a last resort.
        if self.lookup(root) is not None or root in self._lang_modules:
            return None
        parts.append(root)
        parts.reverse()
        qualified = ".".join(parts)
        try:
            from tvm_ffi import core as _core  # noqa: PLC0415

            info = _core._lookup_or_register_type_info_from_type_key(qualified)
        except Exception:  # noqa: BLE001
            return None
        cls = getattr(info, "type_cls", None)
        if cls is None or cls is Object:
            # ``type_cls`` defaults to :class:`Object` for type keys
            # that exist in the FFI-level registry but haven't been
            # bound to a Python class — treat that as a miss.
            return None
        return cls

    # ---- Expression visitors ----

    def visit_id(self, node: Id) -> Any:
        """Look up a variable by name in the current scope."""
        var = self.lookup(node.name)
        if var is not None:
            return var
        raise ValueError(f"Undefined variable: {node.name!r}")

    def visit_index(self, node: Index) -> Any:
        """Evaluate a subscript expression ``obj[indices]``.

        Dispatch order:
        1. If a language module exposes ``load``, call
           ``load(parser, obj, indices)``.
        2. Otherwise invoke ``obj[indices]`` — IR classes like ``Buffer``
           typically define ``__getitem__`` to build a trait-driven Load.
        """
        obj = self.eval_expr(node.obj)
        indices = [self.eval_expr(i) for i in node.idx]
        load = self._lookup_hook("load")
        if load is not None:
            return load(self, obj, indices)
        if len(indices) == 1:
            return obj[indices[0]]
        return obj[tuple(indices)]

    def visit_call(self, node: Call) -> Any:
        """Evaluate a call expression by invoking the resolved callee.

        Dispatch is split into three paths:

        1. **Parse-aware dispatcher** — if ``callee.__ffi_parse_aware__``
           is ``True``, the callee was registered by
           :func:`tvm_ffi.parse_dispatch.register_parser` and expects
           ``(parser, node)``. Handles Tier 1 / 2 / 3 internally.
        2. **IR class (``@py_class``)** — if ``callee`` is a registered
           IR class, look up its dispatcher via
           :func:`tvm_ffi.parse_dispatch.lookup_parser` in the owning
           language module's ``__ffi_parsers__`` registry and invoke it
           as a parse-aware dispatcher. If no dispatcher is registered,
           fall back to value-eager construction (with primitive
           wrapping via the active dialect's
           ``__ffi_default_{int,float,bool}_ty__`` hooks).
        3. **Plain callable** — for every other callable (``range``,
           ``T.float32`` factory, user helper, etc.) evaluate args /
           kwargs eagerly and invoke the callee with those values.
        """
        callee = self.eval_expr(node.callee)

        # Path 1 — the callee is a parse-aware dispatcher.
        if getattr(callee, "__ffi_parse_aware__", False):
            return callee(self, node)

        # Path 1b — the callee is a frame-mutating parse hook (e.g. a
        # ``T.func_attr({...})`` prologue). The hook receives evaluated
        # args / kwargs, mutates the enclosing frame, and returns None.
        # ``visit_body`` filters those Nones so the hook leaves no
        # body-residue. See design_docs/parser_frame_hooks.md.
        if getattr(callee, "__ffi_parse_hook__", False):
            args = [self.eval_expr(a) for a in node.args]
            kwargs = {
                k: self.eval_expr(v)
                for k, v in zip(node.kwargs_keys, node.kwargs_values)
            }
            return callee(self, *args, **kwargs)

        # Path 2 — the callee is an IR class. Look for a registered
        # dispatcher in the owning language module. If found, run the
        # three-tier dispatch; otherwise fall back to value-eager
        # construction.
        if isinstance(callee, type) and hasattr(callee, "__tvm_ffi_type_info__"):
            from tvm_ffi.parse_dispatch import lookup_parser  # noqa: PLC0415

            owner = sys.modules.get(callee.__module__)
            if owner is not None:
                dispatcher = lookup_parser(owner, callee.__name__)
                if dispatcher is not None:
                    return dispatcher(self, node)

            args = [self.eval_expr(a) for a in node.args]
            kwargs = {
                k: self.eval_expr(v)
                for k, v in zip(node.kwargs_keys, node.kwargs_values)
            }
            if hasattr(callee, "__ffi_ir_traits__"):
                args = [self._wrap_primitive_ast(a) for a in args]
                kwargs = {
                    k: self._wrap_primitive_ast(v) for k, v in kwargs.items()
                }
            return callee(*args, **kwargs)

        # Path 3 — plain callable, value-eager.
        args = [self.eval_expr(a) for a in node.args]
        kwargs = {
            k: self.eval_expr(v) for k, v in zip(node.kwargs_keys, node.kwargs_values)
        }
        return callee(*args, **kwargs)

    def _wrap_primitive_ast(self, value: Any) -> Any:
        """Lift raw Python primitives to dialect IR via active-dialect
        ``__ffi_default_{int,float,bool}_ty__`` hooks. Objects pass through."""
        if value is None or not isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, bool):
            hook_name = "__ffi_default_bool_ty__"
        elif isinstance(value, int):
            hook_name = "__ffi_default_int_ty__"
        else:
            hook_name = "__ffi_default_float_ty__"
        for dialect in self.active_dialects():
            handle = getattr(dialect, hook_name, None)
            if handle is not None and callable(handle):
                return handle(value)
        return value

    def visit_operation(self, node: Operation) -> Any:
        """Dispatch a :class:`Operation` via the lang module's
        ``__ffi_op_classes__`` map.

        Each ``{OperationKind: handler}`` entry accepts either a direct
        callable (auto-wired closures from :func:`_wire_binop` /
        :func:`_wire_unaryop`) or a dotted-string path (user-defined
        custom dispatchers — e.g. mini.mlir.arith's type-predicate
        ``_op_add`` / ``_op_sub``). Callables are invoked directly;
        strings are resolved via :meth:`eval_expr` against the
        lang-module registry.
        """
        if node.op == OperationKind.Parens:
            return self.eval_expr(node.operands[0])

        # ----------------------------------------------------------------
        # Cross-dialect operation dispatch
        # ----------------------------------------------------------------

        tried: list[str] = []
        for dialect in self.active_dialects():
            op_classes = getattr(dialect, "__ffi_op_classes__", None)
            if not op_classes:
                continue
            ref = op_classes.get(node.op)
            if ref is None:
                continue
            if callable(ref):
                tried.append(getattr(ref, "__name__", repr(ref)))
                parse_fn: Any = ref
            else:
                tried.append(str(ref))
                parts = str(ref).split(".")
                expr: Expr = Id(name=parts[0])
                for p in parts[1:]:
                    expr = Attr(obj=expr, name=p)
                parse_fn = self.eval_expr(expr)
            result = parse_fn(self, node)
            if result is not None:
                return result

        if tried:
            raise NotImplementedError(
                f"visit_operation: none of the {len(tried)} handlers "
                f"({', '.join(tried)}) accepted operation kind={node.op}. "
                f"Check that at least one dialect's handler matches the "
                f"operand types at this call site.",
            )
        raise NotImplementedError(
            f"visit_operation: no dialect declares ``__ffi_op_classes__`` "
            f"with an entry for OperationKind kind={node.op}. Register a "
            f"``{{OperationKind.X: <callable or 'dotted.path'>, ...}}`` "
            f"map on at least one language module to enable sugar-form "
            f"(``a + b``) parsing.",
        )

    def visit_stmt_block(self, node: StmtBlock) -> Any:
        return self.visit_body(node.stmts)

    def visit_expr_stmt(self, node: ExprStmt) -> Any:
        return self.eval_expr(node.expr)

    # ---- Statement visitors ----

    def visit_function(self, node: Function) -> Any:
        """Parse a function definition via decorator-based registry dispatch."""
        if not node.decorators:
            raise ValueError(
                f"Function {node.name.name!r} must be decorated to dispatch parser",
            )
        handler = self.eval_expr(node.decorators[-1])
        if not callable(handler):
            raise TypeError(
                f"Decorator did not resolve to a callable parse handler: {handler!r}",
            )
        with self.push_frame(FuncFrame(name=node.name.name)):
            return handler(self, node)

    def visit_class(self, node: Class) -> Any:
        """Parse a class definition via decorator-based registry dispatch."""
        if not node.decorators:
            raise ValueError(
                f"Class {node.name.name!r} must be decorated to dispatch parser",
            )
        handler = self.eval_expr(node.decorators[-1])
        if not callable(handler):
            raise TypeError(
                f"Decorator did not resolve to a callable parse handler: {handler!r}",
            )
        with self.push_frame(FuncFrame(name=node.name.name, is_class=True)):
            return handler(self, node)

    def visit_return(self, node: Return) -> Any:
        """Parse a ``return <value>`` stmt."""
        value = self.eval_expr(node.value) if node.value is not None else None
        hook = self._lookup_hook("ret")
        if hook is not None:
            return hook(self, value)
        return value

    def visit_if(self, node: If) -> Any:
        """Parse ``if / else``. Looks up ``if_stmt`` hook on language modules."""
        cond = self.eval_expr(node.cond)
        with self.scoped_frame(), self.push_frame(IfFrame()):
            then_body = self.visit_body(node.then_branch)
        else_body: list[Any] = []
        if node.else_branch:
            with self.scoped_frame(), self.push_frame(IfFrame()):
                else_body = self.visit_body(node.else_branch)
        hook = self._lookup_hook("if_stmt")
        if hook is not None:
            return hook(self, cond, then_body, else_body)
        return (cond, then_body, else_body)

    def visit_for(self, node: For) -> Any:
        """Parse ``for`` loop."""
        iter_val = self.eval_expr(node.rhs)
        handler = getattr(type(iter_val), "__ffi_for_handler__", None)
        if handler is not None:
            return handler(iter_val, self, node)
        hook = self._lookup_hook("for_stmt")
        if hook is not None:
            return hook(self, node, iter_val)
        # Default fallback — visit body without binding the loop var.
        with self.scoped_frame(), self.push_frame(ForFrame()):
            body = self.visit_body(node.body)
        return (node.lhs, iter_val, body)

    def visit_with(self, node: With) -> Any:
        """Parse ``with`` stmt. Looks up ``with_stmt`` hook on language modules."""
        ctx = self.eval_expr(node.rhs)
        handler = getattr(type(ctx), "__ffi_with_handler__", None)
        if handler is not None:
            return handler(ctx, self, node)
        hook = self._lookup_hook("with_stmt")
        if hook is not None:
            return hook(self, node, ctx)
        with self.scoped_frame(), self.push_frame(WithFrame()):
            body = self.visit_body(node.body)
        return (ctx, body)

    def visit_while(self, node: While) -> Any:
        """Parse ``while`` loop. Looks up ``while_stmt`` hook on language modules."""
        cond = self.eval_expr(node.cond)
        with self.scoped_frame(), self.push_frame(WhileFrame()):
            body = self.visit_body(node.body)
        hook = self._lookup_hook("while_stmt")
        if hook is not None:
            return hook(self, cond, body)
        return (cond, body)

    def visit_assert(self, node: Assert) -> Any:
        """Parse ``assert cond, msg``. Looks up ``assert_stmt`` hook."""
        cond = self.eval_expr(node.cond)
        msg = self.eval_expr(node.msg) if node.msg is not None else None
        hook = self._lookup_hook("assert_stmt")
        if hook is not None:
            return hook(self, cond, msg)
        return (cond, msg)

    def visit_assign(self, node: Assign) -> Any:
        """Parse an assignment via the ``__ffi_assign__`` hook."""
        assign_hook = self._lookup_hook("__ffi_assign__")
        if assign_hook is None:
            raise NotImplementedError(
                "visit_assign: no ``__ffi_assign__`` hook on any registered "
                "language module. Register one to handle pyast.Assign.",
            )
        return assign_hook(self, node)


def parse(source: str | Node, lang_modules: dict[str, Any] | None = None) -> Any:
    """Parse Python source text or a PyAST node into IR objects."""
    return IRParser(lang_modules=lang_modules).parse(source)


# ---------------------------------------------------------------------------
# Re-export visitor utilities.
# ---------------------------------------------------------------------------
# isort: off
from tvm_ffi._pyast_visitor import NodeTransformer as NodeTransformer  # noqa: PLC0414
from tvm_ffi._pyast_visitor import NodeVisitor as NodeVisitor  # noqa: PLC0414
from tvm_ffi._pyast_visitor import iter_child_nodes as iter_child_nodes  # noqa: PLC0414
from tvm_ffi._pyast_visitor import iter_fields as iter_fields  # noqa: PLC0414
# isort: on
