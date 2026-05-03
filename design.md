<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# IR Dialects

This document sketches the current standard core dialect and how later dialects
can specialize it.  The node tables use the proposed `$dialect`, `$mnemonic`,
and `$generics` metadata model described in the last section.

The standard dialect lives in C++ as `tvm::ffi::std_`, is exposed to Python as
`tvm_ffi.std`, and uses type keys under `ffi.std.*`.

**Open Questions**

- Type arguments to calls.
- Whether `std.DataTypeImm` should exist as a first-class node.


## Std Dialect

`std` defines the common type, expression, statement, structure, and attribute
nodes.  The following classes are abstract bases and are not directly
constructible:
- `std.Node`
- `std.Ty`
- `std.Stmt`
- `std.Attrs`
- `std.Structure`
- `std.Expr`
- `std.Bind`

### Types

| Node | Fields | $dialect | $mnemonic | $generics |
|---|---|---:|---:|---|
| `std.AnyTy` | none | `std` | `std$AnyTy` | N/A |
| `std.PrimTy` | `dtype` | `std` | `std$PrimTy` | N/A |
| `std.TupleType` | `fields` | `std` | `std$TupleType` | N/A |
| `std.TensorTy` | `shape`, `dtype` | `std` | `std$TensorTy` | N/A |

The std text printer still uses concise type renderings such as `Any`, `i32`,
`tuple[i32, f32]`, and `bf16[3, 12]`; those are concrete text-format choices,
not `$generics` values.

### Structures

`std.Structure` is the base for structural helper nodes that are not statements
or expressions.

| Node | Fields | $dialect | $mnemonic | $generics |
|---|---|---:|---:|---|
| `std.Range` | `start`, `stop`, `step` | `std` | `std$Range` | N/A |

`std.Range` prints with slice/range syntax such as `1:10:2`, `:10`, and `::2`.

### Attributes

All concrete subclasses of `std.Attrs` print as call syntax with keyword-only
arguments: `$name(attr_0=..., attr_1=...)`.

| Node | Fields | $dialect | $mnemonic | $generics |
|---|---|---:|---:|---|
| `std.DictAttrs` | `values` | `std` | `std$DictAttrs` | N/A |

### Expressions

| Node | Fields | $dialect | $mnemonic | $generics |
|---|---|---:|---:|---|
| `std.Var` | `ty`, `name` | `std` | `std$Var` | N/A |
| `std.IntImm` | `ty`, `value` | `std` | `std$IntImm` | N/A |
| `std.FloatImm` | `ty`, `value` | `std` | `std$FloatImm` | N/A |
| `std.StringImm` | `ty`, `value` | `std` | `std$StringImm` | N/A |
| `std.Add` | `ty`, `a`, `b` | `std` | `std$Add` | `__add__` |
| `std.Sub` | `ty`, `a`, `b` | `std` | `std$Sub` | `__sub__` |
| `std.Mul` | `ty`, `a`, `b` | `std` | `std$Mul` | `__mul__` |
| `std.FloorDiv` | `ty`, `a`, `b` | `std` | `std$FloorDiv` | `__floordiv__` |
| `std.FloorMod` | `ty`, `a`, `b` | `std` | `std$FloorMod` | `__mod__` |
| `std.Min` | `ty`, `a`, `b` | `std` | `std$Min` | `min` |
| `std.Max` | `ty`, `a`, `b` | `std` | `std$Max` | `max` |
| `std.Eq` | `ty`, `a`, `b` | `std` | `std$Eq` | `__eq__` |
| `std.Ne` | `ty`, `a`, `b` | `std` | `std$Ne` | `__ne__` |
| `std.Le` | `ty`, `a`, `b` | `std` | `std$Le` | `__le__` |
| `std.Ge` | `ty`, `a`, `b` | `std` | `std$Ge` | `__ge__` |
| `std.Gt` | `ty`, `a`, `b` | `std` | `std$Gt` | `__gt__` |
| `std.Lt` | `ty`, `a`, `b` | `std` | `std$Lt` | `__lt__` |
| `std.And` | `ty`, `a`, `b` | `std` | `std$And` | `__and__` |
| `std.Or` | `ty`, `a`, `b` | `std` | `std$Or` | `__or__` |
| `std.Not` | `ty`, `operand` | `std` | `std$Not` | `__invert__` |
| `std.Load` | `ty`, `var`, `indices` | `std` | `std$Load` | `__load__` |
| `std.Cast` | `ty`, `value` | `std` | `std$Cast` | `__cast__` |
| `std.Call` | `ty`, `callee`, `args`, `attr` | `std` | `std$Call` | N/A |

`std.Expr` conversion accepts existing expressions and Python scalar literals.
Integers, floats, and strings become `std.IntImm`, `std.FloatImm`, and
`std.StringImm` with `std.AnyTy`.

The current text printer renders `std.Load` as indexed access, `std.Cast` with a
dtype callee such as `std.i32(x)`, and `std.Call` as `std.call(callee, *args,
**attrs)`.

### Regional Statements

| Node | Fields | $dialect | $mnemonic | $generics |
|---|---|---:|---:|---|
| `std.Module` | `funcs` | `std` | `std$Module` | N/A |
| `std.Func` | `symbol`, `attrs`, `args`, `ret_type`, `body` | `std` | `std$Func` | N/A |
| `std.IfStmt` | `cond`, `then_body`, `else_body` | `std` | `std$IfStmt` | `__if__` |
| `std.Scope` | `attrs`, `vars`, `body` | `std` | `std$Scope` | N/A |
| `std.For` | `range_`, `attrs`, `vars`, `body` | `std` | `std$For` | `__for__` |
| `std.While` | `cond`, `attrs`, `vars`, `body` | `std` | `std$While` | `__while__` |

`std.For` and `std.While` are subclasses of `std.Scope`, because they share
scope-carried variables, attributes, and a statement body.

### Binding Statements

`std.Bind` is an abstract base for statements that define variables.  It has
common fields `vars: list[std.Var]` and `attrs: std.Attrs | None`.

| Node | Fields | $dialect | $mnemonic | $generics |
|---|---|---:|---:|---|
| `std.ExprBind` | `vars`, `attrs`, `expr` | `std` | `std$ExprBind` | `__bind__` |
| `std.VarDef` | `vars`, `attrs` | `std` | `std$VarDef` | `__var_def__` |

Current text format:

```python
y = rhs
y = std.bind(rhs, tag="demo")
y, z = rhs

y = std.var_def(i32)
y = std.var_def(i32, tag="demo")
y, z = std.var_def(i32, bf16[3, 12])
```

For `std.ExprBind`, type annotations are not printed because the type is usually
derived from the right-hand side.  For `std.VarDef`, declared variable types are
printed as positional arguments to `std.var_def(...)`.

If `std.ExprBind.vars` is empty, the expression is emitted as an expression
statement.  If `std.VarDef.vars` is empty, it emits `pass` when there are no
attributes, or a standalone `std.var_def(**attrs)` when attributes are present.

### Other Statements

| Node | Fields | $dialect | $mnemonic | $generics |
|---|---|---:|---:|---|
| `std.Store` | `var`, `indices`, `rhs` | `std` | `std$Store` | `__store__` |
| `std.Return` | `vars` | `std` | `std$Return` | `__return__` |
| `std.Yield` | `vars` | `std` | `std$Yield` | `__yield__` |
| `std.Break` | none | `std` | `std$Break` | `__break__` |
| `std.Continue` | none | `std` | `std$Continue` | `__continue__` |

`std.Store` prints as indexed assignment, including `x[()] = v` when indices are
empty.  `std.Return` and `std.Yield` support multiple variables.

## TIRx Dialect

The TIRx rows below are directional sketches for how a later dialect can
specialize the standard dialect.

### Types

| Node | $dialect | $mnemonic | $generics | $kind |
|---|---:|---:|---|---|
| `tirx.PointerType` | `tirx` | `tirx$PointerType` | N/A | `std.Ty` |
| `tirx.FuncType` | `tirx` | `tirx$FuncType` | N/A | `std.Ty` |

### Variables

| Node | $dialect | $mnemonic | $generics | $kind |
|---|---:|---:|---|---|
| `tirx.Var` | `tirx` | `tirx$Var` | N/A | `std.Var` |
| `tirx.SizeVar` | `tirx` | `tirx$SizeVar` | N/A | `tirx.Var` |
| `tirx.Buffer` | `tirx` | `tirx$Buffer` | N/A | `std.Var` |

### Expressions

| Node | $dialect | $mnemonic | $generics | $kind |
|---|---:|---:|---|---|
| `tirx.Select` | `tirx` | `tirx$Select` | N/A | `std.Expr` |
| `tirx.Ramp` | `tirx` | `tirx$Ramp` | N/A | `std.Expr` |
| `tirx.Broadcast` | `tirx` | `tirx$Broadcast` | N/A | `std.Expr` |
| `tirx.Shuffle` | `tirx` | `tirx$Shuffle` | N/A | `std.Expr` |
| `tirx.BufferLoad` | `tirx` | `tirx$BufferLoad` | `__load__` | `std.Load` |
| `tirx.Call` | `tirx` | `tirx$Call` | `__call__` | `std.Call` |

### Statements

| Node | $dialect | $mnemonic | $generics | $kind |
|---|---:|---:|---|---|
| `tirx.AttrStmt` | `tirx` | `tirx$AttrStmt` | N/A | `std.Scope` |
| `tirx.Bind` | `tirx` | `tirx$Bind` | N/A | `std.ExprBind` |
| `tirx.Assert` | `tirx` | `tirx$Assert` | `__assert__` | `std.Stmt` |
| `tirx.Store` | `tirx` | `tirx$Store` | `__store__` | `std.Store` |
| `tirx.DeclBuffer` | `tirx` | `tirx$DeclBuffer` | N/A | `std.VarDef` |
| `tirx.AllocBuffer` | `tirx` | `tirx$AllocBuffer` | N/A | `std.VarDef` |
| `tirx.Evaluate` | `tirx` | `tirx$Evaluate` | N/A | `std.Stmt` |
| `tirx.For` | `tirx` | `tirx$For` | N/A | `std.For` |
| `tirx.PrimFunc` | `tirx` | `tirx$PrimFunc` | N/A | `std.Func` |

## Relax Dialect

### Types

Relax struct info should be consolidated into `std.Ty`-like nodes.

| Node | $dialect | $mnemonic | $generics | $kind |
|---|---:|---:|---|---|
| `relax.ShapeTy` | `relax` | `relax$ShapeTy` | N/A | `std.Ty` |
| `relax.ObjectTy` | `relax` | `relax$ObjectTy` | N/A | `std.Ty` |
| `relax.PackedFunc` | `relax` | `relax$PackedFunc` | N/A | `std.Ty` |

### Expressions

| Node | $dialect | $mnemonic | $generics | $kind |
|---|---:|---:|---|---|
| `relax.TupleExpr` | `relax` | `relax$TupleExpr` | N/A | `std.Expr` |
| `relax.ShapeExpr` | `relax` | `relax$ShapeExpr` | N/A | `std.Expr` |
| `relax.Var` | `relax` | `relax$Var` | N/A | `std.Var` |
| `relax.DataflowVar` | `relax` | `relax$DataflowVar` | N/A | `relax.Var` |
| `relax.Constant` | `relax` | `relax$Constant` | N/A | `std.Expr` |

## Dialect And Mnemonic System

Every concrete IR node belongs to a `$dialect` and has a globally unique
`$mnemonic`.  The mnemonic is spelled as `{dialect}${name}`.  For example,
`std.Add` has `$dialect = "std"` and `$mnemonic = "std$Add"`.

`$generics` is separate from `$mnemonic`.  It is not a rendered text form; it is
a non-unique sugar identifier that says a node may participate in a syntax
family.  For example, `std.Add` has `$generics = "__add__"`, which lets a
printer or parser sugar `std.Add(a, b)` as `a + b`.  Different nodes, possibly
from different dialects, may share the same `$generics` string when they share
the same sugar family.

The three concepts have distinct roles:

- `$dialect` groups nodes by language, for example `std`, `tirx`, or `relax`.
- `$mnemonic` identifies a concrete node globally, for example `std$Add`.
- `$generics` identifies optional sugar families, for example `__add__`.

Every concrete IR node should register a TypeAttrColumn entry for its fully
qualified mnemonic.  The intended Python spelling is:

```python
class Add(Expr):
    __ffi_mnemonic__: ClassVar[str] = "std$Add"
```

The equivalent C++ registration should use `.def_type_attr(...)` on the object
definition:

```cpp
refl::ObjectDef<AddObj>()
    .def_type_attr(refl::type_attr::kMnemonic, String("std$Add"));
```

The exact TypeAttrColumn name should be `__ffi_mnemonic__`.  The mnemonic is
metadata, not a reflected field, so it is not part of structural equality,
hashing, serialization payloads, or constructor signatures.  Abstract base
classes may omit it; concrete IR nodes should define it.

This metadata is separate from the C++/Python type key.  A type key such as
`ffi.std.Add` remains the FFI runtime identity.  The mnemonic `std$Add` is the
IR-facing identity used by printers, parsers, sugar dispatch, and cross-dialect
reasoning.

`PrinterConfigObj` and `PrinterConfig` should gain a map that controls how
dialects and individual mnemonics are printed.  A tentative field name is:

```cpp
Dict<String, String> dialect_print_map;
```

Python should expose the same field as a mapping from `str` to `str`.

Keys may be either a dialect name or a fully qualified mnemonic:

| Key form | Example | Meaning |
|---|---|---|
| dialect | `std` | Applies to all nodes whose mnemonic starts with `std$` |
| full mnemonic | `std$Add` | Applies only to that concrete node |

Full-mnemonic entries take precedence over dialect entries.  If neither entry is
present, the printer uses the dialect name as the prefix.

Values define the printed callee name:

| Config entry | Meaning | Example unsugared output |
|---|---|---|
| `{ "std": "*" }` | Drop the `std` prefix for every std node | `Add(...)` |
| `{ "std": "core" }` | Print the `std` dialect as `core` | `core.Add(...)` |
| `{ "std$Add": "*" }` | Drop the prefix only for `std$Add` | `Add(...)` |
| `{ "std$Add": "std.MyAdd" }` | Print `std$Add` with an explicit full name | `std.MyAdd(...)` |

The `*` value means "print without a dialect/module prefix", not "use the
generic sugar".  Sugar selection still comes from `$generics` and the node's
registered text printer.  For example, a printer may still emit `a + b` for
`std$Add` because of `__add__`; if it falls back to call syntax, the
`dialect_print_map` decides whether that fallback is `std.Add(a, b)`,
`Add(a, b)`, `core.Add(a, b)`, or `std.MyAdd(a, b)`.
