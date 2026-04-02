# Parser Design Discussion — Conversation 0

## Background

### What is TVM-FFI?

TVM FFI is an open ABI and FFI for ML systems. `@tvm_ffi.dataclasses.py_class`
provides structural IR definitions with fields, reflection, equality, and
C++/Rust interop. The project is now adding **printer** and **parser** support
so that IR objects can be rendered as Python-style source and parsed back.

### Design Doc: v4.md

The v4 design doc (`v4.md` in repo root) describes a complete traits + text
format system with:

1. **Semantic traits** — decorators (`@ffi.traits.For`, `@ffi.traits.BinOp`,
   etc.) describing what each IR node IS.
2. **Three-tier printer** — Level 0 (zero-config default from `@py_class`
   fields), Level 1 (trait-driven auto-gen), Level 2 (manual `__ffi_text_print__`
   override).
3. **Parser** — value-driven dispatch via `__ffi_text_parse__` on surface
   objects, `SyntaxContext`, language modules.
4. **Language modules** — `@ffi.lang("T")` namespaces (`T.serial`,
   `R.dataflow`, etc.).
5. **Python-style AST** — intermediate AST (Expr/Stmt hierarchy) rendered to
   Python source.

### v3 → v4 Core Differences

The evolution from v3 to v4 centers on:

| Change | Impact |
|--------|--------|
| **`def_carry`/`carry_init`** on `For` and `With` traits | Models loop-carry and scope-carry values. Promotes `scf.for`+iter_args, `scf.if`+results from L2→L1. |
| **`ElementWise` trait** | New trait for linalg-style `ins/outs { body }` ops. Promotes `linalg.generic` from L2→L1. |
| **`Assert` trait** | Promoted from L2→L1. |
| **`Module` / `Class` split** | Separate traits for compilation units vs class definitions. |
| **Level 1a / 1b distinction** | 1a = pure trait (zero custom code), 1b = trait + `$method:` hooks. |
| **`ctx` convenience methods** | `ctx.call`, `ctx.elide`, `ctx.lambda_`, `ctx.scope`, `ctx.assign` for concise L2 code. |
| **Framework conventions** | Transparent wrappers (e.g. `SBlockRealize`) and compound body flattening (e.g. `SeqExpr`) — zero per-node code. |
| **Optional indices on Load/Store** | `None` for scalar access. |
| **`Value.type` → `Value.type_ann`** | Renamed to avoid shadowing Python `type`. |

Net effect: code reduction claim goes from ~70% (v3) to ~95% (v4). MLIR/cuteDSL
coverage dramatically improved — 540 ops validated, 90% at L0, 11% at L1, <1% at L2.

---

## What's Been Implemented (Latest Two Commits)

### Commit #2: `8f6f236` — Text Printer Foundation

**PR #55**: `feat(text): add text printer module for Python-style IR pretty-printing`

Implements the infrastructure layer:

| Component | Files |
|-----------|-------|
| AST node hierarchy (24 types: `NodeAST > ExprAST / StmtAST`) | `include/tvm/ffi/text/ast.h`, `python/tvm_ffi/text/ast.py` |
| AST → Python string renderer (`PythonDocPrinter`) | `src/text/printer.cc` |
| `IRPrinter` (dispatch, var binding, frame stack) | `include/tvm/ffi/text/printer.h`, `python/tvm_ffi/text/ir_printer.py` |
| `__ir_print__` vtable method on `py_class` | `python/tvm_ffi/dataclasses/py_class.py` (added to `_FFI_RECOGNIZED_METHODS`) |
| FFI function exposure | `python/tvm_ffi/text/_ffi_api.py` |
| String utilities (`String::Split`, `PrintEscapeString`) | `include/tvm/ffi/string.h` |
| Toy IR for testing (`ToyVar`, `ToyAdd`, `ToyAssign`, `ToyFunc`) | `python/tvm_ffi/testing/testing.py` |
| Tests (226 cases) | `tests/python/test_text_printer_ast.py`, `tests/python/test_text_printer_ir_printer.py` |

Implements **Level 0** (reflection-based default) and **Level 2** (`__ir_print__`
manual override) of the v4 printer dispatch:

```
Tier 1:  TypeAttrColumn("__ir_print__")  →  manual override (Level 2)
Tier 3:  (fallback)                      →  default from @py_class fields (Level 0)
```

### Commit #1: `9f087e8` — Python AST Translator

**PR #56**: `feat(text): add Python AST to TVM-FFI AST translator`

Implements the parser's input pipeline:

| Component | Files |
|-----------|-------|
| Python `ast.Module` → TVM-FFI AST converter | `python/tvm_ffi/text/ast_translate.py` |
| Extended AST nodes for full Python 3.9+ | `include/tvm/ffi/text/ast.h` (823 lines added) |
| Printer fixes for roundtrip fidelity | `src/text/printer.cc` (605 lines added) |
| Roundtrip test harness | `addons/ast-testsuit/` |
| Tests (327 translate + 62 roundtrip) | `tests/python/test_text_ast_translate.py` |

Validated ③→①→②→③ roundtrip across 4,275 files (TVM, DKG, GraphIR, tvm-ffi)
with 100% fidelity.

---

## Division of Work

| Who | What | Status |
|-----|------|--------|
| Developer | Low-level plumbing (AST, renderer, IRPrinter, ast_translate) | **Done** (commits #1, #2) |
| Developer | Trait-driven printer (Level 1): `@ffi.traits.*` decorators, `TypeAttrColumn("__ffi_traits__")`, per-trait print functions, `$field:`/`$method:` resolution, `ctx` convenience methods | **Next** |
| Me | Parser: `IRParser`, `eval_expr`, surface objects, `SyntaxContext`, auto-generation from traits, language modules (parser side) | **My job** |
| Shared | Language modules (`T`, `R`), trait data structures | Coordinate |

---

## The 4 Representations in the Full Roundtrip

```
①  Python source string     "for i in T.serial(0, n):\n    body"
②  Python stdlib ast         ast.For(target=ast.Name("i"), iter=ast.Call(...), ...)
③  TVM-FFI text AST          tast.For(target=Id("i"), iter=Call(Attr(Id("T"),"serial"),...), ...)
④  User IR                   TIRFor(loop_var=Var("i"), min=0, extent=n, kind=SERIAL, body=...)
```

| # | Representation | Defined in |
|---|---|---|
| ① | Python source string | `str` |
| ② | Python stdlib `ast` | `/usr/lib/python3.X/ast.py` (stdlib) |
| ③ | TVM-FFI text AST | C++: `include/tvm/ffi/text/ast.h` / Python: `python/tvm_ffi/text/ast.py` |
| ④ | User IR | User-defined via `@py_class` (e.g. `TIRFor`, `ToyAdd`, etc.) |

### Full Pipeline

```
PRINTER DIRECTION (④ → ①):

④ User IR ──────────► ③ TVM-FFI AST ──────────► ① Python source
          IRPrinter                DocToPythonScript
          + __ir_print__           (PythonDocPrinter in C++)

Files:    printer.h                printer.cc
          ir_printer.py


PARSER DIRECTION (① → ④):

① Python source ──► ② Py stdlib AST ──► ③ TVM-FFI AST ──► ④ User IR
               ast.parse()         ast_translate()       IRParser
                                                         + eval_expr
                                                         + surface objects

Files:   (stdlib)             ast_translate.py         TO BE BUILT
```

### Roundtrip Property

```
④  →  ③  →  ①  →  ②  →  ③  →  ④'    where ④ ≡ ④'
IR   AST  string  pyAST  AST   IR'
```

The middle segment (③→①→②→③) is validated by commit #1 across 4,275 files.
The outer loop (④→③ printer, ③→④ parser) is what remains.

---

## Parser Algorithm: Value-Driven Dispatch

### The Core Trick: `text_printer_kind` Drives Both Directions

When a trait says `text_printer_kind="$lang.serial"` (resolving `$lang` to `T`):

- **Printer** emits `T.serial(...)` in the output text
- **Parser** needs `T.serial` to exist as a callable returning a surface object

Python modules are mutable — `setattr(T, "serial", surface_obj_factory)` works
at any time. This is the same pattern as `_ffi_api.py` where `init_ffi_api()`
dynamically populates modules via `setattr()`.

### Hybrid: Auto-Generated + Manual Surface Objects

**Auto-generated** (most cases): The trait metadata on an IR type has enough
information to generate a surface object. When a `For` trait with
`text_printer_kind="$lang.serial"` is registered, the framework generates
`_Auto_TIRFor_Serial` and does `setattr(T, "serial", _Auto_TIRFor_Serial)`.
A `.pyi` stub is generated for IDE support.

**Manual** (Level 2): Hand-written surface objects for irregular cases
(e.g. `IRModuleDecorator` with multi-pass parsing). Registered on the
language module the same way.

Both coexist — the language module `T` is just a bag of attributes.

### Expression Evaluator

The parser has a mini-evaluator running on TVM-FFI AST nodes:

```python
def eval_expr(self, node: tast.Expr) -> Any:
    if isinstance(node, tast.Id):
        # Variable table first (IR variables), then language modules (T, R, I)
        if (v := self.var_table.get(node.name)) is not None:
            return v
        if node.name in self.lang_modules:
            return self.lang_modules[node.name]
        raise ParseError(f"Undefined: {node.name}")

    if isinstance(node, tast.Attr):
        base = self.eval_expr(node.object)
        return getattr(base, node.field)

    if isinstance(node, tast.Call):
        callee = self.eval_expr(node.callee)
        args = [self.eval_expr(a) for a in node.args]
        kwargs = {k: self.eval_expr(v) for k, v in node.kwargs}
        return callee(*args, **kwargs)

    if isinstance(node, tast.Literal):
        return node.value

    if isinstance(node, tast.Operation):
        lhs = self.eval_expr(node.lhs)
        rhs = self.eval_expr(node.rhs)
        # Operator overloading on IR types: lhs + rhs → Add(lhs, rhs)
        return _apply_op(node.op, lhs, rhs)
```

### Statement Visitors

```python
def visit_stmt(self, node: tast.Stmt):
    if isinstance(node, tast.For):
        iter_val = self.eval_expr(node.iter)    # T.serial(0, n) → surface obj
        return self._dispatch(iter_val, node)

    if isinstance(node, tast.With):
        ctx_val = self.eval_expr(node.context_expr)
        return self._dispatch(ctx_val, node)

    if isinstance(node, tast.Assign):
        if self._is_subscript_store(node):
            return self._handle_store(node)
        rhs_val = self.eval_expr(node.rhs)
        return self._dispatch_or_bind(rhs_val, node)

    if isinstance(node, tast.Function):
        for dec in node.decorators:
            dec_val = self.eval_expr(dec)
            if r := self._try_dispatch(dec_val, node):
                return r
        raise ParseError("Unrecognized decorator")

    # Structural → delegate to SyntaxContext
    if isinstance(node, tast.If):
        return self.syntax_ctx.handle_if(self, node)
    if isinstance(node, tast.While):
        return self.syntax_ctx.handle_while(self, node)
    if isinstance(node, tast.Return):
        return self.syntax_ctx.handle_return(self, node)
    if isinstance(node, tast.Assert):
        return self.syntax_ctx.handle_assert(self, node)

def _dispatch(self, val, node):
    if hasattr(val, "__ffi_text_parse__"):
        return val.__ffi_text_parse__(self, node)
    raise ParseError(f"Expected IR construct, got {type(val)}")
```

---

## Parser Levels (Mirror the Printer)

### Level 0: Parse `TypeKey(field=value, ...)`

Inverse of L0 printing. Works for every `@py_class` with zero per-type code.

```
Printed:    testing.text.toy_ir.Add(lhs=<lhs>, rhs=<rhs>)
TVM-FFI AST: Call(callee=Attr(..., "Add"), kwargs={"lhs": ..., "rhs": ...})
```

`eval_expr` evaluates the Call: resolves `testing.text.toy_ir.Add` to the
class via type registry, evaluates kwargs recursively, calls the class
constructor. Pure constructor-call evaluation.

### Level 1: Parse Trait-Driven Syntax

Inverse of L1 printing. Split into two mechanisms:

**Expressions** (BinOp, UnaryOp, Load): handled by `eval_expr` + operator
overloading on IR types. `a + b` where `a`, `b` are IR `Var` objects →
`Var.__add__` returns `Add(a, b)`. No surface object needed.

**Statements** (For, With, Func, Assign): handled by value-driven dispatch
to auto-generated surface objects.

| Printed pattern | Parser mechanism |
|---|---|
| `a + b` | `eval_expr(Operation(kAdd))` → operator overloading |
| `for i in T.serial(0, n):` | `eval_expr(T.serial(0, n))` → surface obj → `parse_for` |
| `@T.prim_func def f():` | `eval_expr(T.prim_func)` → surface obj → `parse_function` |
| `with T.block("x") as v:` | `eval_expr(T.block("x"))` → surface obj → `parse_with` |
| `x = rhs` | `eval_expr(rhs)` → IR value → default variable binding |
| `buf[i] = val` | Subscript LHS → recognized as Store pattern |

### Level 2: Manual Surface Objects

Hand-written `parse_*` methods. Same dispatch mechanism, custom logic.

### Summary

| Level | Printer | Parser | Per-type code |
|---|---|---|---|
| L0 | `TypeKey(field=val)` from reflection | `eval_expr` evaluates constructor call | **0 lines** both |
| L1 expr | Trait → AST `Operation`/`Index` nodes | `eval_expr` + operator overloading on IR types | **0 lines** parser |
| L1 stmt | Trait → AST `For`/`With`/`Function` | `eval_expr` → surface object → auto-gen `parse_*` | **0 lines** parser |
| L2 | Manual `__ir_print__` | Manual surface object `parse_*` | **Custom** both |

---

## Concrete Trace: Parsing `@Toy.func def my_func(a, b): ...`

### The IR being parsed into

```python
# python/tvm_ffi/testing/testing.py
@py_class("testing.text.toy_ir.Func")
class ToyFunc(ToyNode):
    name: str
    args: list[ToyVar]
    body: list[ToyAssign]
    ret: ToyExpr
```

### Printed form (with decorator)

```python
@Toy.func
def my_func(a, b):
    c = a + b
    return c
```

### TVM-FFI AST ③ (after ast_translate)

```
Function(
  name = "my_func",
  args = [Id("a"), Id("b")],
  body = StmtBlock([
    Assign(targets=[Id("c")], value=Operation(kAdd, lhs=Id("a"), rhs=Id("b"))),
    Return(value=Id("c"))
  ]),
  decorators = [Attr(Id("Toy"), "func")]
)
```

### Parser trace

```
1. visit_stmt(Function node)
   → for dec in [Attr(Id("Toy"), "func")]:
       dec_val = eval_expr(Attr(Id("Toy"), "func"))
         → eval_expr(Id("Toy"))          →  Toy language module
         → getattr(Toy, "func")          →  _ToyFuncDecorator (surface object)
       _try_dispatch(dec_val, node)
         → dec_val.__ffi_text_parse__(parser, function_node)
           → isinstance(node, tast.Function) → True
           → self.parse_function(parser, node)

2. Inside parse_function:
   → push new var_table frame
   → for param "a": create ToyVar("a"), define in var_table
   → for param "b": create ToyVar("b"), define in var_table
   → parse body:
       visit_stmt(Assign(targets=[Id("c")], value=Operation(kAdd, Id("a"), Id("b"))))
         → eval_expr(Operation(kAdd, Id("a"), Id("b")))
             → lhs = eval_expr(Id("a"))  → var_table["a"] → ToyVar("a")
             → rhs = eval_expr(Id("b"))  → var_table["b"] → ToyVar("b")
             → lhs + rhs  (operator overloading) → ToyAdd(ToyVar("a"), ToyVar("b"))
         → define "c" → ToyVar("c")
         → ToyAssign(var=ToyVar("c"), value=ToyAdd(...))
       visit_stmt(Return(Id("c")))
         → eval_expr(Id("c")) → var_table["c"] → ToyVar("c")
         → return value = ToyVar("c")
   → pop var_table frame
   → return ToyFunc(name="my_func", args=[ToyVar("a"), ToyVar("b")],
                     body=[ToyAssign(...)], ret=ToyVar("c"))
```

Result: ④ `ToyFunc` IR object reconstructed from ③ TVM-FFI AST.

---

## Multi-Pass Parsing

Multi-pass is a **Level 2** concern for scopes with forward references
(primarily module-level):

```python
@I.ir_module
class MyModule:
    def func_a(x):
        y = func_b(x)      # func_b not yet defined in single-pass!
        return y
    def func_b(x):
        return x + 1
```

The `I.ir_module` surface object handles this:

```python
class _IRModuleDecorator:
    def parse_class(self, parser, node):
        # Pass 1: forward-declare all functions
        for stmt in node.body:
            if isinstance(stmt, tast.Function):
                gvar = GlobalVar(stmt.name)
                parser.var_table.define(stmt.name, gvar)  # "func_b" now exists

        # Pass 2: parse function bodies (forward refs resolve)
        funcs = {}
        for stmt in node.body:
            if isinstance(stmt, tast.Function):
                funcs[stmt.name] = parser.visit_stmt(stmt)  # func_b known ✓

        return IRModule(funcs=funcs)
```

Only scopes where **siblings reference each other** need multi-pass.
For-loops, with-blocks, if-statements, function bodies are all single-pass
(references flow top-down). Multi-pass is a surface-object concern, not
a framework feature.

---

## Open Questions

1. **Where does auto-generation happen?** At trait registration time (when
   `@ffi.traits.For(...)` runs) or lazily? Registration time seems cleaner.

2. **How does `$method:` reverse?** Printer calls `$method:computed_end`
   (= `min + extent`). Parser receives `(start, end)` as call args. For
   standard patterns (range convention) the inversion is known. For arbitrary
   `$method:`, inversion must be specified manually — the hybrid approach.

3. **`eval_expr` scope**: Language modules + variables defined so far +
   type registry (for L0 `TypeKey(...)` calls). What else?

4. **Surface object lifetime**: `T.serial` is a class (factory),
   `T.serial(0, n)` creates an ephemeral instance used for dispatch then
   discarded.
