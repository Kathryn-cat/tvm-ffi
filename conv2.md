# Parser Design Discussion — Conversation 2

Background: The problem is clear — the parser constructs user-defined IR
(`@py_class` nodes like `tir.PrimFunc`, `tir.For`) from TVM-FFI AST nodes
(`tast.Function`, `tast.For`). This conversation explores the approach for
doing so, and compares with the existing parser system in TVM/TIR.

---

## How the Parser Finds the Mapping (AST → IR)

### The Core Principle: Evaluate First, the Value Tells You

The parser never has a static mapping from AST types to IR types. It
discovers the mapping dynamically by evaluating expressions within the AST.

The AST node type only determines **where to look**. The **evaluated
value** determines the IR:

```
AST node type       Which expression to evaluate       Value tells parser what to construct
─────────────       ────────────────────────────       ────────────────────────────────────
tast.Function  →    evaluate the DECORATOR             → surface obj knows to build PrimFunc
tast.For       →    evaluate the ITER expression        → surface obj knows to build tir.For
tast.With      →    evaluate the CONTEXT expression     → surface obj knows to build SBlock
tast.Assign    →    evaluate the RHS                    → surface obj OR plain IR value
```

### 4 Mapping Mechanisms

```
Mechanism                  When used                      Who provides mapping
─────────                  ─────────                      ────────────────────
1. Value-driven dispatch   Statements: for, with,         Surface objects on
   (surface objects)       function, class, assign        language modules (T.serial,
                                                          T.prim_func, T.block, ...)

2. Operator overloading    Expressions: a+b, a[i],        __add__, __getitem__, etc.
                           -x, a & b                      on IR types (PrimExpr, Buffer)

3. Direct call evaluation  Expressions: T.float32(1),     Callable entries on language
                           T.Buffer((4,), "f32")          modules (T.float32, T.Buffer)

4. Structural rules        Subscript store: B[i]=val,     SyntaxContext set by the
   (SyntaxContext)         if/while/return/assert          enclosing function's surface obj
```

The parser framework (TVM-FFI, my job) provides the machinery: evaluate
expressions, dispatch to surface objects, apply operators, delegate to
SyntaxContext.

The user's compiler provides the content: surface objects, operator
overloading, language module entries, SyntaxContext implementations.

### Trace Through the TIR PrimFunc Example

Using the example from conv1.md:

```python
@T.prim_func
def func(A: T.Buffer((4,), "float32"), B: T.Buffer((4,), "float32")):
    for i in T.serial(4):
        B[i] = A[i] + T.float32(1)
```

**Step 1: `tast.Function` → evaluate decorator**

```
eval_expr(Attr(Id("T"), "prim_func"))
  → eval Id("T")             → T language module (from lang_modules dict)
  → getattr(T, "prim_func")  → _PrimFuncDecorator surface object
  → has __ffi_text_parse__   → dispatch: parse_function(parser, node)
  → constructs PrimFunc(...)
```

Who provided the mapping? TIR registered `T.prim_func` as a surface
object that constructs `PrimFunc`.

**Step 2: Inside `parse_function`, process args**

```
arg = Id("A", annotation=Call(T.Buffer, [(4,), "float32"]))

eval_expr(annotation)
  → eval Attr(Id("T"), "Buffer")   → T.Buffer (a proxy/factory)
  → call T.Buffer((4,), "float32") → returns a tir.Buffer object
  → create param Var + buffer_map entry
```

Who provided the mapping? `T.Buffer` is a callable registered by TIR.

**Step 3: `tast.For` → evaluate iter expression**

```
eval_expr(Call(Attr(Id("T"), "serial"), [Lit(4)]))
  → eval Attr(Id("T"), "serial")  → T.serial (surface object factory)
  → call T.serial(4)              → _SerialForIter(end=4) instance
  → dispatch: parse_for(parser, node)
  → constructs tir.For(loop_var=Var("i"), min=0, extent=4, kind=SERIAL, ...)
```

**Step 4: `tast.Assign` with subscript LHS — structural recognition**

```
Assign(targets=[Index(Id("B"), [Id("i")])], value=...)

Parser recognizes subscript on LHS:
  → evaluate Id("B") → var_table["B"] → tir.Buffer object
  → SyntaxContext or Buffer.__setitem__ handles Store construction
  → constructs BufferStore(buffer=B, indices=[i], value=...)
```

**Step 5: `tast.Operation(kAdd)` — operator overloading**

```
eval_expr(Operation(kAdd, lhs=Index(Id("A"),[Id("i")]), rhs=Call(T.float32,[1])))
  → lhs: eval Index(Id("A"), [Id("i")])
         → Buffer.__getitem__(Var("i")) → BufferLoad(buffer=A, indices=[i])
  → rhs: eval Call(T.float32, [1])
         → T.float32(1) → FloatImm(1.0, "float32")
  → lhs + rhs
         → PrimExpr.__add__ → Add(a=BufferLoad(...), b=FloatImm(...))
```

---

## Comparison: Existing TVM/TIR Parser vs v4 Design

### Existing TVM/TIR Parser Architecture

```
① Python source
      ↓  ast.parse()
② Python stdlib ast
      ↓  doc.to_doc()
③' "doc" AST (custom mirror of Python ast)     ← NOT the same as TVM-FFI AST
      ↓  Parser + token-based dispatch
④ TIR IR (via IRBuilder context managers)
```

Key files in `/home/kathrync/work/tir/python/tvm/script/parser/`:
- `core/parser.py` — core Parser class with visitor
- `core/dispatch.py` — token-based dispatch table
- `core/doc.py` / `core/doc_core.py` — the "doc" AST
- `core/evaluator.py` — expression evaluator
- `tir/parser.py` — TIR-specific handlers registered in dispatch table

### v4 Design Architecture

```
① Python source
      ↓  ast.parse()
② Python stdlib ast
      ↓  ast_translate()
③ TVM-FFI AST (@c_class FFI objects)           ← different from "doc" AST
      ↓  IRParser + value-driven dispatch
④ User IR (@py_class nodes)
```

### Difference 1: Dispatch — Token-Based vs Value-Driven

**TVM/TIR (current)**: dispatch key is `(token, ast_node_type)` — static
lookup table. Each dialect registers handlers per AST node type.

```python
# TIR registers handlers per AST node type:
@dispatch.register(token="tir", type_name="For")
def visit_for(self: Parser, node: doc.For) -> None:
    for_frame = self.eval_expr(node.iter)
    with for_frame:
        self.visit_body(node.body)

@dispatch.register(token="tir", type_name="FunctionDef")
def visit_function_def(self: Parser, node: doc.FunctionDef) -> None:
    ...

@dispatch.register(token="tir", type_name="Assign")
def visit_assign(self: Parser, node: doc.Assign) -> None:
    ...
```

The parser switches token when entering a function:
```python
# @T.prim_func sets dispatch_token="tir" on the function
# Parser reads the token and switches:
with self.with_dispatch_token("tir"):
    self.visit_body(node.body)
```

**v4 (new design)**: dispatch key is the evaluated value itself. The
parser is generic — no per-dialect registration:

```python
def visit_for(self, node):
    iter_val = self.eval_expr(node.iter)     # eval T.serial(4) → surface obj
    return self._dispatch(iter_val, node)    # surface obj tells parser what to do

def visit_function(self, node):
    for dec in node.decorators:
        dec_val = self.eval_expr(dec)        # eval T.prim_func → surface obj
        if r := self._try_dispatch(dec_val, node):
            return r
```

**Implication**: In TVM/TIR, the parser must register `visit_for` for
`(tir, For)`, `(relax, For)`, `(default, For)` separately. In v4, there's
ONE `visit_for` — the surface object (returned by `T.serial` vs
`R.something`) carries the dialect-specific behavior.

### Difference 2: IR Construction — IRBuilder vs Surface Objects

**TVM/TIR (current)**: `T.serial(128)` returns a `ForFrame` (context
manager). The parser uses `with`:

```python
# In tir/parser.py visit_for:
for_frame = self.eval_expr(node.iter)   # T.serial(128) → ForFrame
with for_frame as iters:                # ForFrame.__enter__ → pushes frame
    self.visit_body(node.body)          # IRBuilder accumulates statements
# ForFrame.__exit__ → pops frame → constructs tir.For node
```

The IR node is built inside the context manager's `__exit__`, not by the
parser directly. The IRBuilder is a separate stateful object that
accumulates IR.

**v4 (new design)**: `T.serial(4)` returns a surface object. The surface
object constructs the IR directly:

```python
# In _SerialForIter.parse_for:
def parse_for(self, parser, node):
    var = tir.Var(name=node.target.name)
    parser.var_table.define(node.target.name, var)
    body = parser.visit_body(node.body)
    return tir.For(loop_var=var, min=self._start, ...)   # direct construction
```

No IRBuilder, no context managers. The surface object takes the AST node
+ its captured args and returns the IR node directly.

### Difference 3: Intermediate AST

**TVM/TIR (current)**: `doc` AST — plain Python objects that mirror
`ast.*` nodes. A stability layer over Python's stdlib `ast` (which
changes between Python versions).

```python
# doc_core.py: plain Python classes, NOT FFI objects
class For:
    target: ...
    iter: ...
    body: ...
```

**v4 (new design)**: TVM-FFI AST — `@c_class` objects with reflection,
equality, serialization.

```python
# ast.h / ast.py: FFI objects that can cross language boundaries
@c_class("ffi.text.ast.For")
class For(Stmt):
    target: Expr
    iter: Expr
    body: StmtBlock
```

The TVM-FFI AST can be passed to C++, serialized, structurally compared.
The `doc` AST cannot.

### Side-by-Side Summary

| Aspect | TVM/TIR (current) | v4 (new design) |
|---|---|---|
| Intermediate AST | `doc` AST (plain Python objects) | TVM-FFI AST (`@c_class` FFI objects) |
| Dispatch key | `(token, node_type)` — static table | Evaluated value — dynamic |
| Handler registration | `@dispatch.register(token="tir", type_name="For")` per dialect per node type | Surface objects on language modules |
| How `T.serial(4)` works | Returns `ForFrame` context manager | Returns surface object |
| IR construction | IRBuilder + `with frame:` + `__exit__` builds IR | Surface object's `parse_for` returns IR directly |
| Parser generality | Parser generic, but handlers per-dialect | Parser fully generic, surface objects per-dialect |
| Per-dialect handler count | ~15 per dialect (one per AST node type) | 0 (framework is fully generic) |
| Per-dialect code lives in | `tir/parser.py`, `relax/parser.py` | Surface objects + SyntaxContext (on language modules) |

### What v4 Improves

The core shift: **move dispatch intelligence from the parser into the
values**.

In TVM/TIR, the parser has a handler for `(tir, For)` that knows to eval
`node.iter` and use it as a context manager. A different handler for
`(relax, For)`. Each dialect registers handlers for every AST node type.

In v4, the parser's `visit_for` is dialect-agnostic — it evaluates
`node.iter` and dispatches to whatever comes back. The dialect-specific
knowledge is in the surface object. This means:
- **No per-dialect parser handlers** — traits auto-generate surface objects
- **New dialects need zero parser code** for L0/L1 nodes
- **Parser framework works for any `@py_class` IR**, not just TIR/Relax

### What's Preserved

Both systems share the same fundamental flow:
1. Parse Python source to an intermediate AST
2. Evaluate key expressions (`T.serial(4)`, `T.prim_func`) to get
   dispatch targets
3. Use the dispatch target to control IR construction
4. Manage variable scoping with a frame stack

The expression evaluation, variable tables, and scoping concepts are
essentially the same — just the dispatch mechanism and IR construction
style change.

---

## Boundary Recap (from conv1.md)

### What TVM-FFI Provides vs What Users Provide

**TVM-FFI provides (this repo):**

Already implemented:
- `Object` — universal base class for all FFI objects
- `@py_class` / `@c_class` — type registration decorators
- TVM-FFI AST nodes — full Python 3.9+ syntax coverage
- `ast_translate` — Python source → TVM-FFI AST
- `PythonDocPrinter` — TVM-FFI AST → Python source string
- `IRPrinter` — IR → TVM-FFI AST (via `__ir_print__` dispatch)
- `TypeAttrColumn` — per-type attribute dispatch

Not yet implemented (design-doc only):
- `@ffi.traits.*` — trait decorators (developer's next job)
- `$field:` / `$method:` / `$global:` resolution (part of traits)
- IRParser, surface objects, SyntaxContext (**my job**)
- Language modules — `T`, `R`, `I` namespaces (shared)

**User's compiler provides (e.g., TIR):**
- All IR node types via `@py_class` (`tir.PrimFunc`, `tir.For`, etc.)
- IR base class hierarchy (`PrimExpr`, `Stmt` — user-defined, NOT in TVM-FFI)
- Trait decorations on IR types (once traits exist)
- Operator overloading (`PrimExpr.__add__` → `tir.Add`, etc.)
- Language module entries (`T.serial`, `T.prim_func`, `T.block`, etc.)
- SyntaxContext implementations (how `if`/`while`/`return` behave in TIR vs Relax)

TVM-FFI provides exactly ONE base class: `tvm_ffi.Object`. Everything
else (`PrimExpr`, `Stmt`, `BaseFunc`) is user-defined.
