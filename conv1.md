# Parser Design Discussion — Conversation 1

## What Are Surface Objects?

### The Split-Information Problem

When the parser sees `for i in T.serial(0, 10): body`, the information
needed to construct the IR node (`ForLoop`) is split across three AST
positions:

```
For(
  target = Id("i"),                              ← provides: loop variable
  iter   = Call(Id("range"), [Lit(0), Lit(10)]), ← provides: start, end
  body   = StmtBlock([...])                      ← provides: body
)
```

The parser processes `iter` first (to decide what kind of construct this
is). At that moment, `eval_expr(T.serial(0, 10))` runs. But we can't
construct the IR yet — we don't have `loop_var` or `body`.

### What a Surface Object Is

A surface object is the return value of `eval_expr(T.serial(0, 10))`.
It captures the call arguments and waits for the structural AST context:

```python
class _RangeSurface:
    """Result of evaluating T.serial(0, 10) at parse time."""
    def __init__(self, start, end):
        self.start = start    # captured from call args
        self.end = end        # captured from call args

    def __ffi_text_parse__(self, parser, node):
        return self.parse_for(parser, node)

    def parse_for(self, parser, node):
        # NOW we have both:
        #   - call args (self.start, self.end) — from eval_expr
        #   - AST context (node.target, node.body) — from the For node
        var = Var(name=node.target.name)
        parser.var_table.define(node.target.name, var)
        body = parser.visit_body(node.body)
        return ForLoop(var=var, start=self.start, end=self.end, body=body)
```

**In one sentence**: a surface object is a deferred IR constructor that
captures call-site arguments and waits for the structural AST context
to complete the construction.

### Why Not Alternatives?


| Alternative                  | Problem                                                                                             |
| ---------------------------- | --------------------------------------------------------------------------------------------------- |
| Return IR directly from eval | Can't — `var` and `body` not available yet. `@py_class` objects are constructed whole.              |
| Return a plain dict          | Dict has no behavior. Need separate registry mapping dicts→constructors. Lose position-specificity. |
| Return a closure             | No position-specific dispatch — can't distinguish for vs with context. Can't introspect or compose. |
| Pattern-match on callee name | Name-based, not value-based. Breaks with `my_serial = T.serial; for i in my_serial(0, 10):`         |


### Position-Specific Dispatch

The same surface object can handle different AST positions:

```python
class SurfaceObject:
    def __ffi_text_parse__(self, parser, node):
        if isinstance(node, tast.For):       return self.parse_for(parser, node)
        elif isinstance(node, tast.With):    return self.parse_with(parser, node)
        elif isinstance(node, tast.Function):return self.parse_function(parser, node)
        elif isinstance(node, tast.Class):   return self.parse_class(parser, node)
        elif isinstance(node, tast.Assign):  return self.parse_assign(parser, node)
        raise ParseError(f"Unexpected position for {type(self)}")
```

---

## How Auto-Generation Works (Trait → Surface Object)

### Two Levels of Mapping

**Level A — Hardcoded per trait type** (the template):
The framework knows, for each trait type, which AST positions map to
which trait parameters. For a `For` trait:

```
AST position        → trait parameter
────────────          ───────────────
call arg 0          → start
call arg 1          → end
call kwarg "step"   → step
node.target         → region.def_values
node.body           → region.body
callee name         → text_printer_kind
```

This is the same for ALL `For` traits.

**Level B — Per-type from decorator** (the field mapping):
The trait decorator says which trait parameters map to which IR fields:

```
Trait parameter      → IR field
───────────────        ────────
start                → "$field:start"   → ForLoop.start
end                  → "$field:end"     → ForLoop.end
region.def_values    → "$field:var"     → ForLoop.var
region.body          → "$field:body"    → ForLoop.body
```

### Chaining Both Levels

The auto-generator chains: **AST position → trait parameter → IR field**:

```
AST position    →  trait parameter     →  IR field
────────────       ───────────────        ────────
call arg 0     →   start              →  ForLoop.start
call arg 1     →   end                →  ForLoop.end
node.target    →   region.def_values  →  ForLoop.var
node.body      →   region.body        →  ForLoop.body
```

From this table, the framework generates the surface object:

```python
class _AutoGen_ForLoop(SurfaceObject):
    def __init__(self, *args, **kwargs):
        # From Level A: For template unpacks range convention
        if len(args) == 1:
            self._start, self._end = 0, args[0]
        elif len(args) == 2:
            self._start, self._end = args[0], args[1]
        self._step = kwargs.get("step", 1)

    def parse_for(self, parser, node):
        # From Level A + B chained:
        # node.target → region.def_values → "$field:var"
        var = Var(name=node.target.name)
        parser.var_table.define(node.target.name, var)
        # node.body → region.body → "$field:body"
        body = parser.visit_body(node.body)
        # call args → start/end → "$field:start"/"$field:end"
        return ForLoop(var=var, start=self._start, end=self._end, body=body)
```

### The `$method:` Complication

`$field:` reversal is trivial — pass value to constructor.
`$method:` is a one-way transform:

```python
@ffi.traits.For(end="$method:computed_end", ...)
class TIRFor:
    min: int; extent: int
    @property
    def computed_end(self): return self.min + self.extent
```

Printer outputs `end` (= min + extent). Parser receives `(start, end)`.
IR constructor needs `(min, extent)`. **The parser can't auto-invert.**

Solution — manual inverse in the language module entry:

```python
T.serial = ffi.lang.for_iter(
    TIRFor, kind="serial",
    construct=lambda start, end, step, loop_var, body: TIRFor(
        loop_var=loop_var, min=start,
        extent=end - start,    # manual inverse
        step=step, kind=ForKind.SERIAL, body=body,
    ),
)
```

Pure `$field:` traits → fully auto-generated. Traits with `$method:` →
auto-generated template + manual `construct` function.

---

## Traits: Setup-Time vs Parse-Time

### Traits Are NOT Consulted at Parse Time

Traits are decorations on `@py_class` IR types. They live on the IR type
definitions. They are NOT in the printed text. They are NOT in the TVM-FFI
AST.

**Timeline:**

```
SETUP TIME (before any parsing happens)
═══════════════════════════════════════

1. User defines IR types with traits:
   @ffi.traits.For(...) class ForLoop: ...
   @ffi.traits.BinOp(op="+") class Add: ...

2. PRINTER side: traits stored in TypeAttrColumn.
   Printer reads them to decide how to render each IR type.

3. PARSER side: traits consumed to GENERATE artifacts:
   For trait on ForLoop → generate surface object → register as Mini.loop
   BinOp trait on Add   → IR base type defines __add__ returning Add

   Artifacts:
   - Language module entries (surface objects)
   - Operator overloading methods (__add__, __sub__, etc.) on IR types


PARSE TIME (actually parsing text)
═══════════════════════════════════

Parser uses the ARTIFACTS, not traits directly:

  "for i in Mini.loop(0, 10): ..."
       → eval Mini.loop(0, 10)  → surface object  → ForLoop(...)

  "a + b"
       → eval_expr: lhs=Var("a"), rhs=Var("b")
       → lhs + rhs  → Var.__add__  → Add(lhs=a, rhs=b)
```

### The Two Parse Paths


| What's being parsed           | How it works                                   | Traits involved?                      |
| ----------------------------- | ---------------------------------------------- | ------------------------------------- |
| `a + b` (expression)          | `eval_expr` → operator overloading (`__add__`) | No. `__add__` defined at setup        |
| `T.serial(0, 10)` in for-loop | `eval_expr` → surface object → `parse_for`     | No. Surface object generated at setup |
| `tir.Add(a=..., b=...)` (L0)  | `eval_expr` → class constructor call           | No. Just `Add(a=..., b=...)`          |


---

## Concrete Example: TVM TIR PrimFunc

### ① Printed Python String

```python
@T.prim_func
def func(A: T.Buffer((4,), "float32"), B: T.Buffer((4,), "float32")):
    for i in T.serial(4):
        B[i] = A[i] + T.float32(1)
```

### ③ TVM-FFI AST (After `ast_translate`)

Pure syntax — no IR knowledge, no traits, no types. Defined in
`include/tvm/ffi/text/ast.h` and `python/tvm_ffi/text/ast.py`.

```python
Function(
    name="func",
    args=[
        # A: T.Buffer((4,), "float32")
        Id("A", annotation=Call(Attr(Id("T"), "Buffer"),
                                [Tuple([Lit(4)]), Lit("float32")])),
        # B: T.Buffer((4,), "float32")
        Id("B", annotation=Call(Attr(Id("T"), "Buffer"),
                                [Tuple([Lit(4)]), Lit("float32")])),
    ],
    body=StmtBlock([
        # for i in T.serial(4):
        For(
            target=Id("i"),
            iter=Call(Attr(Id("T"), "serial"), [Lit(4)]),
            body=StmtBlock([
                # B[i] = A[i] + T.float32(1)
                Assign(
                    targets=[Index(Id("B"), [Id("i")])],
                    value=Operation(kAdd,
                        lhs=Index(Id("A"), [Id("i")]),
                        rhs=Call(Attr(Id("T"), "float32"), [Lit(1)])
                    )
                )
            ])
        )
    ]),
    decorators=[Attr(Id("T"), "prim_func")],
)
```

Every node is a `tast.*` type: `Id`, `Call`, `Attr`, `Lit`, `For`,
`Function`, `Assign`, `Operation`, `Index`, `StmtBlock`, `Tuple`.
These are syntax nodes. They know nothing about TIR.

### ④ User IR (py_class TIR Nodes — Parse Target)

Each type has traits (skeleton). Defined by the TIR dialect author via
`@py_class`. Traits are NOT yet implemented but shown per v4 design.

```python
# --- Expressions ---

@py_class("tir.Var")
@ffi.traits.Value(name="$field:name_hint", type_ann="$method:printed_type")
class Var(PrimExpr):
    name_hint: str
    dtype: DataType

@py_class("tir.Add")
@ffi.traits.BinOp(lhs="$field:a", rhs="$field:b", op="+")
class Add(PrimExpr):
    a: PrimExpr
    b: PrimExpr

@py_class("tir.BufferLoad")
@ffi.traits.Load(source="$field:buffer", indices="$field:indices")
class BufferLoad(PrimExpr):
    buffer: Buffer
    indices: list[PrimExpr]

@py_class("tir.FloatImm")                    # L0 — no trait
class FloatImm(PrimExpr):
    value: float
    dtype: DataType

@py_class("tir.IntImm")                      # L0 — no trait
class IntImm(PrimExpr):
    value: int
    dtype: DataType

# --- Statements ---

@py_class("tir.BufferStore")
@ffi.traits.Store(target="$field:buffer", value="$field:value",
                   indices="$field:indices")
class BufferStore(Stmt):
    buffer: Buffer
    value: PrimExpr
    indices: list[PrimExpr]

@py_class("tir.For")
@ffi.traits.For(
    region=Region(def_values="$field:loop_var", body="$field:body"),
    start="$field:min",
    end="$method:computed_end",
    step="$field:step",
    text_printer_kind="$method:kind_prefix",
)
class For(Stmt):
    loop_var: Var
    min: PrimExpr
    extent: PrimExpr
    kind: ForKind
    body: Stmt
    step: PrimExpr
    annotations: dict[str, Object]

    @property
    def computed_end(self):
        return self.min + self.extent

    @property
    def kind_prefix(self):
        return {ForKind.SERIAL: "T.serial", ...}[self.kind]

# --- Function ---

@py_class("tir.PrimFunc")
@ffi.traits.Func(
    symbol="$field:name",
    region=Region(def_values="$field:params", body="$field:body"),
    text_printer_kind="T.prim_func",
    text_printer_pre="$method:print_prologue",PrimFunc
)
class PrimFunc(BaseFunc):
    params: list[Var]
    body: Stmt
    ret_type: Type
    buffer_map: dict[Var, Buffer]
    attrs: dict[str, Object]

    def print_prologue(self, ctx, frame):
        # emit T.func_attr, T.match_buffer, etc.
        ...

# --- Buffer (L0 as type annotation, L2 for complex cases) ---

@py_class("tir.Buffer")
class Buffer(Object):
    name: str
    shape: list[PrimExpr]
    dtype: DataType
    # ... many more fields
```

The actual IR tree for our example:

```python
PrimFunc(
    params=[Var("A", "handle"), Var("B", "handle")],
    buffer_map={
        Var("A"): Buffer(name="A", shape=[IntImm(4)], dtype="float32"),
        Var("B"): Buffer(name="B", shape=[IntImm(4)], dtype="float32"),
    },
    body=For(
        loop_var=Var("i", "int32"),
        min=IntImm(0),
        extent=IntImm(4),
        kind=ForKind.SERIAL,
        step=IntImm(1),
        body=BufferStore(
            buffer=Buffer("B", ...),
            value=Add(
                a=BufferLoad(buffer=Buffer("A", ...), indices=[Var("i")]),
                b=FloatImm(1.0, "float32"),
            ),
            indices=[Var("i")],
        ),
    ),
)
```

### Side-by-Side: ③ Syntax vs ④ Semantics

```
③ TVM-FFI AST (syntax)              ④ TIR IR (semantics)
══════════════════════               ════════════════════

Function(                            PrimFunc(
  decorators=[T.prim_func]  ───►       (dispatches to PrimFunc surface obj)
  name="func"               ───►       name="func"
  args=[Id("A",ann=..),     ───►       params=[Var("A"), Var("B")]
        Id("B",ann=..)]                buffer_map={Var("A"):Buffer(...), ...}
  body=                                body=
    For(                                 For(
      target=Id("i")         ───►          loop_var=Var("i","int32")
      iter=T.serial(4)       ───►          min=0, extent=4, kind=SERIAL
      body=                                body=
        Assign(                              BufferStore(
          targets=[B[i]]     ───►              buffer=Buffer("B"), indices=[Var("i")]
          value=A[i]+T.f32(1)───►              value=Add(
                                                 a=BufferLoad(Buffer("A"),[Var("i")]),
                                                 b=FloatImm(1.0,"float32"))
        )                                    )
    )                                    )
)                                    )
```

### Key Transformations (③ → ④)


| ③ TVM-FFI AST                            | ④ TIR IR                             | How parser handles it                                        |
| ---------------------------------------- | ------------------------------------ | ------------------------------------------------------------ |
| `decorators=[T.prim_func]`               | `PrimFunc(...)`                      | eval `T.prim_func` → surface obj → `parse_function`          |
| `Id("A", ann=T.Buffer(...))`             | `Var("A")` + `buffer_map` entry      | Surface obj parses annotations → buffer_map                  |
| `Call(T.serial, [4])`                    | `min=0, extent=4, kind=SERIAL`       | eval `T.serial(4)` → surface obj → `parse_for`               |
| `Index(Id("B"),[Id("i")])` on Assign LHS | `BufferStore(buffer=B, indices=[i])` | Subscript on LHS → parser recognizes as Store                |
| `Index(Id("A"),[Id("i")])` in expression | `BufferLoad(buffer=A, indices=[i])`  | eval subscript on Buffer → operator overloading → BufferLoad |
| `Operation(kAdd, lhs, rhs)`              | `Add(a=lhs, b=rhs)`                  | `PrimExpr.__add_`_ → returns Add                             |
| `Call(T.float32, [1])`                   | `FloatImm(1.0, "float32")`           | eval `T.float32(1)` → dtype proxy returns FloatImm           |


### What Has Traits vs What Doesn't

```
③ TVM-FFI AST nodes (ast.h / ast.py):
  tast.Function, tast.For, tast.Id, tast.Call, tast.Operation, ...
  → NO traits. Pure syntax. Shared by ALL dialects.

④ py_class IR nodes (user-defined):
  tir.PrimFunc  → @ffi.traits.Func(...)
  tir.For       → @ffi.traits.For(...)
  tir.Add       → @ffi.traits.BinOp(...)
  tir.BufferLoad→ @ffi.traits.Load(...)
  tir.FloatImm  → (no trait, L0)
  → HAVE traits. Semantic. Dialect-specific.
```

Traits live on ④, are consumed at setup time to produce surface objects
and operator overloading, and are never consulted during parsing of ③.

---

## Boundary: What TVM-FFI Provides vs What Users Provide

### TVM-FFI provides (this repo, `/home/kathrync/work/tvm-ffi/`)

**Already implemented:**

| What | Where |
|------|-------|
| `Object` — universal base class for ALL FFI objects | `python/tvm_ffi/core.py` |
| `@py_class` / `@c_class` — decorators to register types | `python/tvm_ffi/dataclasses/py_class.py` |
| TVM-FFI AST nodes — full Python 3.9+ syntax coverage | `include/tvm/ffi/text/ast.h`, `python/tvm_ffi/text/ast.py` |
| `ast_translate` — Python source → TVM-FFI AST | `python/tvm_ffi/text/ast_translate.py` |
| `PythonDocPrinter` — TVM-FFI AST → Python source string | `src/text/printer.cc` |
| `IRPrinter` — IR → TVM-FFI AST (dispatch via `__ir_print__`) | `include/tvm/ffi/text/printer.h`, `python/tvm_ffi/text/ir_printer.py` |
| `TypeAttrColumn` — per-type attribute dispatch | `include/tvm/ffi/reflection/accessor.h` |

**NOT yet implemented (design-doc only):**

| What | Notes |
|------|-------|
| `@ffi.traits.*` — trait decorators (For, Func, BinOp, Value, etc.) | Developer's next job |
| `$field:` / `$method:` / `$global:` resolution | Part of trait implementation |
| `TypeAttrColumn("__ffi_traits__")` — trait storage/lookup | Part of trait implementation |
| IRParser — TVM-FFI AST → User IR | **My job** |
| Surface objects — deferred IR constructors | **My job** |
| Language modules — `T`, `R`, `I` namespaces | Shared (printer + parser) |

### User's compiler provides (e.g., `/home/kathrync/work/tir/`)

| What | Example |
|------|---------|
| All IR node types via `@py_class` | `tir.PrimFunc`, `tir.For`, `tir.Var`, `tir.Add`, `tir.BufferLoad`, ... |
| IR base class hierarchy | `PrimExpr`, `Stmt`, `BaseFunc` — these are user-defined, NOT in TVM-FFI |
| Trait decorations on IR types | `@ffi.traits.For(...)` on `tir.For` (once traits exist) |
| `__ir_print__` methods (L2 printer) | Manual printer overrides on specific IR types |
| Operator overloading on IR types | `PrimExpr.__add__` returns `tir.Add`, etc. |
| Language module entries | `T.serial`, `T.prim_func`, `T.block`, etc. |

### Key Clarification: Base Classes

TVM-FFI provides exactly **one** base class: `tvm_ffi.Object`. Everything
else is user-defined. The ToyIR in `tvm_ffi/testing/testing.py` shows this:

```python
@py_class("testing.text.toy_ir.Node")
class ToyNode(Object):     # ← inherits from tvm_ffi.Object, the ONLY FFI base
    ...

@py_class("testing.text.toy_ir.Expr")
class ToyExpr(ToyNode):    # ← user-defined intermediate base
    ...

@py_class("testing.text.toy_ir.Stmt")
class ToyStmt(ToyNode):    # ← user-defined intermediate base
    ...
```

So in the TIR example, `PrimExpr`, `Stmt`, `BaseFunc` are all defined by
the TIR compiler — NOT by TVM-FFI. The TIR hierarchy looks like:

```
tvm_ffi.Object          ← TVM-FFI provides this
  └── tir.PrimExpr      ← TIR defines this
        ├── tir.Var
        ├── tir.Add
        ├── tir.BufferLoad
        ├── tir.FloatImm
        └── tir.IntImm
  └── tir.Stmt           ← TIR defines this
        ├── tir.For
        ├── tir.BufferStore
        └── tir.SeqStmt
  └── tir.BaseFunc       ← TIR defines this
        └── tir.PrimFunc
```

### What This Means for the Parser

The parser (my job) is a **TVM-FFI infrastructure component**. It must work
for ANY user-defined IR, not just TIR. The parser:

- **Knows about** TVM-FFI AST nodes (`tast.For`, `tast.Function`, etc.)
- **Knows about** the dispatch protocol (`__ffi_text_parse__` on surface objects)
- **Knows about** the trait system (to auto-generate surface objects)
- **Does NOT know about** specific IR types (`tir.For`, `tir.PrimFunc`)
- **Does NOT know about** specific language modules (`T.serial`, `T.block`)

The user's compiler provides the IR types + traits + language module entries.
The parser framework provides the evaluation, dispatch, and auto-generation
machinery.

### Revisiting the Example with Ownership Annotations

```python
# ① Printed text — produced by printer, consumed by parser

@T.prim_func                                    # "T" = user-defined lang module
def func(A: T.Buffer((4,), "float32"),           # "T.Buffer" = user-defined entry
         B: T.Buffer((4,), "float32")):
    for i in T.serial(4):                        # "T.serial" = user-defined entry
        B[i] = A[i] + T.float32(1)              # "T.float32" = user-defined entry


# ③ TVM-FFI AST — ALL nodes are TVM-FFI types (ast.h)

Function(...)    # tvm_ffi.text.ast.Function — TVM-FFI owns this
  For(...)       # tvm_ffi.text.ast.For      — TVM-FFI owns this
    Assign(...)  # tvm_ffi.text.ast.Assign   — TVM-FFI owns this
    Id(...)      # tvm_ffi.text.ast.Id       — TVM-FFI owns this
    Call(...)    # tvm_ffi.text.ast.Call     — TVM-FFI owns this


# ④ User IR — ALL nodes are user-defined types (@py_class)

PrimFunc(...)      # tir.PrimFunc  — TIR compiler owns this
  For(...)         # tir.For       — TIR compiler owns this
    BufferStore(...)# tir.BufferStore — TIR compiler owns this
    Add(...)       # tir.Add       — TIR compiler owns this
    BufferLoad(...)# tir.BufferLoad — TIR compiler owns this


# Parser infrastructure — TVM-FFI provides this (my job)

IRParser           # tvm_ffi.text.parser.IRParser    — TVM-FFI owns
eval_expr          # tvm_ffi.text.parser.eval_expr   — TVM-FFI owns
SurfaceObject      # tvm_ffi.text.parser.SurfaceObject — TVM-FFI owns
SyntaxContext      # tvm_ffi.text.parser.SyntaxContext  — TVM-FFI owns


# Surface objects on language modules — user provides these
#   (auto-generated from traits, or hand-written)

T.prim_func  → _PrimFuncDecorator   — TIR compiler owns (via trait auto-gen)
T.serial     → _SerialForIter       — TIR compiler owns (via trait auto-gen)
T.Buffer     → _BufferProxy         — TIR compiler owns (hand-written or auto)
T.float32    → _DtypeProxy          — TIR compiler owns (hand-written)
```