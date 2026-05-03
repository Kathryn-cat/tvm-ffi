/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
/*!
 * \file src/ffi/extra/std.cc
 * \brief Standard core dialect registration and text printing.
 */
#include <tvm/ffi/extra/pyast.h>
#include <tvm/ffi/extra/std.h>
#include <tvm/ffi/reflection/accessor.h>
#include <tvm/ffi/reflection/creator.h>
#include <tvm/ffi/reflection/registry.h>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace tvm {
namespace ffi {
namespace std_ {

namespace {

// The text AST APIs use int64_t sizes, while container sizes are size_t.
// Casts in this file are local size conversions for printer construction.
// NOLINTBEGIN(bugprone-narrowing-conversions, bugprone-misplaced-widening-cast)

namespace refl = ::tvm::ffi::reflection;
namespace text = ::tvm::ffi::pyast;

text::ExprAST DTypeAST(DLDataType dtype) { return text::IdAST(DTypeAbbrev(dtype)); }

text::ExprAST PrintExpr(const text::IRPrinter& printer, const Any& value,
                        const refl::AccessPath& path) {
  return printer->operator()(value, path).cast<text::ExprAST>();
}

bool AppendAttrsAsKwargs(const text::IRPrinter& printer, const Optional<Attrs>& attrs,
                         const refl::AccessPath& attrs_path, List<String>* kwargs_keys,
                         List<text::ExprAST>* kwargs_values) {
  if (!attrs.has_value()) return false;
  text::ExprAST attrs_ast = PrintExpr(printer, *attrs, attrs_path);
  const text::CallASTObj* call = attrs_ast.as<text::CallASTObj>();
  if (call == nullptr) {
    TVM_FFI_THROW(ValueError) << "ffi.std.Attrs text printer must return CallAST";
  }
  if (!call->args.empty()) {
    TVM_FFI_THROW(ValueError) << "ffi.std.Attrs text printer must use keyword arguments only";
  }
  kwargs_keys->reserve(static_cast<int64_t>(kwargs_keys->size() + call->kwargs_keys.size()));
  kwargs_values->reserve(static_cast<int64_t>(kwargs_values->size() + call->kwargs_values.size()));
  int64_t n = static_cast<int64_t>(call->kwargs_keys.size());
  for (int64_t i = 0; i < n; ++i) {
    kwargs_keys->push_back(call->kwargs_keys[i]);
    kwargs_values->push_back(call->kwargs_values[i]);
  }
  return n != 0;
}

text::StmtAST PrintStmt(const text::IRPrinter& printer, const Any& value,
                        const refl::AccessPath& path) {
  return printer->operator()(value, path).cast<text::StmtAST>();
}

List<text::ExprAST> PrintExprList(const text::IRPrinter& printer, const List<Expr>& values,
                                  const refl::AccessPath& path) {
  List<text::ExprAST> result;
  int64_t n = static_cast<int64_t>(values.size());
  result.reserve(n);
  for (int64_t i = 0; i < n; ++i) {
    result.push_back(PrintExpr(printer, values[i], path->ArrayItem(i)));
  }
  return result;
}

List<text::StmtAST> PrintStmtList(const text::IRPrinter& printer, const List<Stmt>& values,
                                  const refl::AccessPath& path) {
  List<text::StmtAST> result;
  int64_t n = static_cast<int64_t>(values.size());
  result.reserve(n);
  for (int64_t i = 0; i < n; ++i) {
    result.push_back(PrintStmt(printer, values[i], path->ArrayItem(i)));
  }
  return result;
}

text::ExprAST PrintTy(const text::IRPrinter& printer, Ty ty, const refl::AccessPath& path) {
  return PrintExpr(printer, std::move(ty), path);
}

text::ExprAST PrintVarRef(const text::IRPrinter& printer, const Var& var) {
  if (!printer->VarIsDefined(var)) {
    printer->VarDef(var->name, var, {});
  }
  Optional<text::ExprAST> ret = printer->VarGet(var);
  if (!ret.has_value()) {
    TVM_FFI_THROW(ValueError) << "ffi.std.Var printer failed to define variable " << var->name;
  }
  return *ret;
}

Optional<text::ExprAST> PrintOptionalExpr(const text::IRPrinter& printer,
                                          const Optional<Expr>& value,
                                          const refl::AccessPath& path) {
  if (!value.has_value()) return {};
  return PrintExpr(printer, *value, path);
}

Optional<text::ExprAST> PrintOptionalTy(const text::IRPrinter& printer, const Optional<Ty>& value,
                                        const refl::AccessPath& path) {
  if (!value.has_value()) return {};
  return PrintTy(printer, *value, path);
}

text::ExprAST VarsToExpr(const text::IRPrinter& printer, const List<Var>& vars,
                         const refl::AccessPath& path) {
  if (vars.size() == 1) {
    return PrintExpr(printer, vars[0], path->ArrayItem(0));
  }
  List<text::ExprAST> printed;
  int64_t n = static_cast<int64_t>(vars.size());
  printed.reserve(n);
  for (int64_t i = 0; i < n; ++i) {
    printed.push_back(PrintExpr(printer, vars[i], path->ArrayItem(i)));
  }
  return text::TupleAST(std::move(printed));
}

Optional<text::ExprAST> VarsToOptionalExpr(const text::IRPrinter& printer, const List<Var>& vars,
                                           const refl::AccessPath& path) {
  if (vars.empty()) return {};
  return VarsToExpr(printer, vars, path);
}

text::ExprAST PrintRangeAsIndex(const text::IRPrinter& printer, const Range& range,
                                const refl::AccessPath& path) {
  Optional<text::ExprAST> start = PrintOptionalExpr(printer, range->start, path->Attr("start"));
  Optional<text::ExprAST> stop = PrintOptionalExpr(printer, range->stop, path->Attr("stop"));
  Optional<text::ExprAST> step = PrintOptionalExpr(printer, range->step, path->Attr("step"));
  if (start.has_value() && !stop.has_value() && !step.has_value()) {
    return *start;
  }
  return text::SliceAST(std::move(start), std::move(stop), std::move(step));
}

List<text::ExprAST> PrintRangeAsCallArgs(const text::IRPrinter& printer, const Range& range,
                                         const refl::AccessPath& path) {
  List<text::ExprAST> args;
  if (range->start.has_value()) {
    args.push_back(PrintExpr(printer, *range->start, path->Attr("start")));
  }
  if (range->stop.has_value()) {
    if (!range->start.has_value()) {
      args.push_back(text::LiteralAST::Null());
    }
    args.push_back(PrintExpr(printer, *range->stop, path->Attr("stop")));
  }
  if (range->step.has_value()) {
    if (!range->stop.has_value()) {
      args.push_back(text::LiteralAST::Null());
    }
    args.push_back(PrintExpr(printer, *range->step, path->Attr("step")));
  }
  return args;
}

text::ExprAST PrintIndexedVar(const text::IRPrinter& printer, const Var& var,
                              const List<Range>& indices, const refl::AccessPath& path,
                              bool print_empty_indices = false) {
  text::ExprAST base = PrintVarRef(printer, var);
  if (indices.empty() && !print_empty_indices) return base;
  List<text::ExprAST> printed_indices;
  int64_t n = static_cast<int64_t>(indices.size());
  printed_indices.reserve(n);
  refl::AccessPath indices_path = path->Attr("indices");
  for (int64_t i = 0; i < n; ++i) {
    printed_indices.push_back(PrintRangeAsIndex(printer, indices[i], indices_path->ArrayItem(i)));
  }
  return text::IndexAST(std::move(base), std::move(printed_indices));
}

text::NodeAST FallbackStdReflectionTextPrint(const Node& obj, const text::IRPrinter& printer,
                                             const refl::AccessPath& path) {
  // This fallback is almost impossible to hit because concrete std nodes
  // register dedicated text printers.  Keep it to make abstract/base refs
  // printable if dispatch ever lands here.
  const TVMFFITypeInfo* info = TVMFFIGetTypeInfo(obj->type_index());

  List<String> keys;
  List<text::ExprAST> values;
  refl::ForEachFieldInfo(info, [&](const TVMFFIFieldInfo* finfo) {
    String name(finfo->name.data, finfo->name.size);
    Any field_value = refl::FieldGetter(finfo)(obj.get());
    keys.push_back(name);
    values.push_back(PrintExpr(printer, field_value, path->Attr(name)));
  });
  return text::CallAST(printer->Callee(obj), {}, std::move(keys), std::move(values));
}

text::NodeAST PrintBinaryOp(const Expr& lhs, const Expr& rhs, int64_t op,
                            const text::IRPrinter& printer, const refl::AccessPath& path) {
  return text::OperationAST(
      op, {PrintExpr(printer, lhs, path->Attr("a")), PrintExpr(printer, rhs, path->Attr("b"))});
}

text::NodeAST PrintNamedBinaryCall(const char* name, const Expr& lhs, const Expr& rhs,
                                   const text::IRPrinter& printer, const refl::AccessPath& path) {
  return text::ExprCall(text::IdAST(name), {PrintExpr(printer, lhs, path->Attr("a")),
                                            PrintExpr(printer, rhs, path->Attr("b"))});
}

text::ExprAST PrintCallee(const text::IRPrinter& printer, const Any& callee,
                          const refl::AccessPath& path) {
  if (std::optional<String> symbol = callee.as<String>()) {
    return text::IdAST(*symbol);
  }
  if (std::optional<Func> func = callee.as<Func>()) {
    return text::IdAST((*func)->symbol);
  }
  return PrintExpr(printer, callee, path);
}

text::ExprAST DefineVar(const text::IRPrinter& printer, const Var& var) {
  if (!printer->VarIsDefined(var)) {
    return printer->VarDef(var->name, var, {});
  }
  Optional<text::ExprAST> ret = printer->VarGet(var);
  if (!ret.has_value()) {
    TVM_FFI_THROW(ValueError) << "ffi.std.Var printer failed to fetch variable " << var->name;
  }
  return *ret;
}

Optional<text::ExprAST> DefineScopeVarsAsWithTargets(const text::IRPrinter& printer,
                                                     const List<Var>& vars) {
  if (vars.empty()) return {};
  if (vars.size() == 1) {
    return DefineVar(printer, vars[0]);
  }

  // WithAST accepts one optional target expression.  Multiple scope-carried
  // variables are target syntax rather than a tuple expression, so build a single
  // identifier spelling such as "x, state" after defining each Var.
  std::string lhs_name;
  int64_t n = static_cast<int64_t>(vars.size());
  for (int64_t i = 0; i < n; ++i) {
    text::ExprAST value_ast = DefineVar(printer, vars[i]);
    const text::IdASTObj* id = value_ast.as<text::IdASTObj>();
    if (id == nullptr) {
      TVM_FFI_THROW(ValueError)
          << "ffi.std.Scope expected carried variables to print as identifiers";
    }
    if (i != 0) lhs_name += ", ";
    lhs_name += std::string(id->name.data(), id->name.size());
  }
  return text::IdAST(lhs_name);
}

Optional<text::ExprAST> BindVarAnnotation(const text::IRPrinter& printer, const Var& var,
                                          const refl::AccessPath& path) {
  if (!var->ty.defined()) return {};
  return PrintTy(printer, var->ty, path->Attr("ty"));
}

text::ExprAST ExprBindRhs(const text::IRPrinter& printer, const ExprBind& obj,
                          const refl::AccessPath& path) {
  text::ExprAST plain_expr = PrintExpr(printer, obj->expr, path->Attr("expr"));
  List<String> kwargs_keys;
  List<text::ExprAST> kwargs_values;
  bool has_attrs =
      AppendAttrsAsKwargs(printer, obj->attrs, path->Attr("attrs"), &kwargs_keys, &kwargs_values);
  if (!has_attrs) {
    return plain_expr;
  }
  List<text::ExprAST> args{plain_expr};
  return text::ExprCallKw(printer->Callee(obj, "bind"), std::move(args), std::move(kwargs_keys),
                          std::move(kwargs_values));
}

text::ExprAST VarDefRhs(const text::IRPrinter& printer, const VarDef& obj,
                        const refl::AccessPath& path) {
  List<text::ExprAST> args;
  const List<Var>& vars = obj->vars;
  int64_t n = static_cast<int64_t>(vars.size());
  args.reserve(n);
  refl::AccessPath vars_path = path->Attr("vars");
  for (int64_t i = 0; i < n; ++i) {
    args.push_back(PrintTy(printer, vars[i]->ty, vars_path->ArrayItem(i)->Attr("ty")));
  }
  List<String> kwargs_keys;
  List<text::ExprAST> kwargs_values;
  bool has_attrs =
      AppendAttrsAsKwargs(printer, obj->attrs, path->Attr("attrs"), &kwargs_keys, &kwargs_values);
  if (!has_attrs) {
    return text::ExprCall(printer->Callee(obj, "var_def"), std::move(args));
  }
  return text::ExprCallKw(printer->Callee(obj, "var_def"), std::move(args), std::move(kwargs_keys),
                          std::move(kwargs_values));
}

text::ExprAST DefineBindVars(const text::IRPrinter& printer, const List<Var>& vars) {
  if (vars.size() == 1) {
    return DefineVar(printer, vars[0]);
  }
  List<text::ExprAST> lhs_vars;
  lhs_vars.reserve(static_cast<int64_t>(vars.size()));
  for (const Var& var : vars) {
    lhs_vars.push_back(DefineVar(printer, var));
  }
  return text::TupleAST(std::move(lhs_vars));
}

#define TVM_FFI_STD_GENERIC_TEXT_PRINT(TypeName)                               \
  text::NodeAST TextPrint(const TypeName& obj, const text::IRPrinter& printer, \
                          const refl::AccessPath& path) {                      \
    return FallbackStdReflectionTextPrint(obj, printer, path);                 \
  }

TVM_FFI_STD_GENERIC_TEXT_PRINT(Node)
TVM_FFI_STD_GENERIC_TEXT_PRINT(Ty)
TVM_FFI_STD_GENERIC_TEXT_PRINT(Stmt)
TVM_FFI_STD_GENERIC_TEXT_PRINT(Attrs)
TVM_FFI_STD_GENERIC_TEXT_PRINT(Structure)
TVM_FFI_STD_GENERIC_TEXT_PRINT(Expr)
TVM_FFI_STD_GENERIC_TEXT_PRINT(Bind)

#undef TVM_FFI_STD_GENERIC_TEXT_PRINT

text::NodeAST TextPrint(const Var& obj, const text::IRPrinter& printer,
                        const refl::AccessPath& path) {
  return PrintVarRef(printer, obj);
}

text::NodeAST TextPrint(const Module& obj, const text::IRPrinter& printer,
                        const refl::AccessPath& path) {
  List<text::StmtAST> stmts;
  int64_t n = static_cast<int64_t>(obj->funcs.size());
  stmts.reserve(n);
  refl::AccessPath funcs_path = path->Attr("funcs");
  for (int64_t i = 0; i < n; ++i) {
    stmts.push_back(PrintStmt(printer, obj->funcs[i], funcs_path->ArrayItem(i)));
  }
  return text::StmtBlockAST(std::move(stmts));
}

text::NodeAST TextPrint(const Func& obj, const text::IRPrinter& printer,
                        const refl::AccessPath& path) {
  List<text::AssignAST> args;
  int64_t n = static_cast<int64_t>(obj->args.size());
  args.reserve(n);
  refl::AccessPath args_path = path->Attr("args");
  for (int64_t i = 0; i < n; ++i) {
    const Var& arg = obj->args[i];
    text::ExprAST lhs = DefineVar(printer, arg);
    Optional<text::ExprAST> annotation = BindVarAnnotation(printer, arg, args_path->ArrayItem(i));
    args.push_back(text::AssignAST(std::move(lhs), {}, std::move(annotation)));
  }
  List<text::StmtAST> body = PrintStmtList(printer, obj->body, path->Attr("body"));
  Optional<text::ExprAST> ret_type =
      PrintOptionalTy(printer, obj->ret_type, path->Attr("ret_type"));
  List<String> decorator_keys;
  List<text::ExprAST> decorator_values;
  bool has_attrs = AppendAttrsAsKwargs(printer, obj->attrs, path->Attr("attrs"), &decorator_keys,
                                       &decorator_values);
  List<text::ExprAST> decorators;
  if (!has_attrs) {
    decorators.push_back(printer->Callee(obj, "func"));
  } else {
    decorators.push_back(text::ExprCallKw(printer->Callee(obj, "func"), {},
                                          std::move(decorator_keys), std::move(decorator_values)));
  }
  return text::FunctionAST(text::IdAST(obj->symbol), std::move(args), std::move(decorators),
                           std::move(ret_type), std::move(body));
}

text::NodeAST TextPrint(const Range& obj, const text::IRPrinter& printer,
                        const refl::AccessPath& path) {
  return PrintRangeAsIndex(printer, obj, path);
}

text::NodeAST TextPrint(const AnyTy& obj, const text::IRPrinter& printer,
                        const refl::AccessPath& path) {
  return text::IdAST("Any");
}

text::NodeAST TextPrint(const PrimTy& obj, const text::IRPrinter& printer,
                        const refl::AccessPath& path) {
  return DTypeAST(obj->dtype);
}

text::NodeAST TextPrint(const TupleType& obj, const text::IRPrinter& printer,
                        const refl::AccessPath& path) {
  List<text::ExprAST> fields;
  int64_t n = static_cast<int64_t>(obj->fields.size());
  fields.reserve(n);
  refl::AccessPath fields_path = path->Attr("fields");
  for (int64_t i = 0; i < n; ++i) {
    fields.push_back(PrintTy(printer, obj->fields[i], fields_path->ArrayItem(i)));
  }
  return text::IndexAST(text::IdAST("tuple"), std::move(fields));
}

text::NodeAST TextPrint(const TensorTy& obj, const text::IRPrinter& printer,
                        const refl::AccessPath& path) {
  return text::IndexAST(DTypeAST(obj->dtype),
                        PrintExprList(printer, obj->shape, path->Attr("shape")));
}

#define TVM_FFI_STD_LITERAL_TEXT_PRINT(TypeName, LiteralFactory)               \
  text::NodeAST TextPrint(const TypeName& obj, const text::IRPrinter& printer, \
                          const refl::AccessPath& path) {                      \
    return text::LiteralAST::LiteralFactory(obj->value);                       \
  }

TVM_FFI_STD_LITERAL_TEXT_PRINT(IntImm, Int)
TVM_FFI_STD_LITERAL_TEXT_PRINT(FloatImm, Float)
TVM_FFI_STD_LITERAL_TEXT_PRINT(StringImm, Str)

#undef TVM_FFI_STD_LITERAL_TEXT_PRINT

#define TVM_FFI_STD_BINARY_TEXT_PRINT(TypeName, OpKind)                                 \
  text::NodeAST TextPrint(const TypeName& obj, const text::IRPrinter& printer,          \
                          const refl::AccessPath& path) {                               \
    return PrintBinaryOp(obj->a, obj->b, text::OperationASTObj::OpKind, printer, path); \
  }

TVM_FFI_STD_BINARY_TEXT_PRINT(Add, kAdd)
TVM_FFI_STD_BINARY_TEXT_PRINT(Sub, kSub)
TVM_FFI_STD_BINARY_TEXT_PRINT(Mul, kMult)
TVM_FFI_STD_BINARY_TEXT_PRINT(FloorDiv, kFloorDiv)
TVM_FFI_STD_BINARY_TEXT_PRINT(FloorMod, kMod)

text::NodeAST TextPrint(const Min& obj, const text::IRPrinter& printer,
                        const refl::AccessPath& path) {
  return PrintNamedBinaryCall("min", obj->a, obj->b, printer, path);
}

text::NodeAST TextPrint(const Max& obj, const text::IRPrinter& printer,
                        const refl::AccessPath& path) {
  return PrintNamedBinaryCall("max", obj->a, obj->b, printer, path);
}

TVM_FFI_STD_BINARY_TEXT_PRINT(Eq, kEq)
TVM_FFI_STD_BINARY_TEXT_PRINT(Ne, kNotEq)
TVM_FFI_STD_BINARY_TEXT_PRINT(Le, kLtE)
TVM_FFI_STD_BINARY_TEXT_PRINT(Ge, kGtE)
TVM_FFI_STD_BINARY_TEXT_PRINT(Gt, kGt)
TVM_FFI_STD_BINARY_TEXT_PRINT(Lt, kLt)
TVM_FFI_STD_BINARY_TEXT_PRINT(And, kAnd)
TVM_FFI_STD_BINARY_TEXT_PRINT(Or, kOr)

#undef TVM_FFI_STD_BINARY_TEXT_PRINT

text::NodeAST TextPrint(const Not& obj, const text::IRPrinter& printer,
                        const refl::AccessPath& path) {
  return text::OperationAST(text::OperationASTObj::kNot,
                            {PrintExpr(printer, obj->operand, path->Attr("operand"))});
}

text::NodeAST TextPrint(const Load& obj, const text::IRPrinter& printer,
                        const refl::AccessPath& path) {
  return PrintIndexedVar(printer, obj->var, obj->indices, path);
}

text::NodeAST TextPrint(const Cast& obj, const text::IRPrinter& printer,
                        const refl::AccessPath& path) {
  if (const PrimTyObj* prim_ty = obj->ty.as<PrimTyObj>()) {
    return text::ExprCall(printer->Callee(obj, DTypeAbbrev(prim_ty->dtype)),
                          {PrintExpr(printer, obj->value, path->Attr("value"))});
  }
  return text::ExprCall(printer->Callee(obj, "cast"),
                        {PrintExpr(printer, obj->value, path->Attr("value")),
                         PrintTy(printer, obj->ty, path->Attr("ty"))});
}

text::NodeAST TextPrint(const Call& obj, const text::IRPrinter& printer,
                        const refl::AccessPath& path) {
  text::ExprAST callee = PrintCallee(printer, obj->callee, path->Attr("callee"));
  List<text::ExprAST> args = PrintExprList(printer, obj->args, path->Attr("args"));
  List<text::ExprAST> call_args{std::move(callee)};
  call_args.reserve(static_cast<int64_t>(obj->args.size() + 1));
  for (text::ExprAST arg : args) {
    call_args.push_back(arg);
  }
  List<String> kwargs_keys;
  List<text::ExprAST> kwargs_values;
  if (!AppendAttrsAsKwargs(printer, obj->attr, path->Attr("attr"), &kwargs_keys, &kwargs_values)) {
    return text::ExprCall(printer->Callee(obj, "call"), std::move(call_args));
  }
  return text::ExprCallKw(printer->Callee(obj, "call"), std::move(call_args),
                          std::move(kwargs_keys), std::move(kwargs_values));
}

text::NodeAST TextPrint(const IfStmt& obj, const text::IRPrinter& printer,
                        const refl::AccessPath& path) {
  return text::IfAST(PrintExpr(printer, obj->cond, path->Attr("cond")),
                     PrintStmtList(printer, obj->then_body, path->Attr("then_body")),
                     PrintStmtList(printer, obj->else_body, path->Attr("else_body")));
}

text::NodeAST TextPrint(const For& obj, const text::IRPrinter& printer,
                        const refl::AccessPath& path) {
  text::ExprAST lhs = text::IdAST("_");
  if (!obj->vars.empty()) {
    const List<Var>& vars = obj->vars;
    if (vars.size() == 1) {
      lhs = DefineVar(printer, vars[0]);
    } else {
      List<text::ExprAST> lhs_values;
      lhs_values.reserve(static_cast<int64_t>(vars.size()));
      for (const Var& var : vars) {
        lhs_values.push_back(DefineVar(printer, var));
      }
      lhs = text::TupleAST(std::move(lhs_values));
    }
  }
  List<text::ExprAST> range_args = PrintRangeAsCallArgs(printer, obj->range_, path->Attr("range_"));
  List<String> range_kwarg_keys;
  List<text::ExprAST> range_kwarg_values;
  AppendAttrsAsKwargs(printer, obj->attrs, path->Attr("attrs"), &range_kwarg_keys,
                      &range_kwarg_values);
  text::ExprAST rhs =
      range_kwarg_keys.empty()
          ? text::ExprCall(text::IdAST("range"), std::move(range_args))
          : text::ExprCallKw(text::IdAST("range"), std::move(range_args),
                             std::move(range_kwarg_keys), std::move(range_kwarg_values));
  return text::ForAST(std::move(lhs), std::move(rhs),
                      PrintStmtList(printer, obj->body, path->Attr("body")));
}

text::NodeAST TextPrint(const While& obj, const text::IRPrinter& printer,
                        const refl::AccessPath& path) {
  List<String> while_kwarg_keys;
  List<text::ExprAST> while_kwarg_values;
  bool has_attrs = AppendAttrsAsKwargs(printer, obj->attrs, path->Attr("attrs"), &while_kwarg_keys,
                                       &while_kwarg_values);
  if (obj->vars.empty() && !has_attrs) {
    return text::WhileAST(PrintExpr(printer, obj->cond, path->Attr("cond")),
                          PrintStmtList(printer, obj->body, path->Attr("body")));
  }
  text::ExprAST rhs =
      while_kwarg_keys.empty()
          ? text::ExprCall(printer->Callee(obj, "While"),
                           {PrintExpr(printer, obj->cond, path->Attr("cond"))})
          : text::ExprCallKw(printer->Callee(obj, "While"),
                             {PrintExpr(printer, obj->cond, path->Attr("cond"))},
                             std::move(while_kwarg_keys), std::move(while_kwarg_values));
  Optional<text::ExprAST> lhs = DefineScopeVarsAsWithTargets(printer, obj->vars);
  return text::WithAST(std::move(lhs), std::move(rhs),
                       PrintStmtList(printer, obj->body, path->Attr("body")));
}

text::NodeAST TextPrint(const Scope& obj, const text::IRPrinter& printer,
                        const refl::AccessPath& path) {
  List<String> scope_kwarg_keys;
  List<text::ExprAST> scope_kwarg_values;
  bool has_attrs = AppendAttrsAsKwargs(printer, obj->attrs, path->Attr("attrs"), &scope_kwarg_keys,
                                       &scope_kwarg_values);
  if (obj->vars.empty() && !has_attrs) {
    return text::StmtBlockAST(PrintStmtList(printer, obj->body, path->Attr("body")));
  }
  text::ExprAST rhs = scope_kwarg_keys.empty() ? text::ExprCall(printer->Callee(obj, "Scope"), {})
                                               : text::ExprCallKw(printer->Callee(obj, "Scope"), {},
                                                                  std::move(scope_kwarg_keys),
                                                                  std::move(scope_kwarg_values));
  Optional<text::ExprAST> lhs = DefineScopeVarsAsWithTargets(printer, obj->vars);
  return text::WithAST(std::move(lhs), std::move(rhs),
                       PrintStmtList(printer, obj->body, path->Attr("body")));
}

text::NodeAST TextPrint(const ExprBind& obj, const text::IRPrinter& printer,
                        const refl::AccessPath& path) {
  text::ExprAST rhs = ExprBindRhs(printer, obj, path);
  if (obj->vars.empty()) {
    return text::ExprStmtAST(std::move(rhs));
  }
  text::ExprAST lhs = DefineBindVars(printer, obj->vars);
  return text::AssignAST(std::move(lhs), std::move(rhs));
}

text::NodeAST TextPrint(const VarDef& obj, const text::IRPrinter& printer,
                        const refl::AccessPath& path) {
  if (obj->vars.empty()) {
    List<String> kwargs_keys;
    List<text::ExprAST> kwargs_values;
    bool has_attrs =
        AppendAttrsAsKwargs(printer, obj->attrs, path->Attr("attrs"), &kwargs_keys, &kwargs_values);
    if (!has_attrs) {
      return text::ExprStmtAST(text::IdAST("pass"));
    }
    return text::ExprStmtAST(text::ExprCallKw(printer->Callee(obj, "var_def"), {},
                                              std::move(kwargs_keys), std::move(kwargs_values)));
  }
  text::ExprAST lhs = DefineBindVars(printer, obj->vars);
  text::ExprAST rhs = VarDefRhs(printer, obj, path);
  return text::AssignAST(std::move(lhs), std::move(rhs));
}

text::NodeAST TextPrint(const Store& obj, const text::IRPrinter& printer,
                        const refl::AccessPath& path) {
  return text::AssignAST(PrintIndexedVar(printer, obj->var, obj->indices, path,
                                         /*print_empty_indices=*/true),
                         PrintExpr(printer, obj->rhs, path->Attr("rhs")));
}

text::NodeAST TextPrint(const Return& obj, const text::IRPrinter& printer,
                        const refl::AccessPath& path) {
  return text::ReturnAST(VarsToOptionalExpr(printer, obj->vars, path->Attr("vars")));
}

text::NodeAST TextPrint(const Yield_& obj, const text::IRPrinter& printer,
                        const refl::AccessPath& path) {
  return text::ExprStmtAST(
      text::YieldAST(VarsToOptionalExpr(printer, obj->vars, path->Attr("vars"))));
}

text::NodeAST TextPrint(const Break& obj, const text::IRPrinter& printer,
                        const refl::AccessPath& path) {
  return text::ExprStmtAST(text::IdAST("break"));
}

text::NodeAST TextPrint(const Continue& obj, const text::IRPrinter& printer,
                        const refl::AccessPath& path) {
  return text::ExprStmtAST(text::IdAST("continue"));
}

text::NodeAST TextPrint(const DictAttrs& obj, const text::IRPrinter& printer,
                        const refl::AccessPath& path) {
  List<String> kwargs_keys;
  List<text::ExprAST> kwargs_values;
  std::vector<String> sorted_keys;
  sorted_keys.reserve(obj->values.size());
  for (const auto& kv : obj->values) {
    sorted_keys.push_back(kv.first);
  }
  std::sort(sorted_keys.begin(), sorted_keys.end());

  kwargs_keys.reserve(static_cast<int64_t>(sorted_keys.size()));
  kwargs_values.reserve(static_cast<int64_t>(sorted_keys.size()));
  refl::AccessPath values_path = path->Attr("values");
  int64_t n = static_cast<int64_t>(sorted_keys.size());
  for (int64_t i = 0; i < n; ++i) {
    const String& key = sorted_keys[i];
    kwargs_keys.push_back(key);
    kwargs_values.push_back(PrintExpr(printer, obj->values[key], values_path->MapItem(key)));
  }
  return text::CallAST(printer->Callee(obj, "DictAttrs"), {}, std::move(kwargs_keys),
                       std::move(kwargs_values));
}

template <typename T>
auto TextPrintHook() {
  return
      [](const T& obj, const text::IRPrinter& printer,
         const refl::AccessPath& path) -> text::NodeAST { return TextPrint(obj, printer, path); };
}

// NOLINTEND(bugprone-narrowing-conversions, bugprone-misplaced-widening-cast)

}  // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = ::tvm::ffi::reflection;

#define TVM_FFI_STD_OBJECT_DEF_BASE(ObjType, RefType) \
  refl::ObjectDef<ObjType>().def_type_attr(refl::type_attr::kTextPrint, TextPrintHook<RefType>())

#define TVM_FFI_STD_OBJECT_DEF_BASE_INIT(ObjType, RefType, ...) \
  refl::ObjectDef<ObjType>(__VA_ARGS__)                         \
      .def_type_attr(refl::type_attr::kTextPrint, TextPrintHook<RefType>())

#define TVM_FFI_STD_OBJECT_DEF(ObjType, RefType, Name)                      \
  refl::ObjectDef<ObjType>()                                                \
      .def_type_attr(refl::type_attr::kTextPrint, TextPrintHook<RefType>()) \
      .def_type_attr(refl::type_attr::kMnemonic, "std$" Name)

  TVM_FFI_STD_OBJECT_DEF_BASE_INIT(NodeObj, Node, refl::init(false));
  TVM_FFI_STD_OBJECT_DEF_BASE_INIT(TyObj, Ty, refl::init(false));
  TVM_FFI_STD_OBJECT_DEF_BASE_INIT(StmtObj, Stmt, refl::init(false));
  TVM_FFI_STD_OBJECT_DEF_BASE_INIT(AttrsObj, Attrs, refl::init(false)).def_convert<Attrs>();
  TVM_FFI_STD_OBJECT_DEF_BASE_INIT(StructureObj, Structure, refl::init(false));
  TVM_FFI_STD_OBJECT_DEF_BASE_INIT(ExprObj, Expr, refl::init(false))
      .def_convert<Expr>()
      .def_rw("ty", &ExprObj::ty);
  TVM_FFI_STD_OBJECT_DEF(VarObj, Var, "Var")
      .def_rw("name", &VarObj::name, refl::AttachFieldFlag::SEqHashIgnore());
  TVM_FFI_STD_OBJECT_DEF(FuncObj, Func, "Func")
      .def_rw("symbol", &FuncObj::symbol)
      .def_rw("attrs", &FuncObj::attrs)
      .def_rw("args", &FuncObj::args, refl::AttachFieldFlag::SEqHashDef())
      .def_rw("ret_type", &FuncObj::ret_type)
      .def_rw("body", &FuncObj::body);
  TVM_FFI_STD_OBJECT_DEF(ModuleObj, Module, "Module").def_rw("funcs", &ModuleObj::funcs);
  TVM_FFI_STD_OBJECT_DEF(RangeObj, Range, "Range")
      .def_convert<Range>()
      .def_rw("start", &RangeObj::start, refl::default_value(nullptr))
      .def_rw("stop", &RangeObj::stop, refl::default_value(nullptr))
      .def_rw("step", &RangeObj::step, refl::default_value(nullptr));
  TVM_FFI_STD_OBJECT_DEF(AnyTyObj, AnyTy, "AnyTy");
  TVM_FFI_STD_OBJECT_DEF(PrimTyObj, PrimTy, "PrimTy").def_rw("dtype", &PrimTyObj::dtype);
  TVM_FFI_STD_OBJECT_DEF(TupleTypeObj, TupleType, "TupleType")
      .def_rw("fields", &TupleTypeObj::fields);
  TVM_FFI_STD_OBJECT_DEF(TensorTyObj, TensorTy, "TensorTy")
      .def_rw("shape", &TensorTyObj::shape)
      .def_rw("dtype", &TensorTyObj::dtype);
  TVM_FFI_STD_OBJECT_DEF(IntImmObj, IntImm, "IntImm").def_rw("value", &IntImmObj::value);
  TVM_FFI_STD_OBJECT_DEF(FloatImmObj, FloatImm, "FloatImm").def_rw("value", &FloatImmObj::value);
  TVM_FFI_STD_OBJECT_DEF(StringImmObj, StringImm, "StringImm")
      .def_rw("value", &StringImmObj::value);

#define TVM_FFI_STD_DEF_BINARY(TypeName)                     \
  TVM_FFI_STD_OBJECT_DEF(TypeName##Obj, TypeName, #TypeName) \
      .def_rw("a", &TypeName##Obj::a)                        \
      .def_rw("b", &TypeName##Obj::b)

  TVM_FFI_STD_DEF_BINARY(Add);
  TVM_FFI_STD_DEF_BINARY(Sub);
  TVM_FFI_STD_DEF_BINARY(Mul);
  TVM_FFI_STD_DEF_BINARY(FloorDiv);
  TVM_FFI_STD_DEF_BINARY(FloorMod);
  TVM_FFI_STD_DEF_BINARY(Min);
  TVM_FFI_STD_DEF_BINARY(Max);
  TVM_FFI_STD_DEF_BINARY(Eq);
  TVM_FFI_STD_DEF_BINARY(Ne);
  TVM_FFI_STD_DEF_BINARY(Le);
  TVM_FFI_STD_DEF_BINARY(Ge);
  TVM_FFI_STD_DEF_BINARY(Gt);
  TVM_FFI_STD_DEF_BINARY(Lt);
  TVM_FFI_STD_DEF_BINARY(And);
  TVM_FFI_STD_DEF_BINARY(Or);

#undef TVM_FFI_STD_DEF_BINARY

  TVM_FFI_STD_OBJECT_DEF(NotObj, Not, "Not").def_convert<Not>().def_rw("operand", &NotObj::operand);
  TVM_FFI_STD_OBJECT_DEF(LoadObj, Load, "Load")
      .def_rw("var", &LoadObj::var)
      .def_rw("indices", &LoadObj::indices);
  TVM_FFI_STD_OBJECT_DEF(CastObj, Cast, "Cast").def_rw("value", &CastObj::value);
  TVM_FFI_STD_OBJECT_DEF(CallObj, Call, "Call")
      .def_rw("callee", &CallObj::callee)
      .def_rw("args", &CallObj::args)
      .def_rw("attr", &CallObj::attr, refl::default_value(nullptr));
  TVM_FFI_STD_OBJECT_DEF(IfStmtObj, IfStmt, "IfStmt")
      .def_rw("cond", &IfStmtObj::cond)
      .def_rw("then_body", &IfStmtObj::then_body)
      .def_rw("else_body", &IfStmtObj::else_body);
  TVM_FFI_STD_OBJECT_DEF(ScopeObj, Scope, "Scope")
      .def_rw("attrs", &ScopeObj::attrs)
      .def_rw("vars", &ScopeObj::vars, refl::AttachFieldFlag::SEqHashDef())
      .def_rw("body", &ScopeObj::body);
  TVM_FFI_STD_OBJECT_DEF(ForObj, For, "For").def_rw("range_", &ForObj::range_);
  TVM_FFI_STD_OBJECT_DEF(WhileObj, While, "While").def_rw("cond", &WhileObj::cond);
  TVM_FFI_STD_OBJECT_DEF_BASE_INIT(BindObj, Bind, refl::init(false))
      .def_rw("vars", &BindObj::vars, refl::AttachFieldFlag::SEqHashDef())
      .def_rw("attrs", &BindObj::attrs);
  TVM_FFI_STD_OBJECT_DEF(ExprBindObj, ExprBind, "ExprBind").def_rw("expr", &ExprBindObj::expr);
  TVM_FFI_STD_OBJECT_DEF(VarDefObj, VarDef, "VarDef");
  TVM_FFI_STD_OBJECT_DEF(StoreObj, Store, "Store")
      .def_rw("var", &StoreObj::var)
      .def_rw("indices", &StoreObj::indices)
      .def_rw("rhs", &StoreObj::rhs);
  TVM_FFI_STD_OBJECT_DEF(ReturnObj, Return, "Return").def_rw("vars", &ReturnObj::vars);
  TVM_FFI_STD_OBJECT_DEF(YieldObj, Yield_, "Yield").def_rw("vars", &YieldObj::vars);
  TVM_FFI_STD_OBJECT_DEF(BreakObj, Break, "Break");
  TVM_FFI_STD_OBJECT_DEF(ContinueObj, Continue, "Continue");
  TVM_FFI_STD_OBJECT_DEF(DictAttrsObj, DictAttrs, "DictAttrs")
      .def_rw("values", &DictAttrsObj::values);

#undef TVM_FFI_STD_OBJECT_DEF
#undef TVM_FFI_STD_OBJECT_DEF_BASE
#undef TVM_FFI_STD_OBJECT_DEF_BASE_INIT
}

}  // namespace std_
}  // namespace ffi
}  // namespace tvm
