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

#include <gtest/gtest.h>
#include <tvm/ffi/extra/pyast.h>
#include <tvm/ffi/extra/std.h>
#include <tvm/ffi/reflection/access_path.h>

#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace ffi = tvm::ffi;
namespace stdir = tvm::ffi::std_;
namespace text = tvm::ffi::pyast;
namespace refl = tvm::ffi::reflection;

TEST(StdDialect, ConstructAndAccessFields) {
  stdir::PrimTy i32(ffi::StringToDLDataType("int32"));
  stdir::IntImm one(i32, 1);
  stdir::Var x(i32, "x");
  stdir::Add add(i32, x, one);

  EXPECT_EQ(ffi::DLDataTypeToString(i32->dtype), "int32");
  EXPECT_EQ(one->value, 1);
  EXPECT_EQ(x->name, "x");
  EXPECT_TRUE(add->a.as<stdir::Var>().has_value());
  EXPECT_TRUE(add->b.as<stdir::IntImm>().has_value());
}

TEST(StdDialect, TextPrintFunction) {
  stdir::PrimTy i32(ffi::StringToDLDataType("int32"));
  stdir::Var x(i32, "x");
  stdir::Var y(i32, "y");
  stdir::IntImm one(i32, 1);
  stdir::Add add(i32, x, one);
  stdir::ExprBind bind({y}, ffi::Optional<stdir::Attrs>(), add);
  stdir::Return ret({y});
  stdir::Func func("main", ffi::Optional<stdir::Attrs>(), {x}, ffi::Optional<stdir::Ty>(i32),
                   {bind, ret});
  stdir::Module mod({func});

  text::IRPrinter printer{text::PrinterConfig()};
  text::NodeAST ast = printer->operator()(mod, refl::AccessPath::Root()).cast<text::NodeAST>();
  std::string rendered = ast->ToPython(text::PrinterConfig());

  EXPECT_NE(rendered.find("@std.func"), std::string::npos);
  EXPECT_NE(rendered.find("def main"), std::string::npos);
  EXPECT_NE(rendered.find("x + 1"), std::string::npos);
  EXPECT_NE(rendered.find("return y"), std::string::npos);
}

TEST(StdDialect, TextPrintVarDef) {
  stdir::PrimTy i32(ffi::StringToDLDataType("int32"));
  stdir::Var y(i32, "y");
  stdir::VarDef def({y}, ffi::Optional<stdir::Attrs>());

  text::IRPrinter printer{text::PrinterConfig()};
  text::NodeAST ast = printer->operator()(def, refl::AccessPath::Root()).cast<text::NodeAST>();

  EXPECT_EQ(ast->ToPython(text::PrinterConfig()), "y = std.var_def(i32)");
}

TEST(StdDialect, DialectPrintMap) {
  ffi::Dict<ffi::String, ffi::Any> values;
  values.Set("tag", ffi::String("demo"));
  stdir::DictAttrs attrs(values);

  text::IRPrinter drop_printer{text::PrinterConfig(
      true, 2, 0, -1, false, {}, ffi::Dict<ffi::String, ffi::String>{{"std", "*"}})};
  text::NodeAST drop_ast =
      drop_printer->operator()(attrs, refl::AccessPath::Root()).cast<text::NodeAST>();
  EXPECT_EQ(drop_ast->ToPython(drop_printer->cfg), "DictAttrs(tag=\"demo\")");

  text::IRPrinter rename_printer{text::PrinterConfig(
      true, 2, 0, -1, false, {}, ffi::Dict<ffi::String, ffi::String>{{"std", "core"}})};
  text::NodeAST rename_ast =
      rename_printer->operator()(attrs, refl::AccessPath::Root()).cast<text::NodeAST>();
  EXPECT_EQ(rename_ast->ToPython(rename_printer->cfg), "core.DictAttrs(tag=\"demo\")");

  text::IRPrinter exact_printer{text::PrinterConfig(
      true, 2, 0, -1, false, {},
      ffi::Dict<ffi::String, ffi::String>{{"std", "core"}, {"std$DictAttrs", "std.MyAttrs"}})};
  text::NodeAST exact_ast =
      exact_printer->operator()(attrs, refl::AccessPath::Root()).cast<text::NodeAST>();
  EXPECT_EQ(exact_ast->ToPython(exact_printer->cfg), "std.MyAttrs(tag=\"demo\")");
}

TEST(StdDialect, Mnemonics) {
  refl::TypeAttrColumn mnemonic_col(refl::type_attr::kMnemonic);
  std::vector<std::pair<int32_t, ffi::String>> cases = {
      {stdir::AnyTyObj::RuntimeTypeIndex(), "std$AnyTy"},
      {stdir::PrimTyObj::RuntimeTypeIndex(), "std$PrimTy"},
      {stdir::TupleTypeObj::RuntimeTypeIndex(), "std$TupleType"},
      {stdir::TensorTyObj::RuntimeTypeIndex(), "std$TensorTy"},
      {stdir::RangeObj::RuntimeTypeIndex(), "std$Range"},
      {stdir::DictAttrsObj::RuntimeTypeIndex(), "std$DictAttrs"},
      {stdir::VarObj::RuntimeTypeIndex(), "std$Var"},
      {stdir::FuncObj::RuntimeTypeIndex(), "std$Func"},
      {stdir::ModuleObj::RuntimeTypeIndex(), "std$Module"},
      {stdir::IntImmObj::RuntimeTypeIndex(), "std$IntImm"},
      {stdir::FloatImmObj::RuntimeTypeIndex(), "std$FloatImm"},
      {stdir::StringImmObj::RuntimeTypeIndex(), "std$StringImm"},
      {stdir::AddObj::RuntimeTypeIndex(), "std$Add"},
      {stdir::SubObj::RuntimeTypeIndex(), "std$Sub"},
      {stdir::MulObj::RuntimeTypeIndex(), "std$Mul"},
      {stdir::FloorDivObj::RuntimeTypeIndex(), "std$FloorDiv"},
      {stdir::FloorModObj::RuntimeTypeIndex(), "std$FloorMod"},
      {stdir::MinObj::RuntimeTypeIndex(), "std$Min"},
      {stdir::MaxObj::RuntimeTypeIndex(), "std$Max"},
      {stdir::EqObj::RuntimeTypeIndex(), "std$Eq"},
      {stdir::NeObj::RuntimeTypeIndex(), "std$Ne"},
      {stdir::LeObj::RuntimeTypeIndex(), "std$Le"},
      {stdir::GeObj::RuntimeTypeIndex(), "std$Ge"},
      {stdir::GtObj::RuntimeTypeIndex(), "std$Gt"},
      {stdir::LtObj::RuntimeTypeIndex(), "std$Lt"},
      {stdir::AndObj::RuntimeTypeIndex(), "std$And"},
      {stdir::OrObj::RuntimeTypeIndex(), "std$Or"},
      {stdir::NotObj::RuntimeTypeIndex(), "std$Not"},
      {stdir::LoadObj::RuntimeTypeIndex(), "std$Load"},
      {stdir::CastObj::RuntimeTypeIndex(), "std$Cast"},
      {stdir::CallObj::RuntimeTypeIndex(), "std$Call"},
      {stdir::IfStmtObj::RuntimeTypeIndex(), "std$IfStmt"},
      {stdir::ScopeObj::RuntimeTypeIndex(), "std$Scope"},
      {stdir::ForObj::RuntimeTypeIndex(), "std$For"},
      {stdir::WhileObj::RuntimeTypeIndex(), "std$While"},
      {stdir::ExprBindObj::RuntimeTypeIndex(), "std$ExprBind"},
      {stdir::VarDefObj::RuntimeTypeIndex(), "std$VarDef"},
      {stdir::StoreObj::RuntimeTypeIndex(), "std$Store"},
      {stdir::ReturnObj::RuntimeTypeIndex(), "std$Return"},
      {stdir::YieldObj::RuntimeTypeIndex(), "std$Yield"},
      {stdir::BreakObj::RuntimeTypeIndex(), "std$Break"},
      {stdir::ContinueObj::RuntimeTypeIndex(), "std$Continue"},
  };
  std::set<std::string> seen;

  EXPECT_EQ(cases.size(), 42);
  for (const auto& [type_index, expected] : cases) {
    ffi::AnyView value = mnemonic_col[type_index];

    ASSERT_NE(value.type_index(), ffi::TypeIndex::kTVMFFINone);
    EXPECT_EQ(value.cast<ffi::String>(), expected);
    EXPECT_TRUE(seen.insert(static_cast<std::string>(expected)).second);
  }
}

TEST(StdDialect, AbstractBasesDoNotHaveMnemonics) {
  refl::TypeAttrColumn mnemonic_col(refl::type_attr::kMnemonic);

  EXPECT_EQ(mnemonic_col[stdir::NodeObj::RuntimeTypeIndex()].type_index(),
            ffi::TypeIndex::kTVMFFINone);
  EXPECT_EQ(mnemonic_col[stdir::TyObj::RuntimeTypeIndex()].type_index(),
            ffi::TypeIndex::kTVMFFINone);
  EXPECT_EQ(mnemonic_col[stdir::StmtObj::RuntimeTypeIndex()].type_index(),
            ffi::TypeIndex::kTVMFFINone);
  EXPECT_EQ(mnemonic_col[stdir::AttrsObj::RuntimeTypeIndex()].type_index(),
            ffi::TypeIndex::kTVMFFINone);
  EXPECT_EQ(mnemonic_col[stdir::StructureObj::RuntimeTypeIndex()].type_index(),
            ffi::TypeIndex::kTVMFFINone);
  EXPECT_EQ(mnemonic_col[stdir::ExprObj::RuntimeTypeIndex()].type_index(),
            ffi::TypeIndex::kTVMFFINone);
  EXPECT_EQ(mnemonic_col[stdir::BindObj::RuntimeTypeIndex()].type_index(),
            ffi::TypeIndex::kTVMFFINone);
}

}  // namespace
