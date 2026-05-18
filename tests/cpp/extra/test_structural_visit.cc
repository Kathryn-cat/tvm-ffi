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
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/list.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/string.h>

#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

using namespace tvm::ffi;

using WalkAction = Variant<VisitInterrupt, bool>;
using WalkExpectedAction = Expected<WalkAction>;

WalkExpectedAction WalkContinue() { return WalkAction(true); }

WalkExpectedAction WalkSkip() { return WalkAction(false); }

WalkExpectedAction WalkInterrupt(String payload) { return WalkAction(VisitInterrupt(std::move(payload))); }

ObjectRef MakeLeaf(const char* name) {
  return Function::GetGlobalRequired("testing.structural_visit_make_leaf")(String(name))
      .cast<ObjectRef>();
}

ObjectRef MakePair(ObjectRef lhs, ObjectRef rhs) {
  return Function::GetGlobalRequired("testing.structural_visit_make_pair")(lhs, rhs)
      .cast<ObjectRef>();
}

ObjectRef MakeIgnoredFields(ObjectRef visible, ObjectRef ignored, int64_t scalar) {
  return Function::GetGlobalRequired("testing.structural_visit_make_ignored_fields")(
             visible, ignored, scalar)
      .cast<ObjectRef>();
}

ObjectRef MakeDefRegionFields(ObjectRef recursive, ObjectRef plain_after_recursive,
                              ObjectRef non_recursive, ObjectRef plain_after_non_recursive) {
  return Function::GetGlobalRequired("testing.structural_visit_make_def_region_fields")(
             recursive, plain_after_recursive, non_recursive, plain_after_non_recursive)
      .cast<ObjectRef>();
}

ObjectRef MakeOpaqueOverrideParent(ObjectRef child) {
  return Function::GetGlobalRequired("testing.structural_visit_make_opaque_override_parent")(child)
      .cast<ObjectRef>();
}

ObjectRef MakeFunctionOverrideParent(ObjectRef child) {
  return Function::GetGlobalRequired("testing.structural_visit_make_function_override_parent")(
             child)
      .cast<ObjectRef>();
}

ObjectRef MakeInvalidOverrideParent(ObjectRef child) {
  return Function::GetGlobalRequired("testing.structural_visit_make_invalid_override_parent")(child)
      .cast<ObjectRef>();
}

StructuralVisitor MakeProbeVisitor() {
  return Function::GetGlobalRequired("testing.structural_visit_make_probe_visitor")()
      .cast<StructuralVisitor>();
}

StructuralVisitor MakeThrowingVisitor() {
  return Function::GetGlobalRequired("testing.structural_visit_make_throwing_visitor")()
      .cast<StructuralVisitor>();
}

void ProbeSetInterrupt(StructuralVisitor visitor, ObjectRef value, String payload) {
  Function::GetGlobalRequired("testing.structural_visit_probe_set_interrupt")(visitor, value,
                                                                              payload);
}

int64_t ProbeVisitedSize(StructuralVisitor visitor) {
  return Function::GetGlobalRequired("testing.structural_visit_probe_visited_size")(visitor)
      .cast<int64_t>();
}

ObjectRef ProbeVisited(StructuralVisitor visitor, int64_t index) {
  return Function::GetGlobalRequired("testing.structural_visit_probe_visited")(visitor, index)
      .cast<ObjectRef>();
}

TVMFFIDefRegionKind ProbeVisitedMode(StructuralVisitor visitor, int64_t index) {
  return static_cast<TVMFFIDefRegionKind>(
      Function::GetGlobalRequired("testing.structural_visit_probe_visited_mode")(visitor, index)
          .cast<int64_t>());
}

TVMFFIDefRegionKind ProbeCurrentMode(StructuralVisitor visitor) {
  return static_cast<TVMFFIDefRegionKind>(
      Function::GetGlobalRequired("testing.structural_visit_probe_current_mode")(visitor)
          .cast<int64_t>());
}

void ResetOverrideCounters() {
  Function::GetGlobalRequired("testing.structural_visit_override_reset")();
}

int64_t OpaqueOverrideCalls() {
  return Function::GetGlobalRequired("testing.structural_visit_opaque_override_calls")()
      .cast<int64_t>();
}

int64_t FunctionOverrideCalls() {
  return Function::GetGlobalRequired("testing.structural_visit_function_override_calls")()
      .cast<int64_t>();
}

bool FunctionOverrideSawVisitor() {
  return Function::GetGlobalRequired("testing.structural_visit_function_override_saw_visitor")()
      .cast<bool>();
}

bool FunctionOverrideSawValue() {
  return Function::GetGlobalRequired("testing.structural_visit_function_override_saw_value")()
      .cast<bool>();
}

ObjectRef MakeStructuralVisitVar(const char* name) {
  return Function::GetGlobalRequired("testing.structural_visit_ir_make_var")(String(name))
      .cast<ObjectRef>();
}

ObjectRef MakeStructuralVisitConst(int64_t value) {
  return Function::GetGlobalRequired("testing.structural_visit_ir_make_const")(value)
      .cast<ObjectRef>();
}

ObjectRef MakeStructuralVisitAdd(ObjectRef lhs, ObjectRef rhs) {
  return Function::GetGlobalRequired("testing.structural_visit_ir_make_add")(lhs, rhs)
      .cast<ObjectRef>();
}

ObjectRef MakeStructuralVisitSeq(Array<ObjectRef> stmts) {
  return Function::GetGlobalRequired("testing.structural_visit_ir_make_seq")(stmts)
      .cast<ObjectRef>();
}

ObjectRef MakeStructuralVisitLet(ObjectRef var, ObjectRef value, ObjectRef body) {
  return Function::GetGlobalRequired("testing.structural_visit_ir_make_let")(var, value, body)
      .cast<ObjectRef>();
}

ObjectRef MakeStructuralVisitIf(ObjectRef cond, ObjectRef then_branch, ObjectRef else_branch) {
  return Function::GetGlobalRequired("testing.structural_visit_ir_make_if")(cond, then_branch,
                                                                            else_branch)
      .cast<ObjectRef>();
}

ObjectRef MakeStructuralVisitFunc(Array<ObjectRef> params, ObjectRef body, const char* name) {
  return Function::GetGlobalRequired("testing.structural_visit_ir_make_func")(params, body,
                                                                              String(name))
      .cast<ObjectRef>();
}

ObjectRef MakeStructuralVisitCall(ObjectRef callee, Array<ObjectRef> args,
                                  const char* debug_name) {
  return Function::GetGlobalRequired("testing.structural_visit_ir_make_call")(callee, args,
                                                                              String(debug_name))
      .cast<ObjectRef>();
}

String IRKind(const ObjectRef& node) {
  return Function::GetGlobalRequired("testing.structural_visit_ir_kind")(node).cast<String>();
}

String IRLabel(const ObjectRef& node) {
  return Function::GetGlobalRequired("testing.structural_visit_ir_label")(node).cast<String>();
}

String IRVarName(const ObjectRef& node) {
  return Function::GetGlobalRequired("testing.structural_visit_ir_var_name")(node).cast<String>();
}

ObjectRef MakeIntegratedFunc(ObjectRef x, ObjectRef y) {
  ObjectRef body =
      MakeStructuralVisitAdd(x, MakeStructuralVisitAdd(y, MakeStructuralVisitConst(1)));
  return MakeStructuralVisitFunc({x, y}, MakeStructuralVisitSeq({body}), "main");
}

StructuralVisitor MakeIRRecordingVisitor() {
  return Function::GetGlobalRequired("testing.structural_visit_ir_make_recording_visitor")()
      .cast<StructuralVisitor>();
}

Array<String> IRRecordingTrace(StructuralVisitor visitor) {
  return Function::GetGlobalRequired("testing.structural_visit_ir_recording_trace")(visitor)
      .cast<Array<String>>();
}

Array<String> IRRecordingVarModes(StructuralVisitor visitor) {
  return Function::GetGlobalRequired("testing.structural_visit_ir_recording_var_modes")(visitor)
      .cast<Array<String>>();
}

void ExpectTrace(const std::vector<std::string>& actual,
                 std::initializer_list<const char*> expected) {
  ASSERT_EQ(actual.size(), expected.size());
  size_t i = 0;
  for (const char* item : expected) {
    EXPECT_EQ(actual[i], item);
    ++i;
  }
}

void ExpectTrace(const Array<String>& actual, std::initializer_list<const char*> expected) {
  ASSERT_EQ(actual.size(), expected.size());
  size_t i = 0;
  for (const char* item : expected) {
    EXPECT_EQ(actual[i].operator std::string(), item);
    ++i;
  }
}

// ---------------------------------------------------------------------------
// StructuralVisitorObj vtable behavior.
// ---------------------------------------------------------------------------

TEST(StructuralVisitor, CustomVTableIsCalled) {
  ObjectRef leaf = MakeLeaf("leaf");
  StructuralVisitor visitor = MakeProbeVisitor();

  Optional<VisitInterrupt> result = visitor.Visit(leaf);

  EXPECT_FALSE(result.has_value());
  ASSERT_EQ(ProbeVisitedSize(visitor), 1);
  EXPECT_TRUE(ProbeVisited(visitor, 0).same_as(leaf));
}

TEST(StructuralVisitor, VisitExpectedReturnsSuccess) {
  ObjectRef leaf = MakeLeaf("leaf");
  StructuralVisitor visitor = MakeProbeVisitor();

  Expected<Optional<VisitInterrupt>> result = visitor.VisitExpected(leaf);

  ASSERT_TRUE(result.is_ok());
  EXPECT_FALSE(result.value().has_value());
  ASSERT_EQ(ProbeVisitedSize(visitor), 1);
  EXPECT_TRUE(ProbeVisited(visitor, 0).same_as(leaf));
}

TEST(StructuralVisitor, CustomVTableInterruptPropagates) {
  ObjectRef leaf = MakeLeaf("leaf");
  StructuralVisitor visitor = MakeProbeVisitor();
  ProbeSetInterrupt(visitor, leaf, "custom stop");

  Optional<VisitInterrupt> result = visitor.Visit(leaf);

  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value()->value.cast<String>(), "custom stop");
  ASSERT_EQ(ProbeVisitedSize(visitor), 1);
  EXPECT_TRUE(ProbeVisited(visitor, 0).same_as(leaf));
}

TEST(StructuralVisitor, VisitExpectedReturnsInterrupt) {
  ObjectRef leaf = MakeLeaf("leaf");
  StructuralVisitor visitor = MakeProbeVisitor();
  ProbeSetInterrupt(visitor, leaf, "custom stop");

  Expected<Optional<VisitInterrupt>> result = visitor.VisitExpected(leaf);

  ASSERT_TRUE(result.is_ok());
  Optional<VisitInterrupt> interrupt = result.value();
  ASSERT_TRUE(interrupt.has_value());
  EXPECT_EQ(interrupt.value()->value.cast<String>(), "custom stop");
  ASSERT_EQ(ProbeVisitedSize(visitor), 1);
  EXPECT_TRUE(ProbeVisited(visitor, 0).same_as(leaf));
}

TEST(StructuralVisitor, VisitExpectedCatchesCustomVTableError) {
  ObjectRef leaf = MakeLeaf("leaf");
  StructuralVisitor visitor = MakeThrowingVisitor();

  Expected<Optional<VisitInterrupt>> result = visitor.VisitExpected(leaf);

  ASSERT_TRUE(result.is_err());
  Error err = result.error();
  EXPECT_EQ(err.kind(), "ValueError");
  EXPECT_NE(err.message().find("throwing visitor saw"), std::string::npos);
}

TEST(StructuralVisitor, VisitThrowsCustomVTableError) {
  ObjectRef leaf = MakeLeaf("leaf");
  StructuralVisitor visitor = MakeThrowingVisitor();

  try {
    visitor.Visit(leaf);
    FAIL() << "Expected Visit wrapper to throw";
  } catch (const Error& err) {
    EXPECT_EQ(err.kind(), "ValueError");
    EXPECT_NE(err.message().find("throwing visitor saw"), std::string::npos);
  }
}

// ---------------------------------------------------------------------------
// DefaultVisit type-attribute behavior.
// ---------------------------------------------------------------------------

TEST(StructuralVisitor, DefaultVisitUsesOpaqueTypeAttrBeforeReflection) {
  ResetOverrideCounters();

  ObjectRef child = MakeLeaf("child");
  ObjectRef parent = MakeOpaqueOverrideParent(child);
  StructuralVisitor visitor = MakeProbeVisitor();
  ProbeSetInterrupt(visitor, child, "child stopped");

  Optional<VisitInterrupt> result = visitor.DefaultVisit(parent);

  EXPECT_FALSE(result.has_value());
  EXPECT_EQ(OpaqueOverrideCalls(), 1);
  EXPECT_EQ(ProbeVisitedSize(visitor), 0);
}

TEST(StructuralVisitor, DefaultVisitUsesFunctionTypeAttrBeforeReflection) {
  ResetOverrideCounters();

  ObjectRef child = MakeLeaf("child");
  ObjectRef parent = MakeFunctionOverrideParent(child);
  StructuralVisitor visitor = MakeProbeVisitor();
  ProbeSetInterrupt(visitor, child, "child stopped");

  Optional<VisitInterrupt> result = visitor.DefaultVisit(parent);

  EXPECT_FALSE(result.has_value());
  EXPECT_EQ(FunctionOverrideCalls(), 1);
  EXPECT_TRUE(FunctionOverrideSawVisitor());
  EXPECT_TRUE(FunctionOverrideSawValue());
  EXPECT_EQ(ProbeVisitedSize(visitor), 0);
}

TEST(StructuralVisitor, DefaultVisitExpectedUsesFunctionTypeAttrBeforeReflection) {
  ResetOverrideCounters();

  ObjectRef child = MakeLeaf("child");
  ObjectRef parent = MakeFunctionOverrideParent(child);
  StructuralVisitor visitor = MakeProbeVisitor();
  ProbeSetInterrupt(visitor, child, "child stopped");

  Expected<Optional<VisitInterrupt>> result = visitor.DefaultVisitExpected(parent);

  ASSERT_TRUE(result.is_ok());
  EXPECT_FALSE(result.value().has_value());
  EXPECT_EQ(FunctionOverrideCalls(), 1);
  EXPECT_TRUE(FunctionOverrideSawVisitor());
  EXPECT_TRUE(FunctionOverrideSawValue());
  EXPECT_EQ(ProbeVisitedSize(visitor), 0);
}

TEST(StructuralVisitor, InvalidStructuralVisitTypeAttrThrows) {
  ObjectRef parent = MakeInvalidOverrideParent(MakeLeaf("child"));
  StructuralVisitor visitor;

  try {
    visitor.DefaultVisit(parent);
    FAIL() << "Expected invalid structural visit type attribute to throw";
  } catch (const std::exception& err) {
    EXPECT_NE(std::string(err.what()).find(
                  "__ffi_structural_visit__ must be an opaque function pointer or ffi.Function"),
              std::string::npos);
  }
}

TEST(StructuralVisitor, InvalidStructuralVisitTypeAttrReturnsError) {
  ObjectRef parent = MakeInvalidOverrideParent(MakeLeaf("child"));
  StructuralVisitor visitor;

  Expected<Optional<VisitInterrupt>> result = visitor.DefaultVisitExpected(parent);

  ASSERT_TRUE(result.is_err());
  Error err = result.error();
  EXPECT_EQ(err.kind(), "TypeError");
  EXPECT_NE(err.message().find(
                "__ffi_structural_visit__ must be an opaque function pointer or ffi.Function"),
            std::string::npos);
}

TEST(StructuralVisitor, VisitExpectedPropagatesDefaultVisitError) {
  ObjectRef parent = MakeInvalidOverrideParent(MakeLeaf("child"));
  StructuralVisitor visitor;

  Expected<Optional<VisitInterrupt>> result = visitor.VisitExpected(parent);

  ASSERT_TRUE(result.is_err());
  Error err = result.error();
  EXPECT_EQ(err.kind(), "TypeError");
  EXPECT_NE(err.message().find(
                "__ffi_structural_visit__ must be an opaque function pointer or ffi.Function"),
            std::string::npos);
}

// ---------------------------------------------------------------------------
// DefaultVisit reflected-field behavior.
// ---------------------------------------------------------------------------

TEST(StructuralVisitor, ReflectedTraversalSkipsIgnoredFields) {
  ObjectRef visible = MakeLeaf("visible");
  ObjectRef ignored = MakeLeaf("ignored");
  ObjectRef parent = MakeIgnoredFields(visible, ignored, 42);
  StructuralVisitor visitor = MakeProbeVisitor();
  ProbeSetInterrupt(visitor, ignored, "ignored stopped");

  Optional<VisitInterrupt> result = visitor.DefaultVisit(parent);

  EXPECT_FALSE(result.has_value());
  ASSERT_EQ(ProbeVisitedSize(visitor), 1);
  EXPECT_TRUE(ProbeVisited(visitor, 0).same_as(visible));
}

TEST(StructuralVisitor, ReflectedTraversalStopsOnFirstInterrupt) {
  ObjectRef lhs = MakeLeaf("lhs");
  ObjectRef rhs = MakeLeaf("rhs");
  ObjectRef parent = MakePair(lhs, rhs);
  StructuralVisitor visitor = MakeProbeVisitor();
  ProbeSetInterrupt(visitor, lhs, "lhs stopped");

  Optional<VisitInterrupt> result = visitor.DefaultVisit(parent);

  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value()->value.cast<String>(), "lhs stopped");
  ASSERT_EQ(ProbeVisitedSize(visitor), 1);
  EXPECT_TRUE(ProbeVisited(visitor, 0).same_as(lhs));
}

// ---------------------------------------------------------------------------
// Def-region scoping behavior.
// ---------------------------------------------------------------------------

TEST(StructuralVisitor, WithDefRegionKindRestoresNestedScopes) {
  StructuralVisitor visitor = MakeProbeVisitor();
  EXPECT_EQ(ProbeCurrentMode(visitor), kTVMFFIDefRegionKindNone);

  Optional<VisitInterrupt> result = visitor.WithDefRegionKind(
      kTVMFFIDefRegionKindRecursive, [&]() -> Optional<VisitInterrupt> {
        EXPECT_EQ(ProbeCurrentMode(visitor), kTVMFFIDefRegionKindRecursive);
        Optional<VisitInterrupt> inner = visitor.WithDefRegionKind(
            kTVMFFIDefRegionKindNonRecursive, [&]() -> Optional<VisitInterrupt> {
              EXPECT_EQ(ProbeCurrentMode(visitor), kTVMFFIDefRegionKindNonRecursive);
              return std::nullopt;
            });
        EXPECT_FALSE(inner.has_value());
        EXPECT_EQ(ProbeCurrentMode(visitor), kTVMFFIDefRegionKindRecursive);
        return std::nullopt;
      });

  EXPECT_FALSE(result.has_value());
  EXPECT_EQ(ProbeCurrentMode(visitor), kTVMFFIDefRegionKindNone);
}

TEST(StructuralVisitor, WithDefRegionKindRestoresAfterThrow) {
  StructuralVisitor visitor = MakeProbeVisitor();

  try {
    visitor.WithDefRegionKind(kTVMFFIDefRegionKindRecursive,
                              [&]() -> Optional<VisitInterrupt> {
                                EXPECT_EQ(ProbeCurrentMode(visitor),
                                          kTVMFFIDefRegionKindRecursive);
                                throw std::runtime_error("test exception");
                              });
    FAIL() << "Expected WithDefRegionKind wrapper to throw";
  } catch (const Error& err) {
    EXPECT_EQ(err.kind(), "InternalError");
    EXPECT_EQ(err.message(), "test exception");
  }
  EXPECT_EQ(ProbeCurrentMode(visitor), kTVMFFIDefRegionKindNone);
}

TEST(StructuralVisitor, WithDefRegionKindExpectedCatchesAndRestoresAfterThrow) {
  StructuralVisitor visitor = MakeProbeVisitor();

  Expected<Optional<VisitInterrupt>> result = visitor.WithDefRegionKindExpected(
      kTVMFFIDefRegionKindRecursive, [&]() -> Expected<Optional<VisitInterrupt>> {
        EXPECT_EQ(ProbeCurrentMode(visitor), kTVMFFIDefRegionKindRecursive);
        throw Error("ValueError", "expected scope stop", "");
      });

  ASSERT_TRUE(result.is_err());
  EXPECT_EQ(result.error().kind(), "ValueError");
  EXPECT_EQ(ProbeCurrentMode(visitor), kTVMFFIDefRegionKindNone);
}

TEST(StructuralVisitor, ReflectedDefRegionFlagsDoNotLeakAcrossSiblings) {
  ObjectRef recursive = MakeLeaf("recursive");
  ObjectRef plain_after_recursive = MakeLeaf("plain_after_recursive");
  ObjectRef non_recursive = MakeLeaf("non_recursive");
  ObjectRef plain_after_non_recursive = MakeLeaf("plain_after_non_recursive");
  ObjectRef parent = MakeDefRegionFields(recursive, plain_after_recursive, non_recursive,
                                         plain_after_non_recursive);
  StructuralVisitor visitor = MakeProbeVisitor();

  Optional<VisitInterrupt> result = visitor.DefaultVisit(parent);

  EXPECT_FALSE(result.has_value());
  ASSERT_EQ(ProbeVisitedSize(visitor), 4);
  EXPECT_TRUE(ProbeVisited(visitor, 0).same_as(recursive));
  EXPECT_EQ(ProbeVisitedMode(visitor, 0), kTVMFFIDefRegionKindRecursive);
  EXPECT_TRUE(ProbeVisited(visitor, 1).same_as(plain_after_recursive));
  EXPECT_EQ(ProbeVisitedMode(visitor, 1), kTVMFFIDefRegionKindNone);
  EXPECT_TRUE(ProbeVisited(visitor, 2).same_as(non_recursive));
  EXPECT_EQ(ProbeVisitedMode(visitor, 2), kTVMFFIDefRegionKindNonRecursive);
  EXPECT_TRUE(ProbeVisited(visitor, 3).same_as(plain_after_non_recursive));
  EXPECT_EQ(ProbeVisitedMode(visitor, 3), kTVMFFIDefRegionKindNone);
  EXPECT_EQ(ProbeCurrentMode(visitor), kTVMFFIDefRegionKindNone);
}

TEST(StructuralVisitor, DefRegionRestoresAfterChildInterrupt) {
  ObjectRef recursive = MakeLeaf("recursive");
  ObjectRef plain_after_recursive = MakeLeaf("plain_after_recursive");
  ObjectRef non_recursive = MakeLeaf("non_recursive");
  ObjectRef plain_after_non_recursive = MakeLeaf("plain_after_non_recursive");
  ObjectRef parent = MakeDefRegionFields(recursive, plain_after_recursive, non_recursive,
                                         plain_after_non_recursive);
  StructuralVisitor visitor = MakeProbeVisitor();
  ProbeSetInterrupt(visitor, recursive, "recursive stopped");

  Optional<VisitInterrupt> result = visitor.WithDefRegionKind(
      kTVMFFIDefRegionKindNonRecursive, [&]() -> Optional<VisitInterrupt> {
        Optional<VisitInterrupt> interrupt = visitor.DefaultVisit(parent);
        EXPECT_TRUE(interrupt.has_value());
        EXPECT_EQ(ProbeCurrentMode(visitor), kTVMFFIDefRegionKindNonRecursive);
        return interrupt;
      });

  ASSERT_TRUE(result.has_value());
  ASSERT_EQ(ProbeVisitedSize(visitor), 1);
  EXPECT_TRUE(ProbeVisited(visitor, 0).same_as(recursive));
  EXPECT_EQ(ProbeVisitedMode(visitor, 0), kTVMFFIDefRegionKindRecursive);
  EXPECT_EQ(ProbeCurrentMode(visitor), kTVMFFIDefRegionKindNone);
}

// ---------------------------------------------------------------------------
// structuralWalk control-flow behavior.
// ---------------------------------------------------------------------------

TEST(StructuralVisitor, StructuralWalkPreOrderSkipSkipsChildren) {
  ObjectRef lhs = MakeLeaf("lhs");
  ObjectRef rhs = MakeLeaf("rhs");
  ObjectRef root = MakePair(lhs, rhs);
  std::vector<std::string> visited;

  Optional<VisitInterrupt> result =
      structuralWalk<ObjectRef>(root,
                                [&](const ObjectRef& node) -> WalkExpectedAction {
                                  if (node.same_as(root)) {
                                    visited.push_back("pair");
                                    return WalkSkip();
                                  }
                                  visited.push_back("child");
                                  return WalkContinue();
                                },
                                WalkOrder::kPreOrder);

  EXPECT_FALSE(result.has_value());
  ASSERT_EQ(visited.size(), 1U);
  EXPECT_EQ(visited[0], "pair");
}

TEST(StructuralVisitor, StructuralWalkPostOrderInterruptStopsParentCallback) {
  ObjectRef lhs = MakeLeaf("lhs");
  ObjectRef rhs = MakeLeaf("rhs");
  ObjectRef root = MakePair(lhs, rhs);
  std::vector<std::string> visited;

  Optional<VisitInterrupt> result =
      structuralWalk<ObjectRef>(root,
                                [&](const ObjectRef& node) -> WalkExpectedAction {
                                  if (node.same_as(lhs)) {
                                    visited.push_back("lhs");
                                    return WalkInterrupt(String("child stop"));
                                  }
                                  if (node.same_as(rhs)) {
                                    visited.push_back("rhs");
                                    return WalkContinue();
                                  }
                                  if (node.same_as(root)) {
                                    visited.push_back("pair");
                                  }
                                  return WalkContinue();
                                },
                                WalkOrder::kPostOrder);

  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value()->value.cast<String>(), "child stop");
  ASSERT_EQ(visited.size(), 1U);
  EXPECT_EQ(visited[0], "lhs");
}

TEST(StructuralVisitor, StructuralWalkPostOrderFalseDoesNotStopTraversal) {
  ObjectRef lhs = MakeLeaf("lhs");
  ObjectRef rhs = MakeLeaf("rhs");
  ObjectRef root = MakePair(lhs, rhs);
  std::vector<std::string> visited;

  Optional<VisitInterrupt> result =
      structuralWalk<ObjectRef>(root,
                                [&](const ObjectRef& node) -> WalkExpectedAction {
                                  if (node.same_as(lhs)) {
                                    visited.push_back("lhs");
                                    return WalkSkip();
                                  }
                                  if (node.same_as(rhs)) {
                                    visited.push_back("rhs");
                                    return WalkSkip();
                                  }
                                  if (node.same_as(root)) {
                                    visited.push_back("pair");
                                  }
                                  return WalkContinue();
                                },
                                WalkOrder::kPostOrder);

  EXPECT_FALSE(result.has_value());
  ASSERT_EQ(visited.size(), 3U);
  EXPECT_EQ(visited[0], "lhs");
  EXPECT_EQ(visited[1], "rhs");
  EXPECT_EQ(visited[2], "pair");
}

TEST(StructuralVisitor, StructuralWalkMatchesFirstCompatibleVariadicType) {
  ObjectRef lhs = MakeStructuralVisitConst(1);
  ObjectRef rhs = MakeStructuralVisitConst(2);
  List<ObjectRef> list = {lhs};
  Array<ObjectRef> root = {list, rhs};
  std::vector<std::string> trace;

  Optional<VisitInterrupt> result =
      structuralWalk<ArrayObj, ListObj, ObjectRef>(
          root,
          [&](const auto& node) -> WalkExpectedAction {
            using Node = std::decay_t<decltype(node)>;
            if constexpr (std::is_same_v<Node, const ArrayObj*>) {
              trace.push_back("array");
            } else if constexpr (std::is_same_v<Node, const ListObj*>) {
              trace.push_back("list");
            } else {
              trace.push_back("object:" + node.GetTypeKey());
            }
            return WalkContinue();
          },
          WalkOrder::kPreOrder);

  EXPECT_FALSE(result.has_value());
  ExpectTrace(trace,
              {"array", "list", "object:testing.StructuralVisit.Const",
               "object:testing.StructuralVisit.Const"});
}

TEST(StructuralVisitor, StructuralWalkExpectedReturnsSuccess) {
  ObjectRef lhs = MakeLeaf("lhs");
  ObjectRef rhs = MakeLeaf("rhs");
  ObjectRef root = MakePair(lhs, rhs);
  std::vector<std::string> visited;

  Expected<Optional<VisitInterrupt>> result =
      structuralWalkExpected<ObjectRef>(root,
                                        [&](const ObjectRef& node) -> WalkExpectedAction {
                                          if (node.same_as(root)) {
                                            visited.push_back("pair");
                                          } else if (node.same_as(lhs)) {
                                            visited.push_back("lhs");
                                          } else if (node.same_as(rhs)) {
                                            visited.push_back("rhs");
                                          }
                                          return WalkContinue();
                                        },
                                        WalkOrder::kPreOrder);

  ASSERT_TRUE(result.is_ok());
  EXPECT_FALSE(result.value().has_value());
  ExpectTrace(visited, {"pair", "lhs", "rhs"});
}

TEST(StructuralVisitor, StructuralWalkExpectedReturnsCallbackInterrupt) {
  ObjectRef lhs = MakeLeaf("lhs");
  ObjectRef rhs = MakeLeaf("rhs");
  ObjectRef root = MakePair(lhs, rhs);
  std::vector<std::string> visited;

  Expected<Optional<VisitInterrupt>> result =
      structuralWalkExpected<ObjectRef>(root,
                                        [&](const ObjectRef& node) -> WalkExpectedAction {
                                          if (node.same_as(lhs)) {
                                            visited.push_back("lhs");
                                            return WalkInterrupt(String("found lhs"));
                                          }
                                          if (node.same_as(rhs)) {
                                            visited.push_back("rhs");
                                          }
                                          return WalkContinue();
                                        },
                                        WalkOrder::kPostOrder);

  ASSERT_TRUE(result.is_ok());
  Optional<VisitInterrupt> interrupt = result.value();
  ASSERT_TRUE(interrupt.has_value());
  EXPECT_EQ(interrupt.value()->value.cast<String>(), "found lhs");
  ExpectTrace(visited, {"lhs"});
}

TEST(StructuralVisitor, StructuralWalkExpectedReturnsCallbackError) {
  ObjectRef lhs = MakeLeaf("lhs");
  ObjectRef rhs = MakeLeaf("rhs");
  ObjectRef root = MakePair(lhs, rhs);
  std::vector<std::string> visited;

  Expected<Optional<VisitInterrupt>> result =
      structuralWalkExpected<ObjectRef>(root,
                                        [&](const ObjectRef& node) -> WalkExpectedAction {
                                          if (node.same_as(lhs)) {
                                            visited.push_back("lhs");
                                            return Unexpected(
                                                Error("ValueError", "walk callback failed", ""));
                                          }
                                          if (node.same_as(rhs)) {
                                            visited.push_back("rhs");
                                          }
                                          return WalkContinue();
                                        },
                                        WalkOrder::kPostOrder);

  ASSERT_TRUE(result.is_err());
  EXPECT_EQ(result.error().kind(), "ValueError");
  EXPECT_EQ(result.error().message(), "walk callback failed");
  ExpectTrace(visited, {"lhs"});
}

TEST(StructuralVisitor, StructuralWalkExpectedCatchesCallbackThrow) {
  ObjectRef root = MakeLeaf("root");

  Expected<Optional<VisitInterrupt>> result =
      structuralWalkExpected<ObjectRef>(root,
                                        [&](const ObjectRef&) -> WalkExpectedAction {
                                          throw std::runtime_error("walk callback threw");
                                        },
                                        WalkOrder::kPreOrder);

  ASSERT_TRUE(result.is_err());
  EXPECT_EQ(result.error().kind(), "InternalError");
  EXPECT_EQ(result.error().message(), "walk callback threw");
}

TEST(StructuralVisitor, StructuralWalkThrowsCallbackError) {
  ObjectRef root = MakeLeaf("root");

  try {
    structuralWalk<ObjectRef>(root,
                              [&](const ObjectRef&) -> WalkExpectedAction {
                                return Unexpected(Error("ValueError", "walk wrapper failed", ""));
                              },
                              WalkOrder::kPreOrder);
    FAIL() << "Expected structuralWalk wrapper to throw";
  } catch (const Error& err) {
    EXPECT_EQ(err.kind(), "ValueError");
    EXPECT_EQ(err.message(), "walk wrapper failed");
  }
}

// ---------------------------------------------------------------------------
// Integrated IR traversal behavior.
// ---------------------------------------------------------------------------

TEST(StructuralVisitor, IntegratedVisitReflectsFuncParamsAndBody) {
  ObjectRef x = MakeStructuralVisitVar("x");
  ObjectRef y = MakeStructuralVisitVar("y");
  ObjectRef func = MakeIntegratedFunc(x, y);
  StructuralVisitor visitor = MakeIRRecordingVisitor();

  Optional<VisitInterrupt> result = visitor.Visit(func);

  EXPECT_FALSE(result.has_value());
  ExpectTrace(IRRecordingTrace(visitor),
              {"func", "var:x", "var:y", "seq", "add", "var:x", "add", "var:y", "const:1"});
}

TEST(StructuralVisitor, IntegratedFuncParamsUseDefRegionThroughArray) {
  ObjectRef x = MakeStructuralVisitVar("x");
  ObjectRef y = MakeStructuralVisitVar("y");
  ObjectRef func = MakeIntegratedFunc(x, y);
  StructuralVisitor visitor = MakeIRRecordingVisitor();

  Optional<VisitInterrupt> result = visitor.Visit(func);

  EXPECT_FALSE(result.has_value());
  ExpectTrace(IRRecordingVarModes(visitor), {"x:recursive", "y:recursive", "x:none", "y:none"});
}

TEST(StructuralVisitor, IntegratedLetVarUsesCustomDefRegion) {
  ObjectRef z = MakeStructuralVisitVar("z");
  ObjectRef x = MakeStructuralVisitVar("x");
  ObjectRef y = MakeStructuralVisitVar("y");
  ObjectRef let = MakeStructuralVisitLet(z, MakeStructuralVisitAdd(x, MakeStructuralVisitConst(1)),
                                         MakeStructuralVisitSeq({y}));
  StructuralVisitor visitor = MakeIRRecordingVisitor();

  Optional<VisitInterrupt> result = visitor.Visit(let);

  EXPECT_FALSE(result.has_value());
  ExpectTrace(IRRecordingVarModes(visitor), {"z:non_recursive", "x:none", "y:none"});
}

TEST(StructuralVisitor, IntegratedVisitUsesCallTypeAttrOverride) {
  ObjectRef f = MakeStructuralVisitVar("f");
  ObjectRef x = MakeStructuralVisitVar("x");
  ObjectRef y = MakeStructuralVisitVar("y");
  ObjectRef call = MakeStructuralVisitCall(f, {x, y}, "debug_call");
  StructuralVisitor visitor = MakeIRRecordingVisitor();

  Optional<VisitInterrupt> result = visitor.Visit(call);

  EXPECT_FALSE(result.has_value());
  ExpectTrace(IRRecordingTrace(visitor), {"call", "var:y", "var:x"});
}

TEST(StructuralVisitor, IntegratedVisitUsesIfThenElseTypeAttrOverride) {
  ObjectRef cond = MakeStructuralVisitVar("cond");
  ObjectRef then_var = MakeStructuralVisitVar("then");
  ObjectRef else_var = MakeStructuralVisitVar("else");
  ObjectRef node = MakeStructuralVisitIf(cond, MakeStructuralVisitSeq({then_var}),
                                         MakeStructuralVisitSeq({else_var}));
  StructuralVisitor visitor = MakeIRRecordingVisitor();

  Optional<VisitInterrupt> result = visitor.Visit(node);

  EXPECT_FALSE(result.has_value());
  ExpectTrace(IRRecordingTrace(visitor), {"if", "var:cond", "seq", "var:else", "seq", "var:then"});
}

TEST(StructuralVisitor, IntegratedWalkCollectsVarsThroughReflectionAndContainers) {
  ObjectRef x = MakeStructuralVisitVar("x");
  ObjectRef y = MakeStructuralVisitVar("y");
  ObjectRef func = MakeIntegratedFunc(x, y);
  std::vector<std::string> vars;

  Optional<VisitInterrupt> result =
      structuralWalk<ObjectRef>(func,
                                [&](const ObjectRef& node) -> WalkExpectedAction {
                                  if (IRKind(node) == "var") {
                                    vars.push_back(IRVarName(node).operator std::string());
                                  }
                                  return WalkContinue();
                                },
                                WalkOrder::kPreOrder);

  EXPECT_FALSE(result.has_value());
  ExpectTrace(vars, {"x", "y", "x", "y"});
}

TEST(StructuralVisitor, IntegratedWalkPreOrderSkipPrunesSubtree) {
  ObjectRef x = MakeStructuralVisitVar("x");
  ObjectRef y = MakeStructuralVisitVar("y");
  ObjectRef inner = MakeStructuralVisitAdd(y, MakeStructuralVisitConst(1));
  ObjectRef root = MakeStructuralVisitAdd(x, inner);
  std::vector<std::string> trace;

  Optional<VisitInterrupt> result =
      structuralWalk<ObjectRef>(root,
                                [&](const ObjectRef& node) -> WalkExpectedAction {
                                  if (node.same_as(root)) {
                                    trace.push_back("outer_add");
                                    return WalkContinue();
                                  }
                                  if (node.same_as(inner)) {
                                    trace.push_back("inner_add");
                                    return WalkSkip();
                                  }
                                  std::string label = IRLabel(node).operator std::string();
                                  if (!label.empty()) {
                                    trace.push_back(std::move(label));
                                  }
                                  return WalkContinue();
                                },
                                WalkOrder::kPreOrder);

  EXPECT_FALSE(result.has_value());
  ExpectTrace(trace, {"outer_add", "var:x", "inner_add"});
}

TEST(StructuralVisitor, IntegratedWalkInterruptStopsTraversal) {
  ObjectRef x = MakeStructuralVisitVar("x");
  ObjectRef y = MakeStructuralVisitVar("y");
  ObjectRef z = MakeStructuralVisitVar("z");
  ObjectRef root = MakeStructuralVisitAdd(x, MakeStructuralVisitAdd(y, z));
  std::vector<std::string> vars;

  Optional<VisitInterrupt> result =
      structuralWalk<ObjectRef>(root,
                                [&](const ObjectRef& node) -> WalkExpectedAction {
                                  if (IRKind(node) == "var") {
                                    std::string name = IRVarName(node).operator std::string();
                                    vars.push_back(name);
                                    if (name == "y") {
                                      return WalkInterrupt(String("found y"));
                                    }
                                  }
                                  return WalkContinue();
                                },
                                WalkOrder::kPreOrder);

  ASSERT_TRUE(result.has_value());
  vars.insert(vars.begin(),
              "interrupt:" + result.value()->value.cast<String>().operator std::string());
  ExpectTrace(vars, {"interrupt:found y", "x", "y"});
}

TEST(StructuralVisitor, IntegratedWalkPostOrderChildrenBeforeParent) {
  ObjectRef x = MakeStructuralVisitVar("x");
  ObjectRef y = MakeStructuralVisitVar("y");
  ObjectRef inner = MakeStructuralVisitAdd(x, MakeStructuralVisitConst(1));
  ObjectRef root = MakeStructuralVisitAdd(inner, y);
  std::vector<std::string> trace;

  Optional<VisitInterrupt> result =
      structuralWalk<ObjectRef>(root,
                                [&](const ObjectRef& node) -> WalkExpectedAction {
                                  if (node.same_as(inner)) {
                                    trace.push_back("inner_add");
                                  } else if (node.same_as(root)) {
                                    trace.push_back("outer_add");
                                  } else {
                                    std::string label = IRLabel(node).operator std::string();
                                    if (!label.empty()) {
                                      trace.push_back(std::move(label));
                                    }
                                  }
                                  return WalkContinue();
                                },
                                WalkOrder::kPostOrder);

  EXPECT_FALSE(result.has_value());
  ExpectTrace(trace, {"var:x", "const:1", "inner_add", "var:y", "outer_add"});
}

TEST(StructuralVisitor, IntegratedWalkAcrossLetIfSeqAndCall) {
  ObjectRef f = MakeStructuralVisitVar("f");
  ObjectRef x = MakeStructuralVisitVar("x");
  ObjectRef y = MakeStructuralVisitVar("y");
  ObjectRef z = MakeStructuralVisitVar("z");
  ObjectRef let =
      MakeStructuralVisitLet(z, MakeStructuralVisitAdd(x, MakeStructuralVisitConst(1)),
                             MakeStructuralVisitSeq({MakeStructuralVisitIf(
                                 z, MakeStructuralVisitSeq({y}),
                                 MakeStructuralVisitSeq(
                                     {MakeStructuralVisitCall(f, {x, z}, "else_call")}))}));
  ObjectRef func =
      MakeStructuralVisitFunc({x, y}, MakeStructuralVisitSeq({let, MakeStructuralVisitConst(0)}),
                              "main");
  std::vector<std::string> vars;

  Optional<VisitInterrupt> result =
      structuralWalk<ObjectRef>(func,
                                [&](const ObjectRef& node) -> WalkExpectedAction {
                                  if (IRKind(node) == "var") {
                                    vars.push_back(IRVarName(node).operator std::string());
                                  }
                                  return WalkContinue();
                                },
                                WalkOrder::kPreOrder);

  EXPECT_FALSE(result.has_value());
  ExpectTrace(vars, {"x", "y", "z", "x", "z", "z", "x", "y"});
}

TEST(StructuralVisitor, IntegratedWalkCountsVars) {
  ObjectRef x = MakeStructuralVisitVar("x");
  ObjectRef y = MakeStructuralVisitVar("y");
  ObjectRef root =
      MakeStructuralVisitSeq({MakeStructuralVisitAdd(x, MakeStructuralVisitConst(1)),
                              MakeStructuralVisitAdd(y, x)});
  int64_t num_vars = 0;

  Optional<VisitInterrupt> result =
      structuralWalk<ObjectRef>(root,
                                [&](const ObjectRef& node) -> WalkExpectedAction {
                                  if (IRKind(node) == "var") {
                                    ++num_vars;
                                  }
                                  return WalkContinue();
                                },
                                WalkOrder::kPreOrder);

  EXPECT_FALSE(result.has_value());
  EXPECT_EQ(num_vars, 3);
}

TEST(StructuralVisitor, IntegratedWalkUsesCallTypeAttrOverride) {
  ObjectRef f = MakeStructuralVisitVar("f");
  ObjectRef x = MakeStructuralVisitVar("x");
  ObjectRef y = MakeStructuralVisitVar("y");
  ObjectRef call = MakeStructuralVisitCall(f, {x, y}, "debug_call");
  std::vector<std::string> vars;

  Optional<VisitInterrupt> result =
      structuralWalk<ObjectRef>(call,
                                [&](const ObjectRef& node) -> WalkExpectedAction {
                                  if (IRKind(node) == "var") {
                                    vars.push_back(IRVarName(node).operator std::string());
                                  }
                                  return WalkContinue();
                                },
                                WalkOrder::kPreOrder);

  EXPECT_FALSE(result.has_value());
  ExpectTrace(vars, {"y", "x"});
}

TEST(StructuralVisitor, IntegratedWalkSkipPreventsCallTypeAttrTraversal) {
  ObjectRef f = MakeStructuralVisitVar("f");
  ObjectRef x = MakeStructuralVisitVar("x");
  ObjectRef y = MakeStructuralVisitVar("y");
  ObjectRef call = MakeStructuralVisitCall(f, {x, y}, "debug_call");
  std::vector<std::string> trace;

  Optional<VisitInterrupt> result =
      structuralWalk<ObjectRef>(call,
                                [&](const ObjectRef& node) -> WalkExpectedAction {
                                  if (node.same_as(call)) {
                                    trace.push_back("call");
                                    return WalkSkip();
                                  }
                                  if (IRKind(node) == "var") {
                                    trace.push_back("var:" + IRVarName(node).operator std::string());
                                  }
                                  return WalkContinue();
                                },
                                WalkOrder::kPreOrder);

  EXPECT_FALSE(result.has_value());
  ExpectTrace(trace, {"call"});
}

}  // namespace
