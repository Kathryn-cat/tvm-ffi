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
#include <tvm/ffi/function.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/string.h>

#include <string>
#include <vector>

namespace {

using namespace tvm::ffi;

TEST(StructuralVisitor, RegisteredVisitTrace) {
  ObjectRef root =
      Function::GetGlobalRequired("testing.make_structural_visit_ir")().cast<ObjectRef>();
  Array<String> trace =
      Function::GetGlobalRequired("testing.structural_visit_trace")(root).cast<Array<String>>();

  std::vector<std::string> expected = {"Func|none",   "Var(x)|recursive", "Var(y)|recursive",
                                       "Add|none",    "Var(x)|none",      "Add|none",
                                       "Var(y)|none", "Const(1)|none"};

  ASSERT_EQ(trace.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(std::string(trace[i]), expected[i]) << "trace index " << i;
  }
}

}  // namespace
