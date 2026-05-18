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
 * \file src/ffi/extra/structural_visit.cc
 * \brief Structural visit implementation.
 */
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/accessor.h>

namespace tvm {
namespace ffi {

namespace details {
namespace {

TVMFFIAny EncodeStructuralVisitResult(Expected<Optional<VisitInterrupt>> result) {
  return AnyUnsafe::MoveAnyToTVMFFIAny(Any(std::move(result)));
}

Expected<Optional<VisitInterrupt>> DecodeStructuralVisitResult(TVMFFIAny result) {
  Any result_any = AnyUnsafe::MoveTVMFFIAnyToAny(&result);
  return std::move(result_any).cast<Expected<Optional<VisitInterrupt>>>();
}

}  // namespace

// Invoke a raw visit function and encode C++ exceptions as ffi.Error.
TVMFFIAny InvokeStructuralVisit(FStructuralVisit visit_fn, StructuralVisitorObj* visitor,
                                const ObjectRef& value) {
  try {
    TVM_FFI_ICHECK_NOTNULL(visit_fn);
    return (*visit_fn)(visitor, value);
  } catch (const Error& error) {
    return EncodeStructuralVisitResult(error);
  } catch (const std::exception& ex) {
    return EncodeStructuralVisitResult(Error("InternalError", ex.what(), ""));
  }
}

// Walk reflected fields of `value` and recurse into each non-ignored field.
Optional<VisitInterrupt> VisitReflectedFields(StructuralVisitorObj* visitor,
                                              const ObjectRef& value) {
  const TVMFFITypeInfo* type_info = TVMFFIGetTypeInfo(value.type_index());

  Optional<VisitInterrupt> result = std::nullopt;
  reflection::ForEachFieldInfoWithEarlyStop(type_info, [&](const TVMFFIFieldInfo* field_info) {
    if (field_info->flags & kTVMFFIFieldFlagBitMaskSEqHashIgnore) {
      return false;
    }

    reflection::FieldGetter getter(field_info);
    Any field_value = getter(value);

    TVMFFIDefRegionKind kind = kTVMFFIDefRegionKindNone;
    if (field_info->flags & kTVMFFIFieldFlagBitMaskSEqHashDefNonRecursive) {
      kind = kTVMFFIDefRegionKindNonRecursive;
    } else if (field_info->flags & kTVMFFIFieldFlagBitMaskSEqHashDefRecursive) {
      kind = kTVMFFIDefRegionKindRecursive;
    }

    result = visitor->WithDefRegionKind(
        kind, [&]() { return visitor->Visit(field_value.cast<ObjectRef>()); });
    return result.has_value();
  });
  return result;
}

// Dispatch function installed by the default StructuralVisitor constructor.
TVMFFIAny DispatchDefaultVisit(StructuralVisitorObj* visitor, const ObjectRef& value) {
  return EncodeStructuralVisitResult(visitor->DefaultVisit(value));
}

constexpr StructuralVisitorVTable kDefaultStructuralVisitorVTable = {&DispatchDefaultVisit};

}  // namespace details

StructuralVisitorObj::StructuralVisitorObj() {
  this->vtable = &details::kDefaultStructuralVisitorVTable;
  this->def_region_mode = kTVMFFIDefRegionKindNone;
}

Optional<VisitInterrupt> StructuralVisitorObj::Visit(const ObjectRef& value) {
  TVM_FFI_ICHECK_NOTNULL(vtable);
  TVM_FFI_ICHECK_NOTNULL(vtable->visit);
  return details::DecodeStructuralVisitResult(
             details::InvokeStructuralVisit(vtable->visit, this, value))
      .value();
}

Optional<VisitInterrupt> StructuralVisitorObj::DefaultVisit(const ObjectRef& value) {
  static reflection::TypeAttrColumn column(reflection::type_attr::kStructuralVisit);
  AnyView attr = column[value.type_index()];

  // Type-specific override registered as an opaque visit function pointer.
  if (attr.type_index() == TypeIndex::kTVMFFIOpaquePtr) {
    auto* fn = reinterpret_cast<FStructuralVisit>(attr.cast<void*>());
    return details::DecodeStructuralVisitResult(details::InvokeStructuralVisit(fn, this, value))
        .value();
  }

  // Type-specific override registered as an ffi::Function.
  if (attr.type_index() == TypeIndex::kTVMFFIFunction) {
    Function visit_child = Function::FromTyped(
        [this](const ObjectRef& child, int def_region_kind) -> Optional<VisitInterrupt> {
          return WithDefRegionKind(static_cast<TVMFFIDefRegionKind>(def_region_kind),
                                   [&]() { return Visit(child); });
        });
    return attr.cast<Function>()(value, visit_child).cast<Optional<VisitInterrupt>>();
  }

  if (attr.type_index() != TypeIndex::kTVMFFINone) {
    TVM_FFI_THROW(TypeError) << reflection::type_attr::kStructuralVisit
                             << " must be an opaque function pointer or ffi.Function";
  }

  return details::VisitReflectedFields(this, value);
}

StructuralVisitor::StructuralVisitor() : ObjectRef(make_object<StructuralVisitorObj>()) {}

Optional<VisitInterrupt> StructuralVisitor::Visit(const ObjectRef& value) {
  return get()->Visit(value);
}

Optional<VisitInterrupt> StructuralVisitor::DefaultVisit(const ObjectRef& value) {
  return get()->DefaultVisit(value);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  reflection::ObjectDef<StructuralVisitorObj>();
  reflection::EnsureTypeAttrColumn(reflection::type_attr::kStructuralVisit);
}

}  // namespace ffi
}  // namespace tvm
