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
 * \file src/ffi/extra/structural_map.cc
 * \brief Structural map implementation.
 */
#include <tvm/ffi/extra/structural_map.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/accessor.h>
#include <tvm/ffi/reflection/registry.h>

namespace tvm {
namespace ffi {

namespace details {

// Type-attr hooks store structural map callbacks as opaque C++ function pointers.
using FStructuralMap = decltype(StructuralMapperVTable::map);
using FStructuralInplaceMutate = decltype(StructuralMapperVTable::inplace_mutate);

TVMFFIDefRegionKind GetDefRegionKind(const TVMFFIFieldInfo* field_info) {
  if (field_info->flags & kTVMFFIFieldFlagBitMaskSEqHashDefNonRecursive) {
    return kTVMFFIDefRegionKindNonRecursive;
  }
  if (field_info->flags & kTVMFFIFieldFlagBitMaskSEqHashDefRecursive) {
    return kTVMFFIDefRegionKindRecursive;
  }
  return kTVMFFIDefRegionKindNone;
}

bool IsStructuralHook(AnyView attr, const char* attr_name) {
  if (attr.type_index() == TypeIndex::kTVMFFINone) return false;
  if (attr.type_index() == TypeIndex::kTVMFFIOpaquePtr ||
      attr.type_index() == TypeIndex::kTVMFFIFunction) {
    return true;
  }
  TVM_FFI_THROW(TypeError) << attr_name << " must be an opaque function pointer or ffi.Function";
}

StructuralMapper MakeMapperRef(StructuralMapperObj* mapper) {
  return StructuralMapper(ObjectUnsafe::ObjectPtrFromUnowned<StructuralMapperObj>(mapper));
}

ObjectRef ShallowCopy(const ObjectRef& value) {
  static reflection::TypeAttrColumn column(reflection::type_attr::kShallowCopy);
  AnyView attr = column[value.type_index()];
  if (attr.type_index() == TypeIndex::kTVMFFIFunction) {
    return attr.cast<Function>().CallExpected<ObjectRef>(value).value();
  }
  if (attr.type_index() != TypeIndex::kTVMFFINone) {
    TVM_FFI_THROW(TypeError) << reflection::type_attr::kShallowCopy << " must be an ffi.Function";
  }
  TVM_FFI_THROW(TypeError) << "Cannot structurally map type `" << value.GetTypeKey()
                           << "` after a child changed because it has no "
                           << reflection::type_attr::kShallowCopy << " hook";
}

bool TryMapWithTypeAttr(StructuralMapperObj* mapper, const ObjectRef& value, AnyView attr,
                        ObjectRef* result) {
  if (attr.type_index() == TypeIndex::kTVMFFIOpaquePtr) {
    auto* map_fn = reinterpret_cast<FStructuralMap>(attr.cast<void*>());
    if (TVM_FFI_PREDICT_FALSE(map_fn == nullptr)) {
      TVM_FFI_THROW(InternalError) << "Structural map function pointer is null";
    }
    *result = (*map_fn)(mapper, value);
    return true;
  }
  if (attr.type_index() == TypeIndex::kTVMFFIFunction) {
    *result = attr.cast<Function>().CallExpected<ObjectRef>(MakeMapperRef(mapper), value).value();
    return true;
  }
  return IsStructuralHook(attr, reflection::type_attr::kStructuralMap);
}

bool TryInplaceMutateWithTypeAttr(StructuralMapperObj* mapper, Object* value, AnyView attr) {
  if (attr.type_index() == TypeIndex::kTVMFFIOpaquePtr) {
    auto* mutate_fn = reinterpret_cast<FStructuralInplaceMutate>(attr.cast<void*>());
    if (TVM_FFI_PREDICT_FALSE(mutate_fn == nullptr)) {
      TVM_FFI_THROW(InternalError) << "Structural inplace mutator function pointer is null";
    }
    (*mutate_fn)(mapper, value);
    return true;
  }
  if (attr.type_index() == TypeIndex::kTVMFFIFunction) {
    ObjectRef value_ref(ObjectUnsafe::ObjectPtrFromUnowned<Object>(value));
    attr.cast<Function>()(MakeMapperRef(mapper), value_ref);
    return true;
  }
  return IsStructuralHook(attr, reflection::type_attr::kStructuralInplaceMutator);
}

ObjectRef MapReflectedFields(StructuralMapperObj* mapper, const ObjectRef& value) {
  const TVMFFITypeInfo* type_info = TVMFFIGetTypeInfo(value.type_index());
  ObjectRef result = value;
  Object* writer = nullptr;

  reflection::ForEachFieldInfo(type_info, [&](const TVMFFIFieldInfo* field_info) {
    if (field_info->flags & kTVMFFIFieldFlagBitMaskSEqHashIgnore) return;

    reflection::FieldGetter getter(field_info);
    Any field_value = getter(value);
    if (auto child = field_value.as<ObjectRef>()) {
      TVMFFIDefRegionKind kind = GetDefRegionKind(field_info);
      ObjectRef new_child =
          mapper->WithDefRegionKind(kind, [&]() { return mapper->Map(*child); });
      if (new_child.same_as(*child)) return;

      if (writer == nullptr) {
        result = ShallowCopy(value);
        writer = ObjectUnsafe::RawObjectPtrFromObjectRef(result);
      }
      reflection::FieldSetter setter(field_info);
      setter(writer, new_child);
    }
  });

  return result;
}

void InplaceMutateReflectedFields(StructuralMapperObj* mapper, Object* value) {
  const TVMFFITypeInfo* type_info = TVMFFIGetTypeInfo(value->type_index());

  reflection::ForEachFieldInfo(type_info, [&](const TVMFFIFieldInfo* field_info) {
    if (field_info->flags & kTVMFFIFieldFlagBitMaskSEqHashIgnore) return;

    reflection::FieldGetter getter(field_info);
    Any field_value = getter(value);
    if (auto child = field_value.as<ObjectRef>()) {
      ObjectRef new_child = *child;
      TVMFFIDefRegionKind kind = GetDefRegionKind(field_info);
      mapper->WithDefRegionKind(kind, [&]() { mapper->MapOrInplaceMutate(&new_child); });
      if (!new_child.same_as(*child)) {
        reflection::FieldSetter setter(field_info);
        setter(value, new_child);
      }
    }
  });
}

}  // namespace details

// ---------------------------------------------------------------------------
// StructuralMapper Object implementation.
// ---------------------------------------------------------------------------

ObjectRef StructuralMapperObj::DefaultMap(const ObjectRef& value) {
  static reflection::TypeAttrColumn column(reflection::type_attr::kStructuralMap);
  AnyView attr = column[value.type_index()];

  ObjectRef result;
  if (details::TryMapWithTypeAttr(this, value, attr, &result)) {
    return result;
  }

  return details::MapReflectedFields(this, value);
}

void StructuralMapperObj::DefaultInplaceMutate(Object* value) {
  static reflection::TypeAttrColumn column(reflection::type_attr::kStructuralInplaceMutator);
  AnyView attr = column[value->type_index()];

  if (details::TryInplaceMutateWithTypeAttr(this, value, attr)) {
    return;
  }

  details::InplaceMutateReflectedFields(this, value);
}

void StructuralMapperObj::DefaultMapOrInplaceMutate(ObjectRef* value) {
  TVM_FFI_ICHECK_NOTNULL(value);
  if (!value->defined()) return;

  static reflection::TypeAttrColumn map_column(reflection::type_attr::kStructuralMap);
  static reflection::TypeAttrColumn inplace_column(reflection::type_attr::kStructuralInplaceMutator);
  AnyView map_attr = map_column[value->type_index()];
  AnyView inplace_attr = inplace_column[value->type_index()];

  bool has_map = details::IsStructuralHook(map_attr, reflection::type_attr::kStructuralMap);
  bool has_inplace =
      details::IsStructuralHook(inplace_attr, reflection::type_attr::kStructuralInplaceMutator);

  if (has_map && !has_inplace) {
    *value = this->Map(*value);
    return;
  }

  if (value->unique()) {
    this->InplaceMutate(details::ObjectUnsafe::RawObjectPtrFromObjectRef(*value));
  } else {
    *value = this->Map(*value);
  }
}

// ---------------------------------------------------------------------------
// Static registration.
// ---------------------------------------------------------------------------

TVM_FFI_STATIC_INIT_BLOCK() {
  reflection::ObjectDef<StructuralMapperObj>();
  reflection::EnsureTypeAttrColumn(reflection::type_attr::kStructuralMap);
  reflection::EnsureTypeAttrColumn(reflection::type_attr::kStructuralInplaceMutator);
}

}  // namespace ffi
}  // namespace tvm