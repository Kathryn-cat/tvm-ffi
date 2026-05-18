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
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/dict.h>
#include <tvm/ffi/container/list.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/accessor.h>
#include <tvm/ffi/reflection/registry.h>

namespace tvm {
namespace ffi {

namespace details {

// ---------------------------------------------------------------------------
// StructuralVisitor helpers.
// ---------------------------------------------------------------------------

// Type-attr hooks store the same C++ visit callback as an opaque pointer.
using FStructuralVisit = decltype(StructuralVisitorVTable::visit);

/*!
 * \brief Try to visit \p value using a type-specific structural visit hook.
 *
 * \param visitor The active visitor.
 * \param value The object being visited.
 * \param attr The registered type attribute for \p value.
 * \param handled Set to true if \p attr represents a structural visit hook.
 * \return Expected interrupt state. An error means traversal failed.
 */
Expected<Optional<VisitInterrupt>> TryVisitWithTypeAttr(StructuralVisitorObj* visitor,
                                                        const ObjectRef& value, AnyView attr,
                                                        bool* handled) {
  // case 1: Type-specific override registered as an opaque visit function pointer.
  if (attr.type_index() == TypeIndex::kTVMFFIOpaquePtr) {
    *handled = true;
    auto* visit_fn = reinterpret_cast<FStructuralVisit>(attr.cast<void*>());
    if (TVM_FFI_PREDICT_FALSE(visit_fn == nullptr)) {
      return Unexpected(Error("InternalError", "Structural visit function pointer is null", ""));
    }
    return (*visit_fn)(visitor, value);
  }

  // case 2: Type-specific override registered as an ffi::Function.
  if (attr.type_index() == TypeIndex::kTVMFFIFunction) {
    *handled = true;
    StructuralVisitor visitor_ref(ObjectUnsafe::ObjectPtrFromUnowned<StructuralVisitorObj>(visitor));
    return attr.cast<Function>().CallExpected<Optional<VisitInterrupt>>(visitor_ref, value);
  }

  if (TVM_FFI_PREDICT_FALSE(attr.type_index() != TypeIndex::kTVMFFINone)) {
    return Unexpected(Error("TypeError",
                            std::string(reflection::type_attr::kStructuralVisit) +
                                " must be an opaque function pointer or ffi.Function",
                            ""));
  }
  *handled = false;
  return Optional<VisitInterrupt>(std::nullopt);
}

/*!
 * \brief Walk reflected structural fields of \p value.
 *
 * Fields marked with ``kTVMFFIFieldFlagBitMaskSEqHashIgnore`` are skipped.
 * Def-region field flags are scoped around recursive child visits.
 *
 * \param visitor The active visitor.
 * \param value The object whose reflected fields should be visited.
 * \return Expected interrupt state. An error means traversal failed.
 */
Expected<Optional<VisitInterrupt>> VisitReflectedFields(StructuralVisitorObj* visitor,
                                                        const ObjectRef& value) {
  const TVMFFITypeInfo* type_info = TVMFFIGetTypeInfo(value.type_index());

  Expected<Optional<VisitInterrupt>> result = Optional<VisitInterrupt>(std::nullopt);
  reflection::ForEachFieldInfoWithEarlyStop(type_info, [&](const TVMFFIFieldInfo* field_info) -> bool {
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

    if (auto child = field_value.as<ObjectRef>()) {
      result = visitor->WithDefRegionKindExpected(
          kind, [&]() { return visitor->VisitExpected(*child); });
    }
    return TVM_FFI_PREDICT_FALSE(result.is_err() || result.value().has_value());
  });
  return result;
}

}  // namespace details

// ---------------------------------------------------------------------------
// StructuralVisitor Object implementation.
// ---------------------------------------------------------------------------

/*!
 * \brief Default structural visit implementation.
 *
 * The default path first checks for a type-specific structural visit hook, then
 * falls back to reflected structural fields.
 */
Expected<Optional<VisitInterrupt>> StructuralVisitorObj::DefaultVisitExpected(
    const ObjectRef& value) noexcept {
  try {
    static reflection::TypeAttrColumn column(reflection::type_attr::kStructuralVisit);
    AnyView attr = column[value.type_index()];

    bool handled = false;
    Expected<Optional<VisitInterrupt>> interrupt =
        details::TryVisitWithTypeAttr(this, value, attr, &handled);
    if (TVM_FFI_PREDICT_FALSE(interrupt.is_err())) return interrupt;
    if (handled) return interrupt;

    return details::VisitReflectedFields(this, value);
  } catch (const Error& err) {
    return Unexpected(err);
  } catch (const std::exception& err) {
    return Unexpected(Error("InternalError", err.what(), ""));
  } catch (...) {
    return Unexpected(Error("InternalError", "Unknown structural visit error", ""));
  }
}

// ---------------------------------------------------------------------------
// Built-in container structural visit.
// ---------------------------------------------------------------------------

namespace details {

/*! \brief Visit ObjectRef entries in a sequence container. */
Expected<Optional<VisitInterrupt>> VisitSeqContainer(StructuralVisitorObj* visitor,
                                                     const SeqBaseObj* seq) {
  for (const Any& item : *seq) {
    if (auto child = item.as<ObjectRef>()) {
      Expected<Optional<VisitInterrupt>> interrupt = visitor->VisitExpected(*child);
      if (TVM_FFI_PREDICT_FALSE(interrupt.is_err() || interrupt.value().has_value())) {
        return interrupt;
      }
    }
  }
  return Optional<VisitInterrupt>(std::nullopt);
}

/*! \brief Visit ObjectRef keys and values in a map container. */
Expected<Optional<VisitInterrupt>> VisitMapContainer(StructuralVisitorObj* visitor,
                                                     const MapBaseObj* map) {
  for (const auto& kv : *map) {
    if (auto key = kv.first.as<ObjectRef>()) {
      Expected<Optional<VisitInterrupt>> interrupt = visitor->VisitExpected(*key);
      if (TVM_FFI_PREDICT_FALSE(interrupt.is_err() || interrupt.value().has_value())) {
        return interrupt;
      }
    }
    if (auto val = kv.second.as<ObjectRef>()) {
      Expected<Optional<VisitInterrupt>> interrupt = visitor->VisitExpected(*val);
      if (TVM_FFI_PREDICT_FALSE(interrupt.is_err() || interrupt.value().has_value())) {
        return interrupt;
      }
    }
  }
  return Optional<VisitInterrupt>(std::nullopt);
}

}  // namespace details

/*! \brief Structural visit hook for ArrayObj. */
Expected<Optional<VisitInterrupt>> VisitArray(StructuralVisitorObj* visitor,
                                              const ObjectRef& value) {
  const auto* array = static_cast<const ArrayObj*>(value.get());
  return details::VisitSeqContainer(visitor, array);
}

/*! \brief Structural visit hook for ListObj. */
Expected<Optional<VisitInterrupt>> VisitList(StructuralVisitorObj* visitor,
                                             const ObjectRef& value) {
  const auto* list = static_cast<const ListObj*>(value.get());
  return details::VisitSeqContainer(visitor, list);
}

/*! \brief Structural visit hook for MapObj. */
Expected<Optional<VisitInterrupt>> VisitMap(StructuralVisitorObj* visitor,
                                            const ObjectRef& value) {
  const auto* map = static_cast<const MapObj*>(value.get());
  return details::VisitMapContainer(visitor, map);
}

/*! \brief Structural visit hook for DictObj. */
Expected<Optional<VisitInterrupt>> VisitDict(StructuralVisitorObj* visitor,
                                             const ObjectRef& value) {
  const auto* dict = static_cast<const DictObj*>(value.get());
  return details::VisitMapContainer(visitor, dict);
}

// ---------------------------------------------------------------------------
// Static registration.
// ---------------------------------------------------------------------------

TVM_FFI_STATIC_INIT_BLOCK() {
  reflection::ObjectDef<StructuralVisitorObj>();
  reflection::EnsureTypeAttrColumn(reflection::type_attr::kStructuralVisit);
  reflection::TypeAttrDef<ArrayObj>().attr(
      reflection::type_attr::kStructuralVisit, reinterpret_cast<void*>(&VisitArray));
  reflection::TypeAttrDef<ListObj>().attr(
      reflection::type_attr::kStructuralVisit, reinterpret_cast<void*>(&VisitList));
  reflection::TypeAttrDef<MapObj>().attr(
      reflection::type_attr::kStructuralVisit, reinterpret_cast<void*>(&VisitMap));
  reflection::TypeAttrDef<DictObj>().attr(
      reflection::type_attr::kStructuralVisit, reinterpret_cast<void*>(&VisitDict));
}

}  // namespace ffi
}  // namespace tvm
