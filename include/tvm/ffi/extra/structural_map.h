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
 * \file tvm/ffi/extra/structural_map.h
 * \brief Structural mapping and rewriting utilities.
 */
#ifndef TVM_FFI_EXTRA_STRUCTURAL_MAP_H_
#define TVM_FFI_EXTRA_STRUCTURAL_MAP_H_

#include <tvm/ffi/any.h>
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/object.h>

#include <type_traits>
#include <utility>

namespace tvm {
namespace ffi {

class StructuralMapperObj;

/*!
 * \brief C++ vtable for structural mapper dispatch.
 *
 * The vtable allows custom mapper subclasses to override functional mapping,
 * in-place mutation, and the copy-on-write dispatcher independently.
 */
struct StructuralMapperVTable {
  /*! \brief Functional map callback. */
  ObjectRef (*map)(StructuralMapperObj* mapper, const ObjectRef& value) = nullptr;
  /*! \brief In-place mutation callback. */
  void (*inplace_mutate)(StructuralMapperObj* mapper, Object* value) = nullptr;
  /*! \brief Copy-on-write dispatcher callback. */
  void (*map_or_inplace_mutate)(StructuralMapperObj* mapper, ObjectRef* value) = nullptr;
};

/*!
 * \brief Object node of a structural mapper.
 *
 * A structural mapper rewrites an object graph while preserving sharing
 * semantics. \ref Map is functional and never mutates the input object.
 * \ref InplaceMutate mutates an object known to be safe to update, while
 * \ref MapOrInplaceMutate dispatches between the two based on uniqueness.
 */
class StructuralMapperObj : public Object {
 public:
  /*! \brief Construct the default structural mapper. */
  StructuralMapperObj() {
    static const StructuralMapperVTable vtable{
        [](StructuralMapperObj* mapper, const ObjectRef& value) {
          return mapper->DefaultMap(value);
        },
        [](StructuralMapperObj* mapper, Object* value) { mapper->DefaultInplaceMutate(value); },
        [](StructuralMapperObj* mapper, ObjectRef* value) {
          mapper->DefaultMapOrInplaceMutate(value);
        }};
    this->vtable_ = &vtable;
    this->def_region_mode_ = kTVMFFIDefRegionKindNone;
  }

  /*!
   * \brief Construct a structural mapper with a custom dispatch vtable.
   * \param vtable The dispatch table for this mapper.
   */
  explicit StructuralMapperObj(const StructuralMapperVTable* vtable) : vtable_(vtable) {
    TVM_FFI_ICHECK_NOTNULL(vtable);
    TVM_FFI_ICHECK_NOTNULL(vtable->map);
    TVM_FFI_ICHECK_NOTNULL(vtable->inplace_mutate);
    TVM_FFI_ICHECK_NOTNULL(vtable->map_or_inplace_mutate);
  }

  /*!
   * \brief Functionally map an object.
   * \param value The object to map.
   * \return The mapped object, or \p value if unchanged.
   */
  TVM_FFI_INLINE ObjectRef Map(const ObjectRef& value) {
    if (TVM_FFI_PREDICT_FALSE(vtable_ == nullptr)) {
      TVM_FFI_THROW(InternalError) << "StructuralMapper vtable is null";
    }
    if (TVM_FFI_PREDICT_FALSE(vtable_->map == nullptr)) {
      TVM_FFI_THROW(InternalError) << "StructuralMapper map callback is null";
    }
    return (*vtable_->map)(this, value);
  }

  /*!
   * \brief Mutate an object in place.
   * \param value The object to mutate. Must be safe to mutate in place.
   */
  TVM_FFI_INLINE void InplaceMutate(Object* value) {
    TVM_FFI_ICHECK_NOTNULL(value);
    if (TVM_FFI_PREDICT_FALSE(vtable_ == nullptr)) {
      TVM_FFI_THROW(InternalError) << "StructuralMapper vtable is null";
    }
    if (TVM_FFI_PREDICT_FALSE(vtable_->inplace_mutate == nullptr)) {
      TVM_FFI_THROW(InternalError) << "StructuralMapper inplace mutate callback is null";
    }
    (*vtable_->inplace_mutate)(this, value);
  }

  /*!
   * \brief Mutate a field in place if unique, or replace it with \ref Map result.
   * \param value The field to rewrite.
   */
  TVM_FFI_INLINE void MapOrInplaceMutate(ObjectRef* value) {
    TVM_FFI_ICHECK_NOTNULL(value);
    if (!value->defined()) return;
    if (TVM_FFI_PREDICT_FALSE(vtable_ == nullptr)) {
      TVM_FFI_THROW(InternalError) << "StructuralMapper vtable is null";
    }
    if (TVM_FFI_PREDICT_FALSE(vtable_->map_or_inplace_mutate == nullptr)) {
      TVM_FFI_THROW(InternalError) << "StructuralMapper map-or-inplace callback is null";
    }
    (*vtable_->map_or_inplace_mutate)(this, value);
  }

  /*!
   * \brief Default functional map behavior.
   *
   * The default path first checks the structural-map type attribute for a
   * type-specific override. If none is registered, it maps reflected
   * structural fields and lazily shallow-copies the parent on the first change.
   */
  TVM_FFI_DLL ObjectRef DefaultMap(const ObjectRef& value);

  /*!
   * \brief Default in-place mutation behavior.
   *
   * The default path first checks the structural-inplace-mutator type
   * attribute for a type-specific override. If none is registered, it mutates
   * reflected structural fields through \ref MapOrInplaceMutate.
   */
  TVM_FFI_DLL void DefaultInplaceMutate(Object* value);

  /*!
   * \brief Default copy-on-write dispatcher.
   *
   * If a custom map hook exists without a matching custom in-place hook, this
   * dispatcher calls \ref Map to preserve custom functional semantics.
   */
  TVM_FFI_DLL void DefaultMapOrInplaceMutate(ObjectRef* value);

  /*!
   * \brief Temporarily switch the def-region context while invoking \p callback.
   *
   * \param kind The def-region kind to set during the callback.
   * \param callback A nullary callable that performs recursive mapping.
   * \return The callback result.
   */
  template <typename Callback>
  TVM_FFI_INLINE decltype(auto) WithDefRegionKind(TVMFFIDefRegionKind kind,
                                                  Callback&& callback) {
    class Scope {
     public:
      Scope(StructuralMapperObj* mapper, TVMFFIDefRegionKind kind)
          : mapper_(mapper), old_kind_(mapper->def_region_mode_) {
        mapper_->def_region_mode_ = kind;
      }
      ~Scope() { mapper_->def_region_mode_ = old_kind_; }
      Scope(const Scope&) = delete;
      Scope& operator=(const Scope&) = delete;

     private:
      StructuralMapperObj* mapper_;
      TVMFFIDefRegionKind old_kind_;
    };
    Scope scope(this, kind);
    if constexpr (std::is_void_v<std::invoke_result_t<Callback>>) {
      std::forward<Callback>(callback)();
    } else {
      return std::forward<Callback>(callback)();
    }
  }

  /// \cond Doxygen_Suppress
  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("ffi.StructuralMapper", StructuralMapperObj, Object);
  /// \endcond

 protected:
  /*! \brief Required dispatch table. */
  const StructuralMapperVTable* vtable_ = nullptr;

  /*! \brief Current def-region context for structural map semantics. */
  TVMFFIDefRegionKind def_region_mode_ = kTVMFFIDefRegionKindNone;
};

/*!
 * \brief ObjectRef wrapper of \ref StructuralMapperObj.
 *
 * \sa StructuralMapperObj
 */
class StructuralMapper : public ObjectRef {
 public:
  /*! \brief Construct the default structural mapper. */
  StructuralMapper() : ObjectRef(make_object<StructuralMapperObj>()) {}

  /*!
   * \brief Construct from an existing object pointer.
   * \param n The object pointer to wrap.
   */
  explicit StructuralMapper(ObjectPtr<StructuralMapperObj> n) : ObjectRef(std::move(n)) {}

  /*!
   * \brief Functionally map an object.
   * \param value The object to map.
   * \return The mapped object, or \p value if unchanged.
   */
  TVM_FFI_INLINE ObjectRef Map(const ObjectRef& value) { return get()->Map(value); }

  /*!
   * \brief Mutate an object in place.
   * \param value The object to mutate. Must be safe to mutate in place.
   */
  TVM_FFI_INLINE void InplaceMutate(Object* value) { get()->InplaceMutate(value); }

  /*!
   * \brief Mutate a field in place if unique, or replace it with \ref Map result.
   * \param value The field to rewrite.
   */
  TVM_FFI_INLINE void MapOrInplaceMutate(ObjectRef* value) {
    get()->MapOrInplaceMutate(value);
  }

  /*!
   * \brief Invoke the default functional map behavior.
   * \param value The object to map.
   * \return The mapped object, or \p value if unchanged.
   */
  TVM_FFI_INLINE ObjectRef DefaultMap(const ObjectRef& value) {
    return get()->DefaultMap(value);
  }

  /*!
   * \brief Invoke the default in-place mutation behavior.
   * \param value The object to mutate. Must be safe to mutate in place.
   */
  TVM_FFI_INLINE void DefaultInplaceMutate(Object* value) {
    get()->DefaultInplaceMutate(value);
  }

  /*!
   * \brief Invoke the default copy-on-write dispatcher.
   * \param value The field to rewrite.
   */
  TVM_FFI_INLINE void DefaultMapOrInplaceMutate(ObjectRef* value) {
    get()->DefaultMapOrInplaceMutate(value);
  }

  /*!
   * \brief Temporarily switch the def-region context while invoking \p callback.
   * \param kind The def-region kind to set during the callback.
   * \param callback A nullary callable that performs recursive mapping.
   * \return The callback result.
   */
  template <typename Callback>
  TVM_FFI_INLINE decltype(auto) WithDefRegionKind(TVMFFIDefRegionKind kind,
                                                  Callback&& callback) {
    return get()->WithDefRegionKind(kind, std::forward<Callback>(callback));
  }

  /// \cond Doxygen_Suppress
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(StructuralMapper, ObjectRef, StructuralMapperObj);
  /// \endcond
};


}  // namespace ffi
}  // namespace tvm

#endif  // TVM_FFI_EXTRA_STRUCTURAL_MAP_H_
