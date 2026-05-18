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
 * \file tvm/ffi/extra/structural_visit.h
 * \brief Structural visit implementation
 */
#ifndef TVM_FFI_EXTRA_STRUCTURAL_VISIT_H_
#define TVM_FFI_EXTRA_STRUCTURAL_VISIT_H_

#include <tvm/ffi/any.h>
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/container/variant.h>
#include <tvm/ffi/expected.h>
#include <tvm/ffi/optional.h>

#include <cstddef>
#include <exception>
#include <optional>
#include <type_traits>
#include <utility>

namespace tvm {
namespace ffi {

/*!
 * \brief Object node carrying the optional payload for an interrupted structural visit.
 */
class VisitInterruptObj : public Object {
 public:
  /*! \brief Payload returned with the interrupt, or FFI None for no payload. */
  Any value;

  VisitInterruptObj() = default;
  /*!
   * \brief Construct a VisitInterruptObj with a payload.
   * \param value The payload carried by the interrupt.
   */
  explicit VisitInterruptObj(Any value) : value(std::move(value)) {}

  /// \cond Doxygen_Suppress
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ffi.VisitInterrupt", VisitInterruptObj, Object);
  /// \endcond
};

/*!
 * \brief ObjectRef wrapper for VisitInterruptObj.
 */
class VisitInterrupt : public ObjectRef {
 public:
  /*! \brief Construct an interrupt with no payload. */
  VisitInterrupt() : VisitInterrupt(Any(nullptr)) {}
  /*!
   * \brief Construct an interrupt with a user-defined payload.
   * \param value The payload carried by the interrupt.
   */
  explicit VisitInterrupt(Any value)
      : ObjectRef(make_object<VisitInterruptObj>(std::move(value))) {}

  /// \cond Doxygen_Suppress
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(VisitInterrupt, ObjectRef, VisitInterruptObj);
  /// \endcond
};

class StructuralVisitorObj;

/*!
 * \brief C++ vtable for structural visitor dispatch.
 *
 * The vtable gives \ref StructuralVisitorObj an explicit dispatch table instead
 * of relying on C++ virtual methods.
 */
struct StructuralVisitorVTable {
  /*! \brief Visit callback. */
  Expected<Optional<VisitInterrupt>> (*visit)(StructuralVisitorObj* visitor,
                                              const ObjectRef& value) = nullptr;
};

/*!
 * \brief Object node of a structural visitor.
 *
 * A structural visitor is an active traversal context.  It carries the dispatch
 * table used to visit each object and the current def-region state used by
 * structural equality/hash semantics.  The visitor is ref-counted so it can
 * cross FFI boundaries, but one underlying visitor object should not be shared
 * by overlapping top-level traversals.
 */
class StructuralVisitorObj : public Object {
 public:
  /*! \brief Construct the default structural visitor. */
  StructuralVisitorObj() {
    static const StructuralVisitorVTable vtable{
        [](StructuralVisitorObj* visitor, const ObjectRef& value) {
          return visitor->DefaultVisitExpected(value);
        }};
    this->vtable_ = &vtable;
    this->def_region_mode_ = kTVMFFIDefRegionKindNone;
  }

  /*!
   * \brief Construct a structural visitor with a custom dispatch vtable.
   * \param vtable The dispatch table for this visitor.
   */
  explicit StructuralVisitorObj(const StructuralVisitorVTable* vtable) : vtable_(vtable) {
    TVM_FFI_ICHECK_NOTNULL(vtable);
    TVM_FFI_ICHECK_NOTNULL(vtable->visit);
  }

  /*!
   * \brief Visit a value, dispatching through this visitor's vtable.
   *
   * \param value The object to visit.
   * \return ``std::nullopt`` to continue traversal, or a \ref VisitInterrupt
   *         to halt the entire visit.
   */
  TVM_FFI_INLINE Optional<VisitInterrupt> Visit(const ObjectRef& value) {
    return VisitExpected(value).value();
  }

  /*!
   * \brief Visit a value, propagating error through expected return.
   *
   * \param value The object to visit.
   * \return Expected interrupt state. An error means traversal failed.
   */
  TVM_FFI_INLINE Expected<Optional<VisitInterrupt>> VisitExpected(
      const ObjectRef& value) noexcept {
    try {
      if (TVM_FFI_PREDICT_FALSE(vtable_ == nullptr)) {
        return Unexpected(Error("InternalError", "StructuralVisitor vtable is null", ""));
      }
      if (TVM_FFI_PREDICT_FALSE(vtable_->visit == nullptr)) {
        return Unexpected(Error("InternalError", "StructuralVisitor visit callback is null", ""));
      }
      return (*vtable_->visit)(this, value);
    } catch (const Error& err) {
      return Unexpected(err);
    } catch (const std::exception& err) {
      return Unexpected(Error("InternalError", err.what(), ""));
    } catch (...) {
      return Unexpected(Error("InternalError", "Unknown structural visit error", ""));
    }
  }

  /*!
   * \brief Default visit behavior: type-attr lookup, then reflection fallback.
   *
   * The default path first checks the structural-visit type attribute for a
   * type-specific override.  If none is registered, it visits all reflected
   * fields that participate in structural equality/hash.
   */
  TVM_FFI_INLINE Optional<VisitInterrupt> DefaultVisit(const ObjectRef& value) {
    return DefaultVisitExpected(value).value();
  }

  /*!
   * \brief Default visit behavior, propagating error through expected return.
   *
   * \param value The object to visit.
   * \return Expected interrupt state. An error means traversal failed.
   */
  TVM_FFI_DLL Expected<Optional<VisitInterrupt>> DefaultVisitExpected(
      const ObjectRef& value) noexcept;

  /*!
   * \brief Temporarily switch the def-region context while invoking \p callback.
   *
   * This helper scopes updates to the traversal state used by def/use-region
   * aware visitors. The previous state is restored when the callback returns
   * or throws.
   *
   * \param kind The def-region kind to set during the callback.
   * \param callback A nullary callable that performs recursive visiting.
   * \return ``std::nullopt`` to continue traversal, or a \ref VisitInterrupt
   *         to halt the entire visit.
   * 
   * \note Return type of callback should be Expected<Optional<VisitInterrupt>>.
   */
  template <typename Callback>
  TVM_FFI_INLINE Optional<VisitInterrupt> WithDefRegionKind(TVMFFIDefRegionKind kind,
                                                            Callback&& callback) {
    return WithDefRegionKindExpected(kind, std::forward<Callback>(callback)).value();
  }

  /*!
   * \brief Temporarily switch the def-region context while invoking \p callback.
   *
   * \param kind The def-region kind to set during the callback.
   * \param callback A nullary callable that performs recursive visiting.
   * \return Expected interrupt state. An error means traversal failed.
   *
   * \note Return type of callback should be Expected<Optional<VisitInterrupt>>.
   */
  template <typename Callback>
  TVM_FFI_INLINE Expected<Optional<VisitInterrupt>> WithDefRegionKindExpected(
      TVMFFIDefRegionKind kind, Callback&& callback) noexcept {
    class Scope {
     public:
      Scope(StructuralVisitorObj* visitor, TVMFFIDefRegionKind kind)
          : visitor_(visitor), old_kind_(visitor->def_region_mode_) {
        visitor_->def_region_mode_ = kind;
      }
      ~Scope() { visitor_->def_region_mode_ = old_kind_; }
      Scope(const Scope&) = delete;
      Scope& operator=(const Scope&) = delete;

     private:
      StructuralVisitorObj* visitor_;
      TVMFFIDefRegionKind old_kind_;
    };
    try {
      Scope scope(this, kind);
      return std::forward<Callback>(callback)();
    } catch (const Error& err) {
      return Unexpected(err);
    } catch (const std::exception& err) {
      return Unexpected(Error("InternalError", err.what(), ""));
    } catch (...) {
      return Unexpected(Error("InternalError", "Unknown structural visit error", ""));
    }
  }

  /// \cond Doxygen_Suppress
  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("ffi.StructuralVisitor", StructuralVisitorObj, Object);
  /// \endcond

 protected:
  /*!
   * \brief Required ABI dispatch table. \ref StructuralVisitorVTable
   * It must never be null on a constructed visitor.
   */
  const StructuralVisitorVTable* vtable_ = nullptr;

  /*!
   * \brief Current def-region context for structural equality/hash semantics.
   *
   * This is shared mutable traversal state. Be careful when mutating it through
   * multiple references to the same visitor object. Use \ref WithDefRegionKind
   * to scope temporary changes.
   */
  TVMFFIDefRegionKind def_region_mode_ = kTVMFFIDefRegionKindNone;
};

/*!
 * \brief ObjectRef wrapper of \ref StructuralVisitorObj.
 *
 * \sa StructuralVisitorObj
 */
class StructuralVisitor : public ObjectRef {
 public:
  /*!
   * \brief Construct the default structural visitor.
   */
  StructuralVisitor() : ObjectRef(make_object<StructuralVisitorObj>()) {}
  /*!
   * \brief Construct from an existing object pointer.
   * \param n The object pointer to wrap.
   */
  explicit StructuralVisitor(ObjectPtr<StructuralVisitorObj> n) : ObjectRef(std::move(n)) {}

  /*!
   * \brief Visit a value, dispatching through this visitor's vtable.
   * \param value The object to visit.
   * \return ``std::nullopt`` to continue traversal, or a \ref VisitInterrupt
   *         to halt the entire visit.
   */
  TVM_FFI_INLINE Optional<VisitInterrupt> Visit(const ObjectRef& value) {
    return get()->Visit(value);
  }

  /*!
   * \brief Visit a value, propagating error through expected return.
   * \param value The object to visit.
   * \return Expected interrupt state. An error means traversal failed.
   */
  TVM_FFI_INLINE Expected<Optional<VisitInterrupt>> VisitExpected(
      const ObjectRef& value) noexcept {
    return get()->VisitExpected(value);
  }

  /*!
   * \brief Invoke the default visit behavior on \p value.
   *
   * This bypasses custom visitor dispatch and applies type-attr lookup followed
   * by reflection fallback.
   */
  TVM_FFI_INLINE Optional<VisitInterrupt> DefaultVisit(const ObjectRef& value) {
    return get()->DefaultVisit(value);
  }

  /*!
   * \brief Invoke the default visit behavior on \p value, propagating error through expected return.
   * \param value The object to visit.
   * \return Expected interrupt state. An error means traversal failed.
   */
  TVM_FFI_INLINE Expected<Optional<VisitInterrupt>> DefaultVisitExpected(
      const ObjectRef& value) noexcept {
    return get()->DefaultVisitExpected(value);
  }

  /*!
   * \brief Temporarily switch the def-region context while invoking \p callback.
   *
   * This helper scopes updates to the traversal state used by def/use-region
   * aware visitors. The previous state is restored when the callback returns
   * or throws.
   *
   * \param kind The def-region kind to set during the callback.
   * \param callback A nullary callable that performs recursive visiting.
   * \return ``std::nullopt`` to continue traversal, or a \ref VisitInterrupt
   *         to halt the entire visit.
   *
   * \note Return type of callback should be Expected<Optional<VisitInterrupt>>.
   */
  template <typename Callback>
  TVM_FFI_INLINE Optional<VisitInterrupt> WithDefRegionKind(TVMFFIDefRegionKind kind,
                                                            Callback&& callback) {
    return get()->WithDefRegionKind(kind, std::forward<Callback>(callback));
  }

  /*!
   * \brief Temporarily switch the def-region context while invoking \p callback.
   *
   * \param kind The def-region kind to set during the callback.
   * \param callback A nullary callable that performs recursive visiting.
   * \return Expected interrupt state. An error means traversal failed.
   *
   * \note Return type of callback should be Expected<Optional<VisitInterrupt>>.
   */
  template <typename Callback>
  TVM_FFI_INLINE Expected<Optional<VisitInterrupt>> WithDefRegionKindExpected(
      TVMFFIDefRegionKind kind, Callback&& callback) noexcept {
    return get()->WithDefRegionKindExpected(kind, std::forward<Callback>(callback));
  }

  /// \cond Doxygen_Suppress
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(StructuralVisitor, ObjectRef, StructuralVisitorObj);
  /// \endcond
};

// ---------------------------------------------------------------------------
// Structural Walk API.
// ---------------------------------------------------------------------------

/*!
 * \brief Callback order for \ref structuralWalk.
 */
enum class WalkOrder : int32_t {
  /*! \brief Invoke the callback before visiting children. */
  kPreOrder = 0,
  /*! \brief Invoke the callback after visiting children. */
  kPostOrder = 1,
};

namespace details {

/*!
 * \brief Concrete visitor implementation used by ``structuralWalk``.
 */
template <typename Callback, typename... T>
class StructuralWalkVisitorObj : public StructuralVisitorObj {
 public:
  /*!
   * \brief Construct a structural walk visitor.
   * \param callback Callback to invoke on matching nodes.
   * \param order Callback placement relative to child traversal.
   */
  template <typename F>
  StructuralWalkVisitorObj(F&& callback, WalkOrder order)
      : StructuralVisitorObj(VTable()), callback_(std::forward<F>(callback)), order_(order) {}

 private:
  /*!
   * \brief Return the vtable used by this visitor implementation.
   * \return Pointer to the static structural visitor vtable.
   */
  static const StructuralVisitorVTable* VTable() {
    static const StructuralVisitorVTable vtable{&StructuralWalkVisitorObj::DispatchVisit};
    return &vtable;
  }

  /*!
   * \brief Dispatch from the erased visitor pointer to the concrete walk visitor.
   * \param self The erased structural visitor object.
   * \param value The object to visit.
   * \return Interrupt state, or an error if traversal failed.
   */
  static Expected<Optional<VisitInterrupt>> DispatchVisit(StructuralVisitorObj* self,
                                                          const ObjectRef& value) {
    return static_cast<StructuralWalkVisitorObj*>(self)->VisitImpl(value);
  }

  /*!
   * \brief Visit one object according to the configured walk order.
   * \param value The object to visit.
   * \return Interrupt state, or an error if traversal failed.
   */
  Expected<Optional<VisitInterrupt>> VisitImpl(const ObjectRef& value) {
    if (order_ == WalkOrder::kPreOrder) {
      auto result = InvokeCallback<T...>(value);
      if (TVM_FFI_PREDICT_FALSE(result.is_err())) return Unexpected(result.error());
      auto action = result.value();
      auto interrupt = action.template as<VisitInterrupt>();
      if (TVM_FFI_PREDICT_FALSE(interrupt.has_value())) {
        return Optional<VisitInterrupt>(*interrupt);
      }
      if (TVM_FFI_PREDICT_FALSE(!action.template get<bool>())) {
        return Optional<VisitInterrupt>(std::nullopt);
      }
    }

    Expected<Optional<VisitInterrupt>> interrupt = DefaultVisitExpected(value);
    if (TVM_FFI_PREDICT_FALSE(interrupt.is_err() || interrupt.value().has_value())) {
      return interrupt;
    }

    if (order_ == WalkOrder::kPostOrder) {
      auto result = InvokeCallback<T...>(value);
      if (TVM_FFI_PREDICT_FALSE(result.is_err())) return Unexpected(result.error());
      auto action = result.value();
      auto callback_interrupt = action.template as<VisitInterrupt>();
      if (TVM_FFI_PREDICT_FALSE(callback_interrupt.has_value())) {
        return Optional<VisitInterrupt>(*callback_interrupt);
      }
    }

    return Optional<VisitInterrupt>(std::nullopt);
  }

  /*!
   * \brief Invoke the callback for the first matching candidate type.
   * \param value The object to match against ``First`` and ``Rest...``.
   * \return Callback action, or an error if the callback failed.
   */
  template <typename First, typename... Rest>
  Expected<Variant<VisitInterrupt, bool>> InvokeCallback(const ObjectRef& value) {
    if constexpr (std::is_base_of_v<ObjectRef, First>) {
      if (auto node = value.template as<First>()) {
        return callback_(*node);
      }
    } else {
      if (const First* node = value.template as<First>()) {
        return callback_(node);
      }
    }
    if constexpr (sizeof...(Rest) == 0) {
      return Variant<VisitInterrupt, bool>(true);
    } else {
      return InvokeCallback<Rest...>(value);
    }
  }

  /*! \brief Concrete callback object. */
  Callback callback_;
  /*! \brief Callback placement relative to child traversal. */
  WalkOrder order_;
};

}  // namespace details

/*!
 * \brief Walk an object graph structurally and invoke a callback on selected node types.
 *
 * The callback is invoked only for nodes matching one of the template types
 * ``T...``.  Types are tested in order, and the first match is used.  Each
 * ``T`` may be either an ``ObjectRef`` type or an ``Object`` node type.
 *
 * The callback should return ``Expected<Variant<VisitInterrupt, bool>>``:
 * - ``VisitInterrupt`` halts traversal, analogous to interrupt in MLIR.
 * - ``true`` continues traversal, analogous to advance in MLIR.
 * - ``false`` skips children traversal, analogous to skip in MLIR.
 * - ``Error`` indicates traversal failure.
 * See \ref WalkOrder for more details.
 *
 * Example:
 *
 * \code
 * int num_adds = 0;
 *
 * Expected<Optional<VisitInterrupt>> result = structuralWalkExpected<Add>(
 *     root,
 *     [&](const Add& add) -> Expected<Variant<VisitInterrupt, bool>> {
 *       ++num_adds;
 *       return Variant<VisitInterrupt, bool>(true);
 *     },
 *     WalkOrder::kPreOrder);
 * \endcode
 *
 * \tparam T Node types that should trigger the callback.
 * \tparam F Callback type.
 * \param root The root object to visit.
 * \param callback Callback invoked for matching nodes.
 * \param order Whether to invoke the callback before or after visiting children.
 * \return ``std::nullopt`` if traversal completed, or the interrupt returned by
 *         the callback.
 *
 * \note Return type of callback should be Expected<Variant<VisitInterrupt, bool>>.
 */
template <typename... T, typename F>
Expected<Optional<VisitInterrupt>> structuralWalkExpected(const ObjectRef& root, F&& callback,
                                                          WalkOrder order) noexcept {
  try {
    static_assert(sizeof...(T) != 0, "structuralWalk requires at least one matched type");
    static_assert(((std::is_base_of_v<ObjectRef, T> || std::is_base_of_v<Object, T>) && ...),
                  "structuralWalk matched types must derive from ObjectRef or Object");

    using Callback = std::decay_t<F>;
    StructuralVisitor visitor(make_object<details::StructuralWalkVisitorObj<Callback, T...>>(
        std::forward<F>(callback), order));
    return visitor.VisitExpected(root);
  } catch (const Error& err) {
    return Unexpected(err);
  } catch (const std::exception& err) {
    return Unexpected(Error("InternalError", err.what(), ""));
  } catch (...) {
    return Unexpected(Error("InternalError", "Unknown structural walk error", ""));
  }
}

/*!
 * \brief Throwing error over \ref structuralWalkExpected.
 *
 * See \ref structuralWalkExpected for callback semantics and traversal behavior.
 *
 * \tparam T Node types that should trigger the callback.
 * \tparam F Callback type.
 * \param root The root object to visit.
 * \param callback Callback invoked for matching nodes.
 * \param order Whether to invoke the callback before or after visiting children.
 * \return ``std::nullopt`` if traversal completed, or the interrupt returned by
 *         the callback.
 * \throws Error if traversal or the callback returned an error.
 *
 * \note Return type of callback should be Expected<Variant<VisitInterrupt, bool>>.
 */
template <typename... T, typename F>
Optional<VisitInterrupt> structuralWalk(const ObjectRef& root, F&& callback, WalkOrder order) {
  return structuralWalkExpected<T...>(root, std::forward<F>(callback), order).value();
}

}  // namespace ffi
}  // namespace tvm
#endif  // TVM_FFI_EXTRA_STRUCTURAL_VISIT_H_
