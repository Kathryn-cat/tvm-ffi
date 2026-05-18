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
#include <tvm/ffi/extra/base.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/optional.h>
#include <tvm/ffi/reflection/access_path.h>
#include <tvm/ffi/reflection/registry.h>

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
class StructuralVisitor;

/*!
 * \brief C-ABI visit function pointer.
 *
 * Used as the primary entry point of \ref StructuralVisitorObj so that non-C++
 * bindings (e.g. Rust) can implement and invoke visitors without crossing a
 * C++ exception boundary.
 *
 * \param visitor The active visitor used to recurse into structural children.
 * \param value The object being visited.
 * \return A by-value FFI Any encoding an
 *         ``Expected<Optional<VisitInterrupt>>``. ``None`` means no interrupt,
 *         ``VisitInterrupt`` halts traversal, and ``ffi.Error`` reports an error.
 *
 * \note Implementations should catch language-specific exceptions and return
 *       an ``ffi.Error`` value instead. This keeps errors in the regular return
 *       slot and avoids an additional out pointer on the common path.
 */
using FStructuralVisit = TVMFFIAny (*)(StructuralVisitorObj* visitor, const ObjectRef& value);

/*!
 * \brief C-ABI vtable for structural visitor dispatch.
 *
 * This mirrors the implementation mechanism used for C++ virtual methods while
 * keeping the ABI consumable from non-C++ languages.
 */
struct StructuralVisitorVTable {
  /*! \brief Required C-ABI visit entry. */
  FStructuralVisit visit;
};

/*!
 * \brief Object node backing a structural visitor.
 *
 * The visitor is an active traversal context. It is ref-counted so it can cross
 * FFI boundaries, but one underlying visitor object should not be used for
 * overlapping top-level traversals.
 *
 * Construction modes:
 * - Default-constructed visitors dispatch through the per-type structural
 *   visit attribute registry (\c reflection::type_attr::kStructuralVisit) and
 *   fall back to a reflection-driven field walk when no override is registered.
 *
 * The class deliberately avoids C++ virtual dispatch. Custom dispatch is
 * expressed through the explicit vtable rather than virtual overrides.
 */
class TVM_FFI_DLL StructuralVisitorObj : public Object {
 public:
  // -------- ABI layout: keep these fields after Object and in this order ----
  /*! \brief Required C-ABI vtable. Never null on a constructed visitor. */
  const StructuralVisitorVTable* vtable = nullptr;

  // --------------------------- C++-only API ---------------------------------

  /*!
   * \brief Construct the default structural visitor.
   *
   * Wires up the vtable to the default dispatcher implemented in
   * \c structural_visit.cc, which consults the structural-visit type attribute
   * registry and falls back to a reflection-driven field walk.
   */
  StructuralVisitorObj();

  /*!
   * \brief Visit a value, dispatching through this visitor's vtable.
   *
   * The callback returns a ``TVMFFIAny`` by value. This method decodes it as an
   * ``Expected<Optional<VisitInterrupt>>`` and throws if the result is an error.
   *
   * \param value The object to visit.
   * \return ``std::nullopt`` to continue traversal, or a \ref VisitInterrupt
   *         to halt the entire visit.
   */
  Optional<VisitInterrupt> Visit(const ObjectRef& value);

  /*!
   * \brief Default visit behavior: type-attr lookup, then reflection fallback.
   *
   * Custom visitors typically invoke this after performing their own
   * type-specific handling to obtain standard recursion semantics. Any
   * exceptions raised by registered visit hooks propagate as C++ exceptions.
   */
  Optional<VisitInterrupt> DefaultVisit(const ObjectRef& value);

  /*! \brief Get the current def-region context. */
  TVM_FFI_INLINE TVMFFIDefRegionKind def_region_kind() const { return def_region_mode; }

  /*!
   * \brief Temporarily switch the def-region context while invoking \p callback.
   * \param kind The def-region kind to set during the callback.
   * \param callback A nullary callable whose return value is forwarded.
   * \return The return value of \p callback.
   */
  template <typename Callback>
  TVM_FFI_INLINE auto WithDefRegionKind(TVMFFIDefRegionKind kind, Callback&& callback)
      -> decltype(std::forward<Callback>(callback)()) {
    class Scope {
     public:
      Scope(StructuralVisitorObj* visitor, TVMFFIDefRegionKind kind)
          : visitor_(visitor), old_kind_(visitor->def_region_mode) {
        visitor_->def_region_mode = kind;
      }
      ~Scope() { visitor_->def_region_mode = old_kind_; }
      Scope(const Scope&) = delete;
      Scope& operator=(const Scope&) = delete;

     private:
      StructuralVisitorObj* visitor_;
      TVMFFIDefRegionKind old_kind_;
    };
    Scope scope(this, kind);
    return std::forward<Callback>(callback)();
  }

  /// \cond Doxygen_Suppress
  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("ffi.StructuralVisitor", StructuralVisitorObj, Object);
  /// \endcond

 protected:
  /*! \brief Current def-region context for structural eq/hash semantics. */
  TVMFFIDefRegionKind def_region_mode = kTVMFFIDefRegionKindNone;
};

/*!
 * \brief ObjectRef wrapper for StructuralVisitorObj.
 */
class TVM_FFI_DLL StructuralVisitor : public ObjectRef {
 public:
  /*! \brief Construct the default structural visitor. */
  StructuralVisitor();
  /*! \brief Construct from an existing object pointer. */
  explicit StructuralVisitor(ObjectPtr<StructuralVisitorObj> n) : ObjectRef(std::move(n)) {}

  /*! \brief Visit a value, dispatching through this visitor's vtable. */
  Optional<VisitInterrupt> Visit(const ObjectRef& value);

  /*! \brief Default visit behavior: type-attr lookup, then reflection fallback. */
  Optional<VisitInterrupt> DefaultVisit(const ObjectRef& value);

  /*! \brief Get the current def-region context. */
  TVM_FFI_INLINE TVMFFIDefRegionKind def_region_kind() const { return get()->def_region_kind(); }

  /*!
   * \brief Temporarily switch the def-region context while invoking \p callback.
   * \param kind The def-region kind to set during the callback.
   * \param callback A nullary callable whose return value is forwarded.
   * \return The return value of \p callback.
   */
  template <typename Callback>
  TVM_FFI_INLINE auto WithDefRegionKind(TVMFFIDefRegionKind kind, Callback&& callback)
      -> decltype(std::forward<Callback>(callback)()) {
    return get()->WithDefRegionKind(kind, std::forward<Callback>(callback));
  }

  /// \cond Doxygen_Suppress
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(StructuralVisitor, ObjectRef, StructuralVisitorObj);
  /// \endcond
};

// ---------------------------------------------------------------------------
// Walk helpers.
// ---------------------------------------------------------------------------

enum class WalkOrder : int32_t {
  kPreOrder = 0,
  kPostOrder = 1,
};

enum class WalkResult : int32_t {
  kAdvance = 0,
  kSkip = 1,
};

namespace details {
template <typename... T>
struct FirstTypeImpl;

template <typename T, typename... Rest>
struct FirstTypeImpl<T, Rest...> {
  using type = T;
};

template <typename... T>
using FirstType = typename FirstTypeImpl<T...>::type;
}  // namespace details

template <typename... T>
using StructuralWalkCallbackArg =
    std::conditional_t<sizeof...(T) == 1, details::FirstType<T...>, Variant<T...>>;

template <typename U, typename... T>
void TryMatchCallbackArg(AnyView value, std::optional<StructuralWalkCallbackArg<T...>>* result) {
  if (result->has_value()) return;
  if (std::optional<U> opt = value.as<U>()) {
    if constexpr (sizeof...(T) == 1) {
      *result = *opt;
    } else {
      *result = Variant<T...>(*opt);
    }
  }
}

template <typename... T>
std::optional<StructuralWalkCallbackArg<T...>> MatchCallbackArg(AnyView value) {
  std::optional<StructuralWalkCallbackArg<T...>> result = std::nullopt;
  (TryMatchCallbackArg<T, T...>(value, &result), ...);
  return result;
}

inline Optional<VisitInterrupt> ToInterrupt(Variant<WalkResult, VisitInterrupt> result) {
  if (std::optional<VisitInterrupt> interrupt = result.as<VisitInterrupt>()) {
    return *interrupt;
  }
  return std::nullopt;
}

template <typename... T, typename F>
Optional<VisitInterrupt> structuralWalk(AnyView root, F&& callback,
                                        WalkOrder order = WalkOrder::kPreOrder) {
  class VisitorObj : public StructuralVisitorObj {
   public:
    explicit VisitorObj(F&& callback, WalkOrder order)
        : callback_(std::forward<F>(callback)), order_(order) {
      this->vtable = VTable();
    }

    VisitorObj(const VisitorObj&) = delete;
    VisitorObj& operator=(const VisitorObj&) = delete;

   private:
    static TVMFFIAny VisitThunk(StructuralVisitorObj* visitor, const ObjectRef& value) {
      try {
        auto* self = static_cast<VisitorObj*>(visitor);
        TVM_FFI_ICHECK_NOTNULL(self);
        return details::AnyUnsafe::MoveAnyToTVMFFIAny(
            Any(Expected<Optional<VisitInterrupt>>(self->VisitImpl(visitor, value))));
      } catch (const Error& error) {
        return details::AnyUnsafe::MoveAnyToTVMFFIAny(Any(error));
      } catch (const std::exception& ex) {
        return details::AnyUnsafe::MoveAnyToTVMFFIAny(Any(Error("InternalError", ex.what(), "")));
      }
    }

    Optional<VisitInterrupt> VisitImpl(StructuralVisitorObj* visitor, const ObjectRef& value) {
      auto& callback_fn = callback_;
      WalkOrder order = order_;
      if (order == WalkOrder::kPreOrder) {
        if (auto matched = MatchCallbackArg<T...>(value)) {
          Variant<WalkResult, VisitInterrupt> result =
              callback_fn(*matched, visitor->def_region_kind());
          if (auto interrupt = ToInterrupt(result)) {
            return interrupt;
          }
          if (result.template get<WalkResult>() == WalkResult::kSkip) {
            return std::nullopt;
          }
        }
      }
      if (auto interrupt = visitor->DefaultVisit(value)) {
        return interrupt;
      }
      if (order == WalkOrder::kPostOrder) {
        if (auto matched = MatchCallbackArg<T...>(value)) {
          Variant<WalkResult, VisitInterrupt> result =
              callback_fn(*matched, visitor->def_region_kind());
          if (auto interrupt = ToInterrupt(result)) {
            return interrupt;
          }
        }
      }
      return std::nullopt;
    }

    std::decay_t<F> callback_;
    WalkOrder order_;

    static const StructuralVisitorVTable* VTable() {
      static constexpr StructuralVisitorVTable vtable = {&VisitorObj::VisitThunk};
      return &vtable;
    }
  };

  StructuralVisitor visitor(make_object<VisitorObj>(std::forward<F>(callback), order));
  return visitor.Visit(root.cast<ObjectRef>());
}

}  // namespace ffi
}  // namespace tvm
#endif  // TVM_FFI_EXTRA_STRUCTURAL_VISIT_H_
