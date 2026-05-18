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
//! Rust bindings for `tvm::ffi::StructuralVisitor`.
//!
//! The C++ side exposes the visitor as an object whose payload contains a visit
//! vtable pointer and a def-region tag. This module
//! mirrors that object layout so visitors can be authored in either language
//! and interoperate over the FFI.

use crate::any::{Any, AnyView};
use crate::derive::{Object, ObjectRef};
use crate::error::{Error, Result};
use crate::function::Function;
use crate::object::{Object, ObjectArc, ObjectCore, ObjectRef as BaseObjectRef, ObjectRefCore};

use std::ffi::{c_int, c_void};
use std::marker::PhantomData;
use std::pin::Pin;

use tvm_ffi_sys::{
    TVMFFIAny, TVMFFIByteArray, TVMFFIFieldInfo, TVMFFIGetTypeAttrColumn, TVMFFIGetTypeInfo,
    TVMFFIObject, TVMFFITypeAttrColumn, TVMFFITypeIndex as TypeIndex,
};

//-----------------------------------------------------
// DefRegionKind
//-----------------------------------------------------

/// Mirrors C++ `TVMFFIDefRegionKind`.
///
/// Identifies whether the visitor is currently inside a def-region (a
/// binding scope that affects structural eq/hash semantics).
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DefRegionKind {
    None = 0,
    Recursive = 1,
    NonRecursive = 2,
}

impl DefRegionKind {
    /// Convert from the raw `i32` carried in `StructuralVisitorObj::def_region_mode`.
    pub fn from_i32(value: i32) -> Result<Self> {
        match value {
            0 => Ok(Self::None),
            1 => Ok(Self::Recursive),
            2 => Ok(Self::NonRecursive),
            _ => crate::bail!(crate::error::VALUE_ERROR, "Invalid DefRegionKind: {}", value),
        }
    }
}

//-----------------------------------------------------
// VisitInterrupt
//-----------------------------------------------------

/// Object node carrying the optional payload for an interrupted structural visit.
///
/// Mirrors C++ `tvm::ffi::VisitInterruptObj`.
#[repr(C)]
#[derive(Object)]
#[type_key = "ffi.VisitInterrupt"]
pub struct VisitInterruptObj {
    object: Object,
    /// Payload returned with the interrupt, or FFI None for no payload.
    pub value: TVMFFIAny,
}

/// ABI-stable owned `VisitInterrupt` ref class.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct VisitInterrupt {
    data: ObjectArc<VisitInterruptObj>,
}

//-----------------------------------------------------
// StructuralVisitor
//-----------------------------------------------------

/// C-ABI signature for structural visit.
///
/// Matches `tvm::ffi::FStructuralVisit` on the C++ side. The first argument
/// is the active visitor and is also the recursion context. The return value is
/// a raw FFI Any encoding C++ `Expected<Optional<VisitInterrupt>>`: `None`
/// means no interrupt, `VisitInterrupt` halts traversal, and `ffi.Error`
/// reports an error.
///
/// # Error propagation
///
/// Rust callbacks catch panics / `Err(_)` and return an owned `ffi.Error` in
/// the same return slot. The C++ caller decodes that object and rethrows it.
pub type StructuralVisitCallType = unsafe extern "C" fn(
    visitor: *mut StructuralVisitorObj,
    value: *const BaseObjectRef,
) -> TVMFFIAny;

/// Layout-compatible mirror of C++ `tvm::ffi::StructuralVisitorVTable`.
#[repr(C)]
pub struct StructuralVisitorVTable {
    /// Required C-ABI visit entry.
    pub visit: Option<StructuralVisitCallType>,
}

/// Layout-compatible mirror of C++ `tvm::ffi::StructuralVisitorObj`.
///
/// Field order after the object header MUST match the C++ object for ABI
/// compatibility:
///
/// 1. `vtable` — required C-ABI dispatch table.
/// 2. `def_region_mode` — current def-region context.
#[repr(C)]
#[derive(Object)]
#[type_key = "ffi.StructuralVisitor"]
pub struct StructuralVisitorObj {
    object: Object,
    pub vtable: *const StructuralVisitorVTable,
    pub def_region_mode: c_int,
}

/// ABI-stable owned structural visitor ref class.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct StructuralVisitor {
    data: ObjectArc<StructuralVisitorObj>,
}

const STRUCTURAL_VISIT_ATTR: &str = "__ffi_structural_visit__";
const FIELD_FLAG_S_EQ_HASH_IGNORE: i64 = 1 << 3;
const FIELD_FLAG_S_EQ_HASH_DEF_RECURSIVE: i64 = 1 << 4;
const FIELD_FLAG_S_EQ_HASH_DEF_NON_RECURSIVE: i64 = 1 << 12;

fn decode_visit_result(result: TVMFFIAny) -> Result<Option<VisitInterrupt>> {
    let result = unsafe { Any::from_raw_ffi_any(result) };
    if result.type_index() == TypeIndex::kTVMFFINone as i32 {
        return Ok(None);
    }
    if result.type_index() == TypeIndex::kTVMFFIError as i32 {
        return Err(result.try_into()?);
    }
    result.try_into().map(Some)
}

fn encode_visit_result(result: Option<VisitInterrupt>) -> Any {
    match result {
        Some(interrupt) => Any::from(interrupt),
        None => Any::new(),
    }
}

unsafe fn object_header(value: &BaseObjectRef) -> *mut TVMFFIObject {
    ObjectArc::as_raw(<BaseObjectRef as ObjectRefCore>::data(value)) as *mut TVMFFIObject
}

unsafe fn type_attr_at(column: *const TVMFFITypeAttrColumn, type_index: i32) -> TVMFFIAny {
    if column.is_null() {
        return TVMFFIAny::new();
    }
    let column = &*column;
    if type_index < column.begin_index || type_index >= column.begin_index + column.size {
        return TVMFFIAny::new();
    }
    *column.data.add((type_index - column.begin_index) as usize)
}

fn structural_visit_attr(type_index: i32) -> TVMFFIAny {
    unsafe {
        let attr_name = TVMFFIByteArray::from_str(STRUCTURAL_VISIT_ATTR);
        type_attr_at(TVMFFIGetTypeAttrColumn(&attr_name), type_index)
    }
}

unsafe fn field_value(object: *mut TVMFFIObject, field_info: &TVMFFIFieldInfo) -> Result<Any> {
    let getter = field_info.getter.ok_or_else(|| {
        Error::new(
            crate::error::TYPE_ERROR,
            "StructuralVisitor requires reflected fields to provide a getter",
            "",
        )
    })?;
    let field_addr = (object as *mut u8).add(field_info.offset as usize) as *mut c_void;
    let mut result = Any::new();
    crate::check_safe_call!(getter(field_addr, result.as_data_ptr()))?;
    Ok(result)
}

impl StructuralVisitor {
    /// Visit a value, dispatching through this visitor's vtable.
    ///
    /// Returns `Ok(None)` to continue traversal, `Ok(Some(interrupt))` to halt
    /// with an interrupt payload, or `Err(_)` if the underlying visitor raised
    /// an error.
    pub fn visit(&mut self, value: &BaseObjectRef) -> Result<Option<VisitInterrupt>> {
        unsafe {
            let visitor = ObjectArc::as_raw_mut(&mut self.data);
            let vtable = (*visitor)
                .vtable
                .as_ref()
                .expect("StructuralVisitor::vtable is null");
            let visit = vtable.visit.expect("StructuralVisitor::vtable.visit is null");
            decode_visit_result(visit(visitor, value as *const BaseObjectRef))
        }
    }

    /// Invoke the base/default traversal implementation.
    ///
    /// Custom Rust visitors should call this after their own per-node behavior
    /// when they want the standard type-attribute dispatch and reflection
    /// fallback to recurse into children.
    pub fn default_visit(&mut self, value: &BaseObjectRef) -> Result<Option<VisitInterrupt>> {
        unsafe {
            let type_index = (*object_header(value)).type_index;
            let attr = structural_visit_attr(type_index);
            let attr_view = AnyView::from_raw_ffi_any(attr);

            if attr_view.type_index() == TypeIndex::kTVMFFIOpaquePtr as i32 {
                let visit_fn: StructuralVisitCallType =
                    std::mem::transmute(attr.data_union.v_ptr);
                let visitor = ObjectArc::as_raw_mut(&mut self.data);
                return decode_visit_result(visit_fn(visitor, value as *const BaseObjectRef));
            }

            if attr_view.type_index() == TypeIndex::kTVMFFIFunction as i32 {
                let func: Function = attr_view.try_into()?;
                let visitor = ObjectArc::as_raw_mut(&mut self.data);
                let visit_child = Function::from_packed(move |args| -> Result<Any> {
                    crate::ensure!(
                        args.len() == 2,
                        crate::error::TYPE_ERROR,
                        "Structural visit child callback expects 2 arguments"
                    );
                    let child: BaseObjectRef = args[0].try_into()?;
                    let kind: i32 = args[1].try_into()?;
                    let kind = DefRegionKind::from_i32(kind)?;
                    let visitor_arc = ObjectArc::<StructuralVisitorObj>::from_raw(visitor);
                    let mut visitor_ref = std::mem::ManuallyDrop::new(StructuralVisitor {
                        data: visitor_arc,
                    });
                    let result = (&mut *visitor_ref)
                        .with_def_region_kind(kind, |visitor| visitor.visit(&child))?;
                    Ok(encode_visit_result(result))
                });
                let args = [AnyView::from(value), AnyView::from(&visit_child)];
                return func.call_packed(&args)?.try_into();
            }

            if attr_view.type_index() != TypeIndex::kTVMFFINone as i32 {
                crate::bail!(
                    crate::error::TYPE_ERROR,
                    "{} must be an opaque function pointer or ffi.Function",
                    STRUCTURAL_VISIT_ATTR
                );
            }

            self.visit_reflected_fields(value)
        }
    }

    fn visit_reflected_fields(&mut self, value: &BaseObjectRef) -> Result<Option<VisitInterrupt>> {
        unsafe {
            let object = object_header(value);
            let type_info = TVMFFIGetTypeInfo((*object).type_index);
            crate::ensure!(
                !type_info.is_null(),
                crate::error::TYPE_ERROR,
                "Cannot find type info for structural visit"
            );

            let fields =
                std::slice::from_raw_parts((*type_info).fields, (*type_info).num_fields as usize);
            for field_info in fields {
                if field_info.flags & FIELD_FLAG_S_EQ_HASH_IGNORE != 0 {
                    continue;
                }

                let mut kind = DefRegionKind::None;
                if field_info.flags & FIELD_FLAG_S_EQ_HASH_DEF_RECURSIVE != 0 {
                    kind = DefRegionKind::Recursive;
                } else if field_info.flags & FIELD_FLAG_S_EQ_HASH_DEF_NON_RECURSIVE != 0 {
                    kind = DefRegionKind::NonRecursive;
                }

                let field_value = field_value(object, field_info)?;
                let child: BaseObjectRef = field_value.try_into()?;
                if let Some(interrupt) =
                    self.with_def_region_kind(kind, |visitor| visitor.visit(&child))?
                {
                    return Ok(Some(interrupt));
                }
            }
            Ok(None)
        }
    }

    /// Get the current def-region context.
    pub fn def_region_kind(&self) -> DefRegionKind {
        DefRegionKind::from_i32(self.data.def_region_mode).unwrap_or(DefRegionKind::None)
    }

    /// Temporarily switch the def-region context while invoking `callback`.
    pub fn with_def_region_kind<R>(
        &mut self,
        kind: DefRegionKind,
        callback: impl FnOnce(&mut Self) -> R,
    ) -> R {
        let saved = self.data.def_region_mode;
        self.data.def_region_mode = kind as c_int;
        let result = callback(self);
        self.data.def_region_mode = saved;
        result
    }
}

impl StructuralVisitorObj {
    fn with_null_vtable() -> Self {
        Self {
            object: Object::new(),
            vtable: std::ptr::null(),
            def_region_mode: DefRegionKind::None as c_int,
        }
    }
}

//-----------------------------------------------------
// RustStructuralVisitor: build a StructuralVisitor from a Rust closure
//-----------------------------------------------------

/// A Rust-defined structural visitor backed by a closure.
///
/// The closure receives the active visitor (so it can recurse via
/// `visitor.visit(...)`) and the value being visited. Returning:
///
/// * `Ok(None)` continues traversal,
/// * `Ok(Some(interrupt))` halts with an interrupt payload,
/// * `Err(_)` returns an owned `ffi.Error` in the FFI return slot, where
///   the C++ caller will rethrow it.
///
/// Panics inside the closure are caught and converted into a `RuntimeError`
/// so that the C++ caller never observes an unwind across the FFI boundary.
///
pub struct RustStructuralVisitor<F>
where
    F: FnMut(&mut StructuralVisitor, &BaseObjectRef) -> Result<Option<VisitInterrupt>>,
{
    visitor: StructuralVisitor,
    _marker: PhantomData<F>,
}

#[repr(C)]
struct RustStructuralVisitorObj<F>
where
    F: FnMut(&mut StructuralVisitor, &BaseObjectRef) -> Result<Option<VisitInterrupt>>,
{
    visitor: StructuralVisitorObj,
    vtable: StructuralVisitorVTable,
    callback: F,
}

impl<F> RustStructuralVisitorObj<F>
where
    F: FnMut(&mut StructuralVisitor, &BaseObjectRef) -> Result<Option<VisitInterrupt>> + 'static,
{
    fn new(callback: F) -> Self {
        Self {
            visitor: StructuralVisitorObj::with_null_vtable(),
            vtable: StructuralVisitorVTable {
                visit: Some(RustStructuralVisitor::<F>::visit_thunk),
            },
            callback,
        }
    }
}

unsafe impl<F> ObjectCore for RustStructuralVisitorObj<F>
where
    F: FnMut(&mut StructuralVisitor, &BaseObjectRef) -> Result<Option<VisitInterrupt>> + 'static,
{
    const TYPE_KEY: &'static str = StructuralVisitorObj::TYPE_KEY;

    fn type_index() -> i32 {
        StructuralVisitorObj::type_index()
    }

    unsafe fn object_header_mut(this: &mut Self) -> &mut tvm_ffi_sys::TVMFFIObject {
        StructuralVisitorObj::object_header_mut(&mut this.visitor)
    }
}

impl<F> RustStructuralVisitor<F>
where
    F: FnMut(&mut StructuralVisitor, &BaseObjectRef) -> Result<Option<VisitInterrupt>> + 'static,
{
    /// Construct a new Rust-defined structural visitor.
    pub fn new(callback: F) -> Pin<Box<Self>> {
        unsafe {
            let mut callback_arc = ObjectArc::new(RustStructuralVisitorObj::<F>::new(callback));
            let callback_ptr = ObjectArc::as_raw_mut(&mut callback_arc);
            (*callback_ptr).visitor.vtable = &(*callback_ptr).vtable;
            let visitor_arc = ObjectArc::<StructuralVisitorObj>::from_raw(
                ObjectArc::into_raw(callback_arc) as *const StructuralVisitorObj,
            );
            Box::pin(Self {
                visitor: StructuralVisitor { data: visitor_arc },
                _marker: PhantomData,
            })
        }
    }

    /// Get a mutable reference to the underlying [`StructuralVisitor`].
    ///
    /// Useful when the visitor must be passed by `&mut StructuralVisitor` to
    /// another API that operates on the C-ABI shape directly.
    pub fn as_visitor_mut(self: Pin<&mut Self>) -> &mut StructuralVisitor {
        unsafe { &mut Pin::into_inner_unchecked(self).visitor }
    }

    /// Convenience: invoke [`StructuralVisitor::visit`] on this visitor.
    pub fn visit(
        self: Pin<&mut Self>,
        value: &BaseObjectRef,
    ) -> Result<Option<VisitInterrupt>> {
        self.as_visitor_mut().visit(value)
    }

    /// Visit thunk plugged into the visitor vtable.
    ///
    /// Translates between Rust's panic/`Result`-based error reporting and the
    /// FFI's "return None / VisitInterrupt / Error as TVMFFIAny" convention.
    unsafe extern "C" fn visit_thunk(
        visitor: *mut StructuralVisitorObj,
        value: *const BaseObjectRef,
    ) -> TVMFFIAny {
        let panic_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let this = &mut *(visitor as *mut RustStructuralVisitorObj<F>);
            let value_ref = &*value;
            let visitor_arc = ObjectArc::<StructuralVisitorObj>::from_raw(visitor);
            let mut visitor_ref = std::mem::ManuallyDrop::new(StructuralVisitor {
                data: visitor_arc,
            });
            (this.callback)(&mut *visitor_ref, value_ref)
        }));

        match panic_result {
            Ok(Ok(None)) => TVMFFIAny::new(),
            Ok(Ok(Some(interrupt))) => Any::into_raw_ffi_any(Any::from(interrupt)),
            Ok(Err(err)) => Any::into_raw_ffi_any(Any::from(err)),
            Err(_panic) => {
                let err = Error::new(
                    crate::error::RUNTIME_ERROR,
                    "panic in Rust structural visit callback",
                    "",
                );
                Any::into_raw_ffi_any(Any::from(err))
            }
        }
    }
}
