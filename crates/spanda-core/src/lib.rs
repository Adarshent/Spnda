//! Spanda Core Engine (Spanda Research)
//!
//! Sub-microsecond epistemic uncertainty quantification & LLM guardrail gateway:
//! - Discrete Equivalence Class Partitioning & Exact-Match Normalized Entropy (R_sc)
//! - 7-State Epistemic Consensus State Machine
//! - Attractor Basin Curvature & Mode Collapse Defense
//! - Multi-Path Reasoning Coherence Verification

pub mod math;
pub mod gateway;

pub use math::{SpandaAuditReceipt, SpandaEngine, EpistemicState};
pub use gateway::{create_router, GatewayState};

use std::ffi::{CStr, CString};
use std::os::raw::c_char;

/// C-compatible FFI function for direct sub-microsecond invocation from Python / C / Go.
///
/// Input:
/// - `samples_json`: JSON array of strings (e.g. `["42", "42", "41"]`)
/// - `context`: Optional C string
/// - `threshold`: Float threshold (e.g. 0.35)
///
/// Returns:
/// - Heap-allocated C string containing JSON serialized `SpandaAuditReceipt`.
///   Caller must free using `spanda_free_string`.
#[no_mangle]
pub extern "C" fn spanda_eval_c(
    samples_json: *const c_char,
    context: *const c_char,
    threshold: f64,
) -> *mut c_char {
    if samples_json.is_null() {
        return std::ptr::null_mut();
    }

    let c_str = unsafe { CStr::from_ptr(samples_json) };
    let json_str = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };

    let samples: Vec<String> = match serde_json::from_str(json_str) {
        Ok(v) => v,
        Err(_) => return std::ptr::null_mut(),
    };

    let ctx = if !context.is_null() {
        unsafe { CStr::from_ptr(context).to_str().ok() }
    } else {
        None
    };

    let receipt = SpandaEngine::evaluate(&samples, ctx, threshold);
    let output_json = match serde_json::to_string(&receipt) {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };

    let c_output = match CString::new(output_json) {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };

    c_output.into_raw()
}

/// Frees memory allocated by `spanda_eval_c`.
#[no_mangle]
pub extern "C" fn spanda_free_string(s: *mut c_char) {
    if !s.is_null() {
        unsafe {
            let _ = CString::from_raw(s);
        }
    }
}
