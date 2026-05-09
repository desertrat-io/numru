//! # Arith
//!
//! Core arithmetic operations on vectors (first rank tensor if that's your thing)
//! Designed to be extensible, but opinionated defaults are provided for core runtime loops
//! As of today, no ownership is exchanged, only borrows. Thus proper lifetimes are up to the
//! consumer
//!
//! Additionally, while the high level api uses the Array struct for convenience, all arithmetic
//! operations are on raw slices only
//!
//! Note that due to optimization needed, all arithmetic is architecture dependent and will
//! be for the foreseeable future
//!
//! Some unary operations are not supported directly by the intrinsic library in stable rust versions
//! and are only available in nightly. for now we will not be using anything and just use slower
//! non par non neon implementations

use crate::data::array::Array;
use crate::matrix::ops::{
    binary_op_1, binary_op_2, binary_op_3, neon_1, neon_2, neon_3, par_1, par_2, par_3,
};
use rayon::prelude::*;
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{float32x4_t, vaddq_f32, vdivq_f32, vfmaq_f32, vmulq_f32, vsubq_f32};
use std::arch::aarch64::{vabsq_f32, vnegq_f32, vsqrtq_f32};

#[derive(Clone, Copy)]
pub enum Mode {
    Normal,
    Neon,
    ParNeon,
}

/// Single abstraction point of entry
/// The returned function is preconfigured to represent a fast operational loop based on the op
/// selected
/// Currently only operates on vectors of 32 bit floats
/// All functions returned have an accumulator in the last param position to represent the
/// vector arithmetic results
/// The returned generated function signature takes two array structs of equal length
/// and then a crate::data::arith::Mode
/// The mode tells the engine which calculation method to use (single, parallel, simd, parallel + simd)
/// to calculate the vector arithmetic operations, allowing consumers to pick the most efficient method
/// for their available runtime
// TODO: Just handle 32 bit cases for now

/// unary (1 vector) operations
pub fn abs(vector: Array, mode: Mode) -> Array {
    let mut result = Array::zero_padded(vector.len());
    match mode {
        Mode::Normal => binary_op_1(vector.slice(), result.mut_slice(), abs_scalar_32),
        Mode::Neon => neon_1(vector.slice(), result.mut_slice(), vabsq_f32, abs_scalar_32),
        Mode::ParNeon => par_1(vector.slice(), result.mut_slice(), |left, res| {
            neon_1(left, res, vabsq_f32, abs_scalar_32);
        }),
    }
    result
}

#[cfg(target_arch = "aarch64")]
pub fn log(vector: Array, mode: Mode) -> Array {
    let mut result = Array::zero_padded(vector.len());
    match mode {
        Mode::Normal => binary_op_1(vector.slice(), result.mut_slice(), log10_scalar_32),
        _ => panic!("log only supported in scalar mode for now"),
    }
    result
}

#[cfg(target_arch = "aarch64")]
pub fn exp(vector: Array, mode: Mode) -> Array {
    let mut result = Array::zero_padded(vector.len());
    match mode {
        Mode::Normal => binary_op_1(vector.slice(), result.mut_slice(), exp_scalar_32),
        _ => panic!("exp only supported in scalar mode for now"),
    }
    result
}

pub fn neg(vector: Array, mode: Mode) -> Array {
    let mut result = Array::zero_padded(vector.len());
    match mode {
        Mode::Normal => binary_op_1(vector.slice(), result.mut_slice(), neg_scalar_32),
        Mode::Neon => neon_1(vector.slice(), result.mut_slice(), vnegq_f32, neg_scalar_32),
        Mode::ParNeon => par_1(vector.slice(), result.mut_slice(), |left, res| {
            neon_1(left, res, vnegq_f32, neg_scalar_32);
        }),
    }
    result
}

pub fn sqrt(vector: Array, mode: Mode) -> Array {
    let mut result = Array::zero_padded(vector.len());
    match mode {
        Mode::Normal => binary_op_1(vector.slice(), result.mut_slice(), sqrt_scalar_32),
        Mode::Neon => neon_1(
            vector.slice(),
            result.mut_slice(),
            vsqrtq_f32,
            sqrt_scalar_32,
        ),
        Mode::ParNeon => par_1(vector.slice(), result.mut_slice(), |left, res| {
            neon_1(left, res, vsqrtq_f32, sqrt_scalar_32);
        }),
    }
    result
}

/// binary (2 vector) operations
#[cfg(target_arch = "aarch64")]
pub fn add(left: Array, right: Array, mode: Mode) -> Array {
    assert_eq!(left.len(), right.len());
    let (left_slice, right_slice, mut result) = as_flat_slices(&left, &right);
    let result_slice = result.mut_slice();
    match mode {
        Mode::Normal => binary_op_2(left_slice, right_slice, result_slice, add_scalar_32),
        Mode::Neon => neon_2(
            left_slice,
            right_slice,
            result_slice,
            vaddq_f32,
            add_scalar_32,
        ),
        Mode::ParNeon => par_2(left_slice, right_slice, result_slice, |left, right, res| {
            neon_2(left, right, res, vaddq_f32, add_scalar_32);
        }),
    }

    result
}

#[cfg(target_arch = "aarch64")]
pub fn sub(left: Array, right: Array, mode: Mode) -> Array {
    assert_eq!(left.len(), right.len());
    let (left_slice, right_slice, mut result) = as_flat_slices(&left, &right);
    let result_slice = result.mut_slice();
    match mode {
        Mode::Normal => binary_op_2(left_slice, right_slice, result_slice, sub_scalar_32),
        Mode::Neon => neon_2(
            left_slice,
            right_slice,
            result_slice,
            vsubq_f32,
            sub_scalar_32,
        ),
        Mode::ParNeon => par_2(left_slice, right_slice, result_slice, |left, right, res| {
            neon_2(left, right, res, vsubq_f32, sub_scalar_32);
        }),
    }
    result
}

#[cfg(target_arch = "aarch64")]
pub fn mul(left: Array, right: Array, mode: Mode) -> Array {
    assert_eq!(left.len(), right.len());
    let (left_slice, right_slice, mut result) = as_flat_slices(&left, &right);
    let result_slice = result.mut_slice();
    match mode {
        Mode::Normal => binary_op_2(left_slice, right_slice, result_slice, mul_scalar_32),
        Mode::Neon => neon_2(
            left_slice,
            right_slice,
            result_slice,
            vmulq_f32,
            mul_scalar_32,
        ),
        Mode::ParNeon => par_2(left_slice, right_slice, result_slice, |left, right, res| {
            neon_2(left, right, res, vmulq_f32, mul_scalar_32);
        }),
    }
    result
}

#[cfg(target_arch = "aarch64")]
pub fn div(left: Array, right: Array, mode: Mode) -> Array {
    assert_eq!(left.len(), right.len());
    let (left_slice, right_slice, mut result) = as_flat_slices(&left, &right);
    let result_slice = result.mut_slice();
    match mode {
        Mode::Normal => binary_op_2(left_slice, right_slice, result_slice, div_scalar_32),
        Mode::Neon => neon_2(
            left_slice,
            right_slice,
            result_slice,
            vdivq_f32,
            div_scalar_32,
        ),
        Mode::ParNeon => par_2(left_slice, right_slice, result_slice, |left, right, res| {
            neon_2(left, right, res, vdivq_f32, div_scalar_32);
        }),
    }
    result
}

/// Fused operations
#[cfg(target_arch = "aarch64")]
pub fn add_mul(left: Array, middle: Array, right: Array, mode: Mode) -> Array {
    assert_eq!(left.len(), middle.len());
    assert_eq!(left.len(), right.len());
    let left_slice = left.slice();
    let middle_slice = middle.slice();
    let right_slice = right.slice();
    let mut result = Array::zero_padded(left.len());
    let result_slice = result.mut_slice();
    match mode {
        Mode::Normal => binary_op_3(
            left_slice,
            middle_slice,
            right_slice,
            result_slice,
            add_mul_scalar_32,
        ),
        Mode::Neon => neon_3(
            left_slice,
            middle_slice,
            right_slice,
            result_slice,
            vfmaq_f32,
            add_mul_scalar_32,
        ),
        Mode::ParNeon => par_3(
            left_slice,
            middle_slice,
            right_slice,
            result_slice,
            |left, middle, right, res| {
                neon_3(left, middle, right, res, vfmaq_f32, add_mul_scalar_32);
            },
        ),
    }
    result
}

fn add_scalar_32(left: f32, right: f32) -> f32 {
    left + right
}

fn sub_scalar_32(left: f32, right: f32) -> f32 {
    left - right
}

fn mul_scalar_32(left: f32, right: f32) -> f32 {
    left * right
}

fn div_scalar_32(left: f32, right: f32) -> f32 {
    assert_ne!(right, 0.0, "Division by zero is undefined");
    left / right
}
fn add_mul_scalar_32(left: f32, middle: f32, right: f32) -> f32 {
    left + middle * right
}

fn abs_scalar_32(value: f32) -> f32 {
    value.abs()
}

fn neg_scalar_32(value: f32) -> f32 {
    -value
}

fn sqrt_scalar_32(value: f32) -> f32 {
    value.sqrt()
}

fn pow_scalar_32(value: f32, exp: f32) -> f32 {
    value.powf(exp)
}

#[cfg(target_arch = "aarch64")]
// TODO:
fn arm_svlogb_f32(value: float32x4_t) -> float32x4_t {
    // values are 128 bit in an ARM vector register of the same size
    // however those bits represent 4 32 bit floats

    value
}

fn exp_scalar_32(value: f32) -> f32 {
    value.exp()
}

// the intrinsic only operates on base 2, so run the conversion formula
fn log2_scalar_32(value: f32) -> f32 {
    value.log(10.0) / value.log(2.0)
}

// we need to be able to pass a reference to this operation around, otherwise
// we could just call this inline
fn log10_scalar_32(value: f32) -> f32 {
    value.log(10.0)
}

// convenience function to convert the Array struct to slices
// this is meant for use in dual vector operations
fn as_flat_slices<'a>(left: &'a Array, right: &'a Array) -> (&'a [f32], &'a [f32], Array) {
    (left.slice(), right.slice(), Array::zero_padded(left.len()))
}
