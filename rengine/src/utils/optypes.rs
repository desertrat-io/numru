#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::float32x4_t;

/// the argument type is an unsafe function that receives two 32 bit register definitions
/// and have two operands each representing a vector to be loaded
/// into the specific vector registers
pub type IntrinsicOp = unsafe fn(float32x4_t) -> float32x4_t;
pub type IntrinsicOp2 = unsafe fn(float32x4_t, float32x4_t) -> float32x4_t;
pub type IntrinsicOp3 = unsafe fn(float32x4_t, float32x4_t, float32x4_t) -> float32x4_t;

pub type ScalarOp = fn(f32) -> f32;
pub type ScalarOp2 = fn(f32, f32) -> f32;
pub type ScalarOp3 = fn(f32, f32, f32) -> f32;

pub type VecOp = fn(&[f32], &mut [f32]);
pub type VecOp2 = fn(&[f32], &[f32], &mut [f32]);
pub type VecOp3 = fn(&[f32], &[f32], &[f32], &mut [f32]);
