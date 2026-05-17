use rayon::prelude::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::float32x4_t;
use std::arch::aarch64::{uint32x4_t, vandq_u32, vcvtq_f32_u32, vdupq_n_u32, vld1q_f32, vst1q_f32};

const NUM_FLOATS_32: usize = 4;
const PAR_CHUNK_SIZE: usize = 4096; // BYTES

/// the argument type is an unsafe function that receives two 32 bit register definitions
/// and have two operands each representing a vector to be loaded
/// into the specific vector registers
pub type IntrinsicOp = unsafe fn(float32x4_t) -> float32x4_t;
pub type IntrinsicOp2 = unsafe fn(float32x4_t, float32x4_t) -> float32x4_t;
pub type IntrinsicOp3 = unsafe fn(float32x4_t, float32x4_t, float32x4_t) -> float32x4_t;

pub type IntrinsicBool2 = unsafe fn(float32x4_t, float32x4_t) -> uint32x4_t;

pub type ScalarOp = fn(f32) -> f32;
pub type ScalarOp2 = fn(f32, f32) -> f32;
pub type ScalarOp3 = fn(f32, f32, f32) -> f32;

pub type VecOp = fn(&[f32], &mut [f32]);
pub type VecOp2 = fn(&[f32], &[f32], &mut [f32]);
pub type VecOp3 = fn(&[f32], &[f32], &[f32], &mut [f32]);

#[derive(Clone, Copy)]
pub enum Mode {
    Normal,
    Neon,
    ParNeon,
}

pub fn binary_op_1(left: &[f32], result: &mut [f32], op: ScalarOp) {
    assert_eq!(left.len(), result.len());
    for i in 0..left.len() {
        result[i] = op(left[i]);
    }
}

pub fn binary_op_2(left: &[f32], right: &[f32], result: &mut [f32], op: ScalarOp2) {
    assert_eq!(left.len(), right.len());
    assert_eq!(left.len(), result.len());
    for i in 0..left.len() {
        result[i] = op(left[i], right[i]);
    }
}

pub fn binary_op_3(left: &[f32], middle: &[f32], right: &[f32], result: &mut [f32], op: ScalarOp3) {
    assert_eq!(left.len(), middle.len());
    assert_eq!(left.len(), right.len());
    assert_eq!(left.len(), result.len());
    for i in 0..left.len() {
        result[i] = op(left[i], middle[i], right[i]);
    }
}

/// conditional intrinsics implementations
/// 64bit ARM only right now
/// op is a defined vector operation that define the intrinsic instruction to use
/// when dealing with a vector. this is abstracted to allow any intrinsic that operates on vectors
/// the vector registers used are always the same, and all rules regarding size are always the same
/// registers are managed by the CPU
pub fn neon_1(left: &[f32], result: &mut [f32], intrinsic_op: IntrinsicOp, scalar_op: ScalarOp) {
    assert_eq!(left.len(), result.len());
    let len = left.len();
    let mem_chunks = len / NUM_FLOATS_32;
    unsafe {
        for i in 0..mem_chunks {
            let idx = i * NUM_FLOATS_32;
            let v_left = vld1q_f32(left.as_ptr().add(idx));
            let v_result = intrinsic_op(v_left);
            vst1q_f32(result.as_mut_ptr().add(idx), v_result);
        }
    }

    for i in (mem_chunks * NUM_FLOATS_32)..len {
        result[i] = scalar_op(left[i]);
    }
}

// some intrinsics return uint values instead of f32 for the underlying operation
// to keep implementations clean, we use the uint versions of particular vector functions
// instead of f32 in cases where a boolean is being used that cannot be easily described as a
// floating point number
pub fn neon_bool_2(
    left: &[f32],
    right: &[f32],
    result: &mut [f32],
    intrinsic_bool: IntrinsicBool2,
    scalar_op: ScalarOp2,
) {
    assert_eq!(left.len(), right.len());
    assert_eq!(left.len(), result.len());
    let len = left.len();
    let mem_chunks = len / NUM_FLOATS_32;
    unsafe {
        for i in 0..mem_chunks {
            let idx = i * NUM_FLOATS_32;
            // note the intrinsic functions used here are operating on u32 and not f32
            // however there's no need to change the underlying Array structure when what we need is
            // to just cast appropriately
            // all scalars operations for boolean handlers ensure that the result is always 0 or 1
            // since the result of the actual operation in the scalar operation returns a boolean
            // in the first place. So we are just making a conversion from bool -> u8 -> f32
            let v_left = vld1q_f32(left.as_ptr().add(idx));
            let v_right = vld1q_f32(right.as_ptr().add(idx));
            // the arch sets "greater than" to all 1's in the result register, so that needs
            // to be masked and converted back to our expected float
            // so v_result in this case is a bitbask of all 1's for true
            let v_result = intrinsic_bool(v_left, v_right);
            // 0xFFFFFFFF to 0x00000001
            let post_mask_result = vandq_u32(v_result, vdupq_n_u32(1));
            let final_floats = vcvtq_f32_u32(post_mask_result);
            vst1q_f32(result.as_mut_ptr().add(idx), final_floats);
        }
    }
    for i in (mem_chunks * NUM_FLOATS_32)..len {
        result[i] = scalar_op(left[i], right[i]);
    }
}
pub fn neon_2(
    left: &[f32],
    right: &[f32],
    result: &mut [f32],
    intrinsic_op: IntrinsicOp2,
    scalar_op: ScalarOp2,
) {
    assert_eq!(left.len(), right.len());
    assert_eq!(left.len(), result.len());

    let len = left.len();

    // may need tuning. loads 4 floats at a time
    let mem_chunks = len / NUM_FLOATS_32;

    // working at the CPU level for faster ops, hence unsafe is needed
    unsafe {
        for i in 0..mem_chunks {
            let idx = i * NUM_FLOATS_32;

            let v_left = vld1q_f32(left.as_ptr().add(idx));
            let v_right = vld1q_f32(right.as_ptr().add(idx));
            let v_result = intrinsic_op(v_left, v_right);
            vst1q_f32(result.as_mut_ptr().add(idx), v_result);
        }
    }

    for i in (mem_chunks * NUM_FLOATS_32)..len {
        result[i] = scalar_op(left[i], right[i]);
    }
}

pub fn neon_3(
    left: &[f32],
    middle: &[f32],
    right: &[f32],
    result: &mut [f32],
    intrinsic_op: IntrinsicOp3,
    scalar_op: ScalarOp3,
) {
    assert_eq!(left.len(), middle.len());
    assert_eq!(left.len(), right.len());
    assert_eq!(left.len(), result.len());
    let len = left.len();

    // may need tuning. loads 4 floats at a time
    let mem_chunks = len / NUM_FLOATS_32;

    // working at the CPU level for faster ops, hence unsafe is needed
    unsafe {
        for i in 0..mem_chunks {
            let idx = i * NUM_FLOATS_32;

            let v_left = vld1q_f32(left.as_ptr().add(idx));
            let v_middle = vld1q_f32(middle.as_ptr().add(idx));
            let v_right = vld1q_f32(right.as_ptr().add(idx));
            let v_result = intrinsic_op(v_left, v_middle, v_right);
            vst1q_f32(result.as_mut_ptr().add(idx), v_result);
        }
    }

    for i in (mem_chunks * NUM_FLOATS_32)..len {
        result[i] = scalar_op(left[i], middle[i], right[i]);
    }
}

pub fn par_1(left: &[f32], result: &mut [f32], op: VecOp) {
    assert_eq!(left.len(), result.len());
    result
        .par_chunks_mut(PAR_CHUNK_SIZE)
        .zip(left.par_chunks(PAR_CHUNK_SIZE))
        .for_each(|(result_chunk, left_chunk)| {
            op(left_chunk, result_chunk);
        });
}
pub fn par_2(left: &[f32], right: &[f32], result: &mut [f32], op: VecOp2) {
    assert_eq!(left.len(), right.len());
    assert_eq!(left.len(), result.len());
    result
        .par_chunks_mut(PAR_CHUNK_SIZE)
        .zip(left.par_chunks(PAR_CHUNK_SIZE))
        .zip(right.par_chunks(PAR_CHUNK_SIZE))
        .for_each(|((result_chunk, left_chunk), right_chunk)| {
            op(left_chunk, right_chunk, result_chunk);
        });
}

pub fn par_3(
    left: &[f32],
    middle: &[f32],
    right: &[f32],
    result: &mut [f32],
    neon_math_op: VecOp3,
) {
    let mem_size = 4096; // bytes
    result
        .par_chunks_mut(mem_size)
        .zip(left.par_chunks(mem_size))
        .zip(middle.par_chunks(mem_size))
        .zip(right.par_chunks(mem_size))
        .for_each(
            |(((result_chunk, left_chunk), middle_chunk), right_chunk)| {
                neon_math_op(left_chunk, middle_chunk, right_chunk, result_chunk);
            },
        );
}
