use rayon::prelude::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{
    float32x4_t, uint32x4_t, uint8x16_t, vandq_u32, vandq_u8, vcvtq_f32_u32, vdupq_n_u32,
    vdupq_n_u8, vld1q_f32, vld1q_u8, vst1q_f32, vst1q_u8,
};
use std::arch::aarch64::{int32x4_t, vld1q_s32};

const NUM_FLOAT_LANES_32: usize = 4;

// Rust bools are always one byte and will always be u8
const BOOL_SIZE: u8 = 0b0000_0001;
const NUM_BOOL_LANES_8: usize = 16;

const NUM_INT_LANES_32: usize = 4;
const PAR_CHUNK_SIZE: usize = 4096; // BYTES

/// the argument type is an unsafe function that receives two 32 bit register definitions
/// and have two operands each representing a vector to be loaded
/// into the specific vector registers
pub type IntrinsicOp = unsafe fn(float32x4_t) -> float32x4_t;
pub type IntrinsicOp2 = unsafe fn(float32x4_t, float32x4_t) -> float32x4_t;
pub type IntrinsicOp3 = unsafe fn(float32x4_t, float32x4_t, float32x4_t) -> float32x4_t;

pub type IntrinsicInt = unsafe fn(float32x4_t) -> uint32x4_t;

pub type IntrinsicSignedInt = unsafe fn(int32x4_t) -> i32;
pub type IntrinsicInt2 = unsafe fn(float32x4_t, float32x4_t) -> uint32x4_t;

pub type IntrinsicBool = unsafe fn(uint8x16_t) -> uint8x16_t;

pub type IntrinsicBool2 = unsafe fn(uint8x16_t, uint8x16_t) -> uint8x16_t;

pub type ScalarOp = fn(f32) -> f32;
pub type ScalarOp2 = fn(f32, f32) -> f32;
pub type ScalarOp3 = fn(f32, f32, f32) -> f32;

pub type SignedIntScalarVecOp = fn(&[i32], Option<&[i32]>) -> i32;

pub type SignedIntReductionOp = fn(i32, i32) -> i32;

pub type BooleanScalarOp = fn(bool) -> bool;

pub type BooleanScalarOp2 = fn(bool, bool) -> bool;

pub type VecOp = fn(&[f32], &mut [f32]);
pub type VecOp2 = fn(&[f32], &[f32], &mut [f32]);
pub type VecOp3 = fn(&[f32], &[f32], &[f32], &mut [f32]);

pub type BooleanVecOp = fn(&[bool], &mut [bool]);

pub type BooleanVecOp2 = fn(&[bool], &[bool], &mut [bool]);

pub type SignedIntVecOp = fn(&[i32]) -> i32;

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

pub fn boolean_binary_op_1(left: &[bool], result: &mut [bool], op: BooleanScalarOp) {
    assert_eq!(left.len(), result.len());
    for i in 0..left.len() {
        result[i] = op(left[i]);
    }
}

pub fn boolean_binary_op_2(
    left: &[bool],
    right: &[bool],
    result: &mut [bool],
    op: BooleanScalarOp2,
) {
    assert_eq!(left.len(), right.len());
    assert_eq!(left.len(), result.len());
    for i in 0..left.len() {
        result[i] = op(left[i], right[i]);
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
    let mem_chunks = len / NUM_FLOAT_LANES_32;
    println!("mem_chunks: {}", mem_chunks);
    unsafe {
        for i in 0..mem_chunks {
            let idx = i * NUM_FLOAT_LANES_32;
            let v_left = vld1q_f32(left.as_ptr().add(idx));
            let v_result = intrinsic_op(v_left);
            vst1q_f32(result.as_mut_ptr().add(idx), v_result);
        }
    }

    for i in (mem_chunks * NUM_FLOAT_LANES_32)..len {
        result[i] = scalar_op(left[i]);
    }
}

// TODO: Evaluate true correctness, used a suggestion from codex
// UPDATE: Codex suggestion was bad. Tweaked to use optionals instead
// since not all reductive operations have exactly the same purpose and methodology
// The intrinsic uses intrinsic types (duh) that are not easily castable
// to i32, which is what we need, but we need to lay this vector out in the
// vector registers, and handle this like a normal NEON call that we've implemented
// elsewhere.
pub fn signed_int_neon_1(
    left: &[i32],
    intrinsic_op: IntrinsicSignedInt,
    signed_int_scalar_op: SignedIntScalarVecOp,
    reduction_op: SignedIntReductionOp,
) -> i32 {
    let len = left.len();
    let mut result: Option<i32> = None;
    let mem_chunks = len / NUM_INT_LANES_32;
    unsafe {
        for i in 0..mem_chunks {
            let idx = i * NUM_INT_LANES_32;
            let v_left = vld1q_s32(left.as_ptr().add(idx));
            let intrinsic_result = intrinsic_op(v_left);
            // the reduction_op in this case acts as an accumualtor to keep track of ongoing results
            // TODO: there may be an allocation  bottleneck here, revisit.
            result = Some(match result {
                Some(existing_result) => reduction_op(existing_result, intrinsic_result),
                None => intrinsic_result,
            });
        }
    }
    // why so weird?
    // because condensing a vector into a single number is a reductive operation
    // and so any remainder in memory has to be handled by directly accessing the last elements
    // of the vector since they do not fit neatly into the vector registers
    if (mem_chunks * NUM_INT_LANES_32) < len {
        let remainder = &left[(mem_chunks * NUM_INT_LANES_32)..];
        let existing = match result.as_ref() {
            Some(existing_result) => Some(std::slice::from_ref(existing_result)),
            None => None,
        };
        // using the scalar op in this loop we can keep checking each value, left and right
        // using the scalar operation that should be the same for all reductive operations
        result = Some(signed_int_scalar_op(
            remainder,
            // result will keep mutating, and we need to be very careful and ensure
            // that we're only using references to result instead of totally allocating new
            // space for the result.
            existing,
        ));
    }

    // then if we are absolutely sure that we have no remainder, we can just return the result
    result.unwrap_or_else(|| signed_int_scalar_op(left, None))
}

// bools in rust are 8bits, thus u8 intrinsics
// also, rust bools only use the lowest order bit, thus any boolean in rust
// in binary is 0b00000001 true or 0b00000000 false
pub fn boolean_neon_1(
    left: &[bool],
    result: &mut [bool],
    intrinsic_op: IntrinsicBool,
    scalar_op: BooleanScalarOp,
) {
    assert_eq!(left.len(), result.len());
    let len = left.len();
    // follows intrinsic type defs
    // 16 lanes of 1 byte each in a 128 bit d-register
    // thus a vector of 5 booleans should result in a 128 bit wide vector of 16 lanes each, 1 byte per lane

    let mem_chunks = len / NUM_BOOL_LANES_8;
    // be extremely careful with this
    unsafe {
        let mask = vdupq_n_u8(BOOL_SIZE);
        for i in 0..mem_chunks {
            let idx = i * NUM_BOOL_LANES_8;
            let bin_left = left.as_ptr().add(idx) as *const u8;

            let v_left = vld1q_u8(bin_left);
            let v_result = intrinsic_op(v_left);
            let masked_result = vandq_u8(v_result, mask);
            vst1q_u8(result.as_mut_ptr().add(idx) as *mut u8, masked_result);
        }
    }

    for i in (mem_chunks * NUM_BOOL_LANES_8)..len {
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
    intrinsic_bool: IntrinsicInt2,
    scalar_op: ScalarOp2,
) {
    assert_eq!(left.len(), right.len());
    assert_eq!(left.len(), result.len());
    let len = left.len();
    let mem_chunks = len / NUM_FLOAT_LANES_32;
    unsafe {
        for i in 0..mem_chunks {
            let idx = i * NUM_FLOAT_LANES_32;
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
    for i in (mem_chunks * NUM_FLOAT_LANES_32)..len {
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
    let mem_chunks = len / NUM_FLOAT_LANES_32;

    // working at the CPU level for faster ops, hence unsafe is needed
    unsafe {
        for i in 0..mem_chunks {
            let idx = i * NUM_FLOAT_LANES_32;

            let v_left = vld1q_f32(left.as_ptr().add(idx));
            let v_right = vld1q_f32(right.as_ptr().add(idx));
            let v_result = intrinsic_op(v_left, v_right);
            vst1q_f32(result.as_mut_ptr().add(idx), v_result);
        }
    }

    for i in (mem_chunks * NUM_FLOAT_LANES_32)..len {
        result[i] = scalar_op(left[i], right[i]);
    }
}

pub fn boolean_neon_2(
    left: &[bool],
    right: &[bool],
    result: &mut [bool],
    intrinsic_op: IntrinsicBool2,
    scalar_op: BooleanScalarOp2,
) {
    assert_eq!(left.len(), result.len());
    assert_eq!(left.len(), right.len());
    let len = left.len();
    // follows intrinsic type defs
    // 16 lanes of 1 byte each in a 128 bit d-register
    // thus a vector of 5 booleans should result in a 128 bit wide vector of 16 lanes each, 1 byte per lane

    let mem_chunks = len / NUM_BOOL_LANES_8;
    // be extremely careful with this
    unsafe {
        let mask = vdupq_n_u8(BOOL_SIZE);
        for i in 0..mem_chunks {
            let idx = i * NUM_BOOL_LANES_8;
            let bin_left = left.as_ptr().add(idx) as *const u8;
            let bin_right = right.as_ptr().add(idx) as *const u8;

            let v_left = vld1q_u8(bin_left);
            let v_right = vld1q_u8(bin_right);
            let v_result = intrinsic_op(v_left, v_right);
            let masked_result = vandq_u8(v_result, mask);
            vst1q_u8(result.as_mut_ptr().add(idx) as *mut u8, masked_result);
        }
    }

    for i in (mem_chunks * NUM_BOOL_LANES_8)..len {
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
    let mem_chunks = len / NUM_FLOAT_LANES_32;

    // working at the CPU level for faster ops, hence unsafe is needed
    unsafe {
        for i in 0..mem_chunks {
            let idx = i * NUM_FLOAT_LANES_32;

            let v_left = vld1q_f32(left.as_ptr().add(idx));
            let v_middle = vld1q_f32(middle.as_ptr().add(idx));
            let v_right = vld1q_f32(right.as_ptr().add(idx));
            let v_result = intrinsic_op(v_left, v_middle, v_right);
            vst1q_f32(result.as_mut_ptr().add(idx), v_result);
        }
    }

    for i in (mem_chunks * NUM_FLOAT_LANES_32)..len {
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

pub fn signed_int_par_1(vector: &[i32], op: SignedIntVecOp) -> i32 {
    vector.par_chunks(PAR_CHUNK_SIZE).map(op).sum()
}

// TODO: literally just the same as f32 neon par but with bools
// fix this ASAP!!!!!
// quick hack, igore
pub fn boolean_par_1(left: &[bool], result: &mut [bool], op: BooleanVecOp) {
    assert_eq!(left.len(), result.len());
    result
        .par_chunks_mut(PAR_CHUNK_SIZE)
        .zip(left.par_chunks(PAR_CHUNK_SIZE))
        .for_each(|(result_chunk, left_chunk)| {
            op(left_chunk, result_chunk);
        });
}

// TODO: literally just the same as f32 neon par but with bools
// fix this ASAP!!!!!
// quick hack, igore
pub fn boolean_par_2(left: &[bool], right: &[bool], result: &mut [bool], op: BooleanVecOp2) {
    assert_eq!(left.len(), result.len());
    assert_eq!(left.len(), right.len());
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
