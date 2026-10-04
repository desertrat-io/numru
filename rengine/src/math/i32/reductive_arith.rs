use crate::data::array::SignedInt32Array;
use crate::matrix::ops::{Mode, PAR_CHUNK_SIZE, signed_int_neon_1, signed_int_par_1};
use rayon::prelude::*;
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// so the sum is a useful function, but we don't want to keep cloning vectors all over, so it's cheaper
// to just borrow
pub fn sum(vector: &SignedInt32Array, mode: Mode) -> i32 {
    let result: i32;
    match mode {
        Mode::Normal => result = reductive_sum_scalar_32(vector.slice(), None),
        Mode::Neon => {
            result = signed_int_neon_1(
                vector.slice(),
                vaddvq_s32,
                reductive_sum_scalar_32,
                sum_accumulator_32,
            )
        }
        Mode::ParNeon => {
            result = signed_int_par_1(
                vector.slice(),
                |vec| {
                    signed_int_neon_1(vec, vaddvq_s32, reductive_sum_scalar_32, sum_accumulator_32)
                },
                0,
                sum_accumulator_32,
            )
        }
    }
    result
}

pub fn min(vector: SignedInt32Array, mode: Mode) -> i32 {
    let result: i32;
    match mode {
        Mode::Normal => result = reductive_min_scalar_32(vector.slice(), None),
        Mode::Neon => {
            result = signed_int_neon_1(
                vector.slice(),
                vminvq_s32,
                reductive_min_scalar_32,
                min_accumulator_32,
            )
        }
        Mode::ParNeon => {
            result = signed_int_par_1(
                vector.slice(),
                |vec| {
                    signed_int_neon_1(vec, vminvq_s32, reductive_min_scalar_32, min_accumulator_32)
                },
                i32::MAX,
                min_accumulator_32,
            )
        }
    }
    result
}

pub fn max(vector: SignedInt32Array, mode: Mode) -> i32 {
    let result: i32;
    match mode {
        Mode::Normal => result = reductive_max_scalar_32(vector.slice(), None),
        Mode::Neon => {
            result = signed_int_neon_1(
                vector.slice(),
                vmaxvq_s32,
                reductive_max_scalar_32,
                max_accumulator_32,
            )
        }
        Mode::ParNeon => {
            result = signed_int_par_1(
                vector.slice(),
                |vec| {
                    signed_int_neon_1(vec, vmaxvq_s32, reductive_max_scalar_32, max_accumulator_32)
                },
                i32::MIN,
                max_accumulator_32,
            )
        }
    }
    result
}

pub fn mean(vector: SignedInt32Array, mode: Mode) -> i32 {
    // this is actually much simpler than before because we can just reuse sum and divide
    // at the end of processing
    if vector.len() == 0 {
        panic!("Divide by zero risk: vector length is 0")
    }
    // vector is consumed later on, set the length aside here
    // lengths beyond 32 bit pointers are not supported by this approach
    let len = vector.len() as i32;
    sum(&vector, mode) / len
}

// for i32 this WILL round the resulting sqrt towards floor
pub fn l2_norm(vector: SignedInt32Array, mode: Mode) -> i32 {
    match mode {
        Mode::Normal => L2norm::result(arith_reduce_scalar::<L2norm>(vector.slice())),
        Mode::Neon => {
            let result = arith_reduce_neon::<L2norm>(vector.slice());
            L2norm::neon_result(result.0, result.1)
        }
        Mode::ParNeon => {
            let result = arith_reduce_neon_par::<L2norm>(vector.slice());
            L2norm::neon_result(result.0, result.1)
        }
    }
}
fn reductive_sum_scalar_32(vector: &[i32], existing: Option<&[i32]>) -> i32 {
    vector.iter().sum::<i32>() + existing.unwrap_or_default().iter().sum::<i32>()
}

fn sum_accumulator_32(left: i32, right: i32) -> i32 {
    left + right
}

fn reductive_min_scalar_32(vector: &[i32], existing: Option<&[i32]>) -> i32 {
    let current = *vector.iter().min().unwrap();
    match existing {
        None => current,
        Some(existing) => current.min(*existing.iter().min().unwrap()),
    }
}

fn min_accumulator_32(left: i32, right: i32) -> i32 {
    left.min(right)
}

fn reductive_max_scalar_32(vector: &[i32], existing: Option<&[i32]>) -> i32 {
    let current = *vector.iter().max().unwrap();
    match existing {
        None => current,
        Some(existing) => current.max(*existing.iter().max().unwrap()),
    }
}

fn max_accumulator_32(left: i32, right: i32) -> i32 {
    left.max(right)
}

// reduction returns are wide on purpose to help with bounds on i32 (and u32 eventually)
trait ArithReductionOp {
    const NUM_LANES: usize;
    fn reduce(laned_chunk: int32x4_t) -> u64;

    fn scalar_reduce(vector: &[i32]) -> u64;

    fn result(accumulated_result: u64) -> i32;

    fn neon_result(accumulated_neon_result: u64, accumulated_result: u64) -> i32;
}
struct L2norm;
impl ArithReductionOp for L2norm {
    const NUM_LANES: usize = 4;
    #[inline(always)]
    fn reduce(laned_chunk: int32x4_t) -> u64 {
        unsafe {
            // square each chunk the naive way after expanding into i32
            let lower_64 = vmull_s32(vget_low_s32(laned_chunk), vget_low_s32(laned_chunk));
            let upper_64 = vmull_s32(vget_high_s32(laned_chunk), vget_high_s32(laned_chunk));
            let lower_sum_64 = vaddvq_s64(lower_64) as u64;
            let upper_sum_64 = vaddvq_s64(upper_64) as u64;

            lower_sum_64 + upper_sum_64
        }
    }

    #[inline(always)]
    fn scalar_reduce(vector: &[i32]) -> u64 {
        let mut sum_of_squares: u64 = 0;

        for val in vector {
            let value = *val as i64;
            sum_of_squares = sum_of_squares
                .checked_add((value * value) as u64)
                .expect("Scalar accumulator overflow, cannot fit in i32");
        }

        sum_of_squares
    }

    fn result(accumulated_result: u64) -> i32 {
        if accumulated_result <= (i32::MAX as u64).pow(2) {
            accumulated_result.isqrt() as i32
        } else {
            panic!("Overflow, cannot produce squareroot for signed 32 bit integer")
        }
    }

    fn neon_result(accumulated_neon_result: u64, accumulated_result: u64) -> i32 {
        let checked_total = accumulated_neon_result
            .checked_add(accumulated_result)
            .expect("L2 norm overflow, cannot produce squaerooot for signed 32 bit integer");
        if checked_total <= (i32::MAX as u64).pow(2) {
            checked_total.isqrt() as i32
        } else {
            panic!("Overflow, cannot produce squarerooot for signed 32 bit integer")
        }
    }
}

fn arith_reduce_scalar<T: ArithReductionOp>(vector: &[i32]) -> u64 {
    T::scalar_reduce(vector)
}

fn arith_reduce_neon<T: ArithReductionOp>(vector: &[i32]) -> (u64, u64) {
    let len = vector.len();
    let mut neon_accumulation: u64 = 0;
    let mut remaining_result: u64 = 0;
    let mem_chunks = len / T::NUM_LANES;
    for i in 0..mem_chunks {
        let idx = i * T::NUM_LANES;
        unsafe {
            let current_laned_chunk = vld1q_s32(vector.as_ptr().add(idx));
            neon_accumulation = neon_accumulation
                .checked_add(T::reduce(current_laned_chunk))
                .expect("NEON accumulator overflow, cannot fit in i32");
        }
    }
    if (mem_chunks * T::NUM_LANES) < len {
        let remainder = &vector[(mem_chunks * T::NUM_LANES)..];
        remaining_result = T::scalar_reduce(remainder);
    }

    (neon_accumulation, remaining_result)
}

fn arith_reduce_neon_par<T: ArithReductionOp>(vector: &[i32]) -> (u64, u64) {
    let (neon_result, scalar_result) = vector
        .par_chunks(PAR_CHUNK_SIZE)
        .map(arith_reduce_neon::<T>)
        .reduce(
            || (0, 0),
            |left, right| {
                (
                    left.0.checked_add(right.0).unwrap(),
                    left.1.checked_add(right.1).unwrap(),
                )
            },
        );
    (neon_result, scalar_result)
}
