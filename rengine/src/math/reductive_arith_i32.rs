use crate::data::array::SignedIntArray;

use crate::matrix::ops::{signed_int_neon_1, signed_int_par_1, Mode};
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Trait for reduction operations on signed integer arrays that perform selective reductions (track multiple variables)

pub fn sum(vector: SignedIntArray, mode: Mode) -> i32 {
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

pub fn min(vector: SignedIntArray, mode: Mode) -> i32 {
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

pub fn max(vector: SignedIntArray, mode: Mode) -> i32 {
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

pub fn mean(vector: SignedIntArray, mode: Mode) -> i32 {
    // this is actually much simpler than before because we can just reuse sum and divide
    // at the end of processing
    if vector.len() == 0 {
        panic!("Divide by zero risk: vector length is 0")
    }
    // vector is consumed later on, set the length aside here
    // lengths beyond 32 bit pointers are not supported by this approach
    let len = vector.len() as i32;
    sum(vector, mode) / len
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
