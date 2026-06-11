use crate::data::array::IntArray;

use crate::matrix::ops::{signed_int_neon_1, signed_int_par_1, Mode};
use std::arch::aarch64::vaddvq_s32;

#[cfg(target_arch = "aarch64")]
pub fn reductive_sum(vector: IntArray, mode: Mode) -> i32 {
    let result: i32;
    match mode {
        Mode::Normal => result = reductive_sum_scalar_32(vector.slice()),
        Mode::Neon => {
            result = signed_int_neon_1(vector.slice(), vaddvq_s32, reductive_sum_scalar_32)
        }
        Mode::ParNeon => {
            result = signed_int_par_1(vector.slice(), |vec| {
                signed_int_neon_1(vec, vaddvq_s32, reductive_sum_scalar_32)
            })
        }
    }
    result
}

fn reductive_sum_scalar_32(vector: &[i32]) -> i32 {
    vector.iter().sum()
}
