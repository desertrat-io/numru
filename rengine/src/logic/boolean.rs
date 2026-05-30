//! # Boolean
//! Some vectors will contain boolean values only, so all vector functions here handle that.
//! A special boolean specific vector struct has been added to represent these special vectors.
//! Why not generics? This is part of the opinionated nature of the library. Yes, it sacrifices a tiny bit
//! of DRY but for the benefits of squeezing out optimizations and performance gains. This also means that
//! a change to one struct doesn't impact all others. Since the vector structs all live in one place it's still
//! pretty easy to maintain
//! Now, given the realities of booleans in rust, and the speed we're trying to work with, things get complicated
//! The intrinsic function for boolean not is bitwise, but logical not is the scalar operation and the expect return type
//! is a boolean. TODO: enhance the bit conversions to ensure peak performance

use crate::data::array::BoolArray;
use crate::matrix::ops::{
    boolean_binary_op_1, boolean_binary_op_2, boolean_neon_1, boolean_neon_2, boolean_par_1, boolean_par_2,
    Mode,
};
use std::arch::aarch64::{vandq_u8, vmvnq_u8};

#[cfg(target_arch = "aarch64")]
pub fn and(left_vector: BoolArray, right_vector: BoolArray, mode: Mode) -> BoolArray {
    let mut result = BoolArray::new(vec![false; left_vector.len()]);
    match mode {
        Mode::Normal => boolean_binary_op_2(
            left_vector.slice(),
            right_vector.slice(),
            &mut result.mut_slice(),
            and_scalar_32,
        ),
        Mode::Neon => boolean_neon_2(
            left_vector.slice(),
            right_vector.slice(),
            &mut result.mut_slice(),
            vandq_u8,
            and_scalar_32,
        ),
        Mode::ParNeon => boolean_par_2(
            left_vector.slice(),
            right_vector.slice(),
            result.mut_slice(),
            |left, right, res| {
                boolean_neon_2(left, right, res, vandq_u8, and_scalar_32);
            },
        ),
    }
    result
}

#[cfg(target_arch = "aarch64")]
pub fn not(input: BoolArray, mode: Mode) -> BoolArray {
    let mut result = BoolArray::new(vec![false; input.len()]);
    match mode {
        Mode::Normal => boolean_binary_op_1(input.slice(), &mut result.mut_slice(), not_scalar_32),
        Mode::Neon => boolean_neon_1(
            input.slice(),
            &mut result.mut_slice(),
            vmvnq_u8,
            not_scalar_32,
        ),
        Mode::ParNeon => boolean_par_1(input.slice(), result.mut_slice(), |left, res| {
            boolean_neon_1(left, res, vmvnq_u8, not_scalar_32);
        }),
    }
    result
}

fn and_scalar_32(left: bool, right: bool) -> bool {
    left && right
}

fn not_scalar_32(input: bool) -> bool {
    !input
}
