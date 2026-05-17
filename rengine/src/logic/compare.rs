use crate::data::array::Array;
use crate::matrix::ops::*;
use std::arch::aarch64::{vceqq_f32, vcgtq_f32};
///
/// #Compare operations
/// All compare operations still return f32, even though one would think they should return bool.
/// In order to be consistent with the opinionated api, we will only send back 0 or 1 for truthfulness
/// however defining a constant for true and false is fine, but not currently needed
///

pub fn gt(left: Array, right: Array, mode: Mode) -> Array {
    assert_eq!(left.len(), right.len());
    let mut result = Array::zero_padded(left.len());
    let left_slice = left.slice();
    let right_slice = right.slice();
    match mode {
        Mode::Normal => binary_op_2(left_slice, right_slice, result.mut_slice(), gt_scalar_32),
        Mode::Neon => neon_bool_2(
            left_slice,
            right_slice,
            result.mut_slice(),
            // vceqq does a bitwise compare, not over the entire value
            vcgtq_f32,
            gt_scalar_32,
        ),
        Mode::ParNeon => par_2(
            left_slice,
            right_slice,
            result.mut_slice(),
            |left, right, res| {
                neon_bool_2(left, right, res, vcgtq_f32, gt_scalar_32);
            },
        ),
    }
    result
}

fn gt_scalar_32(left_element: f32, right_element: f32) -> f32 {
    // TODO: do some analysis here to see if this is faster than the naive implementation
    (left_element > right_element) as u8 as f32
}

pub fn eq(left: Array, right: Array, mode: Mode) -> Array {
    assert_eq!(left.len(), right.len());
    let mut result = Array::zero_padded(left.len());
    match mode {
        Mode::Normal => binary_op_2(
            left.slice(),
            right.slice(),
            result.mut_slice(),
            eq_scalar_32,
        ),
        Mode::Neon => {
            neon_bool_2(
                left.slice(),
                right.slice(),
                result.mut_slice(),
                // vceqq does a bitwise compare, not over the entire value
                vceqq_f32,
                eq_scalar_32,
            )
        }
        Mode::ParNeon => par_2(
            left.slice(),
            right.slice(),
            result.mut_slice(),
            |left, right, res| {
                neon_bool_2(left, right, res, vceqq_f32, eq_scalar_32);
            },
        ),
    }
    result
}

pub fn eq_scalar_32(left_element: f32, right_element: f32) -> f32 {
    (left_element == right_element) as u8 as f32
}
