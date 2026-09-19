use crate::data::array::{CONTIGUOUS_STRIDE, SignedF64Array};
use crate::data::matrix::F64Matrix;
use crate::matrix::ops::{LinAlgMode, Mode};
use blas::{ddot, dgemm};

pub fn mat_mul(left_matrix: F64Matrix, right_matrix: F64Matrix, mode: LinAlgMode) -> F64Matrix {
    assert_eq!(
        left_matrix.cols(),
        right_matrix.rows(),
        "Invalid matrix shape, LHS columns do not match RHS rows"
    );
    let result: F64Matrix = F64Matrix::empty(left_matrix.rows(), right_matrix.cols());
    match mode {
        LinAlgMode::Normal => mat_mul_scalar(left_matrix, right_matrix, result),
        LinAlgMode::Blas => mat_mul_blas(left_matrix, right_matrix, result),
        _ => todo!(),
    }
}

fn mat_mul_scalar(
    left_matrix: F64Matrix,
    right_matrix: F64Matrix,
    mut result_matrix: F64Matrix,
) -> F64Matrix {
    // don't need to check the dimensions, it's pre-validated by the time it gets here

    // pull out each row?
    for i in 0..left_matrix.rows() {
        for k in 0..left_matrix.cols() {
            unsafe {
                let current_lhs_col_val = left_matrix.row_unchecked(i)[k];
                let right_row = right_matrix.row_unchecked(k);
                let result_row = result_matrix.row_unchecked_mut(i);

                for j in 0..right_matrix.cols() {
                    result_row[j] += current_lhs_col_val * right_row[j];
                }
            }
        }
    }

    result_matrix
}

fn mat_mul_blas(
    left_matrix: F64Matrix,
    right_matrix: F64Matrix,
    mut result_matrix: F64Matrix,
) -> F64Matrix {
    #[cfg(feature = "blas-apple")]
    unsafe {
        dgemm(
            b'N',
            b'N',
            right_matrix.cols() as i32,
            left_matrix.rows() as i32,
            left_matrix.cols() as i32,
            1.0,
            right_matrix.slice(),
            right_matrix.cols() as i32,
            left_matrix.slice(),
            left_matrix.cols() as i32,
            0.0,
            result_matrix.slice_mut(),
            right_matrix.cols() as i32,
        )
    }
    result_matrix
}

pub fn dot(left_vector: SignedF64Array, right_vector: SignedF64Array, mode: Mode) -> f64 {
    assert_eq!(left_vector.len(), right_vector.len());
    let result;
    match mode {
        Mode::Normal => result = dot_blas(left_vector.slice(), right_vector.slice()),
        _ => todo!(),
    };
    result
}

fn dot_blas(left_vector: &[f64], right_vector: &[f64]) -> f64 {
    let mut sum = 0.0;
    let lim = i32::MAX as usize;
    // certain blas implementations return f64 for f32 operations, and I do believe that's how
    // apple's accelerate works? Not all implementations work this way
    #[cfg(feature = "blas-apple")]
    unsafe {
        // this will cleanly divide the vector into parts that fit inside the total length
        // no iterator overhead, just the chunks
        for (left_chunk, right_chunk) in left_vector.chunks(lim).zip(right_vector.chunks(lim)) {
            sum += ddot(
                left_chunk.len() as i32,
                left_chunk,
                CONTIGUOUS_STRIDE,
                right_chunk,
                CONTIGUOUS_STRIDE,
            );
        }
    };
    sum
}
