use crate::data::array::{Array, CONTIGUOUS_STRIDE, SignedF32Array};
use crate::data::matrix::F32Matrix;
use crate::matrix::ops::LinAlgMode;
use blas::{dsdot, sgemm};

pub fn transpose(matrix: F32Matrix, mode: LinAlgMode) -> F32Matrix {
    assert_ne!(matrix.cols(), usize::MAX, "Invalid matrix dimensions");
    if matrix.rows() == 0 || matrix.cols() == 0 {
        return F32Matrix::empty(matrix.rows(), matrix.cols());
    }
    match mode {
        LinAlgMode::Normal => transpose_scalar(matrix),
        LinAlgMode::Blas => transpose_blas(matrix),
        _ => todo!(),
    }
}

fn transpose_scalar(matrix: F32Matrix) -> F32Matrix {
    let mut result_matrix = F32Matrix::empty(matrix.rows(), matrix.cols());
    // rust shenanigans, might try to redo this later so we don't need a placeholder to keep
    // the value in memory
    let result = result_matrix.slice_mut();
    unsafe {
        for i in 0..matrix.rows() {
            let current_row = matrix.row_unchecked(i);
            for j in 0..matrix.cols() {
                result[j * matrix.rows() + i] = current_row[j];
            }
        }
    }
    // TODO: benchmark, this looks like it'll result in a memory spike until the function returns
    F32Matrix::new(
        matrix.rows(),
        matrix.cols(),
        Array::<f32>::new(result.to_vec()),
    )
}

fn transpose_blas(matrix: F32Matrix) -> F32Matrix {
    let cols = matrix.cols();
    let rows = matrix.rows();
    // TODO: move to base matrix struct definition
    let mut ident_contiguous_matrix = vec![0.0_f32; cols * cols];
    for i in 0..cols {
        ident_contiguous_matrix[i * cols + i] = 1.0;
    }

    let mut result = F32Matrix::empty(rows, cols);

    unsafe {
        sgemm(
            b'T',
            b'N',
            rows as i32,
            cols as i32,
            cols as i32,
            1.0,
            matrix.slice(),
            cols as i32,
            &ident_contiguous_matrix,
            cols as i32,
            0.0,
            result.slice_mut(),
            rows as i32,
        );
    }

    result
}

/// zero dimensional matrices are not supported in this version
pub fn mat_mul(left_matrix: F32Matrix, right_matrix: F32Matrix, mode: LinAlgMode) -> F32Matrix {
    let left_cols = left_matrix.cols();
    let right_cols = right_matrix.cols();
    let left_rows = left_matrix.rows();
    let right_rows = right_matrix.rows();
    assert_eq!(
        left_cols, right_rows,
        "Invalid matrix shape, LHS columns do not match RHS rows"
    );
    let result: F32Matrix = F32Matrix::empty(right_cols, left_rows);

    // if we have zero dim matrices, just return the empty matrix of that zero initialized to 0
    if left_rows == 0 || left_cols == 0 || right_cols == 0 {
        return result;
    }
    match mode {
        LinAlgMode::Normal => mat_mul_scalar(left_matrix, right_matrix, result),
        LinAlgMode::Blas => mat_mul_blas(left_matrix, right_matrix, result),
        _ => todo!(),
    }
}

fn mat_mul_scalar(
    left_matrix: F32Matrix,
    right_matrix: F32Matrix,
    mut result_matrix: F32Matrix,
) -> F32Matrix {
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
    left_matrix: F32Matrix,
    right_matrix: F32Matrix,
    mut result_matrix: F32Matrix,
) -> F32Matrix {
    #[cfg(feature = "blas-apple")]
    unsafe {
        sgemm(
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
pub fn dot(left_vector: SignedF32Array, right_vector: SignedF32Array, mode: LinAlgMode) -> f64 {
    assert_eq!(left_vector.len(), right_vector.len());
    let result;
    match mode {
        LinAlgMode::Normal => result = dot_scalar(left_vector.slice(), right_vector.slice()),
        LinAlgMode::Blas => result = dot_blas(left_vector.slice(), right_vector.slice()),
        _ => todo!(),
    };
    result
}

fn dot_scalar(left_vector: &[f32], right_vector: &[f32]) -> f64 {
    let mut result: f64 = 0.;

    for i in 0..left_vector.len() {
        result += left_vector[i] as f64 * right_vector[i] as f64;
    }

    result
}
fn dot_blas(left_vector: &[f32], right_vector: &[f32]) -> f64 {
    let mut sum = 0.0;
    let lim = i32::MAX as usize;
    // certain blas implementations return f64 for f32 operations, and I do believe that's how
    // apple's accelerate works? Not all implementations work this way
    #[cfg(feature = "blas-apple")]
    unsafe {
        // this will cleanly divide the vector into parts that fit inside the total length
        // no iterator overhead, just the chunks
        for (left_chunk, right_chunk) in left_vector.chunks(lim).zip(right_vector.chunks(lim)) {
            sum += dsdot(
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

// TODO: Benchmarking needed for parallel, not needed yet
fn dot_blas_par(left_vector: &[f32], right_vector: &[f32]) -> f64 {
    #[cfg(feature = "blas-apple")]
    todo!()
}
