#[cfg(test)]
mod tests {
    use crate::data::array::{Array, SignedF64Array};
    use crate::data::matrix::F64Matrix;
    use crate::math::f64::linalg::{dot, mat_mul};
    use crate::matrix::ops::Mode;
    use crate::matrix::ops::LinAlgMode;

    fn matrix(cols: usize, rows: usize, values: &[f64]) -> F64Matrix {
        F64Matrix::new(cols, rows, Array::new(values.to_vec()))
    }

    fn assert_matrix(matrix: &F64Matrix, expected_cols: usize, expected: &[&[f64]]) {
        assert_eq!(matrix.cols(), expected_cols);
        assert_eq!(matrix.rows(), expected.len());
        for (row, values) in expected.iter().enumerate() {
            assert_eq!(matrix.row(row), Some(*values));
        }
    }

    fn assert_mat_mul(mode: LinAlgMode) {
        let left = matrix(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let right = matrix(2, 3, &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);

        let result = mat_mul(left, right, mode);

        assert_matrix(&result, 2, &[&[58.0, 64.0], &[139.0, 154.0]]);
    }

    #[test]
    fn mat_mul_scalar_multiplies_rectangular_matrices() {
        assert_mat_mul(LinAlgMode::Normal);
    }

    #[test]
    fn mat_mul_blas_multiplies_rectangular_matrices() {
        assert_mat_mul(LinAlgMode::Blas);
    }

    #[test]
    fn mat_mul_scalar_handles_single_element_matrices() {
        let result = mat_mul(
            matrix(1, 1, &[3.0]),
            matrix(1, 1, &[-4.0]),
            LinAlgMode::Normal,
        );

        assert_matrix(&result, 1, &[&[-12.0]]);
    }

    #[test]
    fn mat_mul_blas_handles_single_element_matrices() {
        let result = mat_mul(
            matrix(1, 1, &[3.0]),
            matrix(1, 1, &[-4.0]),
            LinAlgMode::Blas,
        );

        assert_matrix(&result, 1, &[&[-12.0]]);
    }

    #[test]
    fn mat_mul_scalar_handles_zero_values() {
        let result = mat_mul(
            matrix(2, 2, &[0.0, 0.0, 0.0, 0.0]),
            matrix(2, 2, &[1.0, 2.0, 3.0, 4.0]),
            LinAlgMode::Normal,
        );

        assert_matrix(&result, 2, &[&[0.0, 0.0], &[0.0, 0.0]]);
    }

    #[test]
    #[should_panic(expected = "LHS columns do not match RHS rows")]
    fn mat_mul_rejects_incompatible_shapes() {
        mat_mul(
            matrix(3, 2, &[1.0; 6]),
            matrix(2, 2, &[1.0; 4]),
            LinAlgMode::Normal,
        );
    }

    #[test]
    #[should_panic]
    fn matrix_rejects_dimension_product_overflow() {
        F64Matrix::empty(usize::MAX, 2);
    }

    #[test]
    fn dot_multiplies_and_sums_contiguous_vectors() {
        let result: f64 = dot(
            SignedF64Array::new(vec![1.0, 2.0, 3.0]),
            SignedF64Array::new(vec![4.0, 5.0, 6.0]),
            Mode::Normal,
        );

        assert_eq!(result, 32.0_f64);
    }

    #[test]
    fn dot_handles_negative_values() {
        let result: f64 = dot(
            SignedF64Array::new(vec![-2.0, 3.0, -4.0]),
            SignedF64Array::new(vec![5.0, -6.0, 7.0]),
            Mode::Normal,
        );

        assert_eq!(result, -56.0_f64);
    }

    #[test]
    fn dot_returns_zero_when_one_vector_is_zero() {
        let result: f64 = dot(
            SignedF64Array::new(vec![0.0, 0.0, 0.0]),
            SignedF64Array::new(vec![4.0, -5.0, 6.0]),
            Mode::Normal,
        );

        assert_eq!(result, 0.0_f64);
    }
}
