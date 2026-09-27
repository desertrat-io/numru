#[cfg(test)]
mod tests {
    use crate::data::array::Array;
    use crate::data::array::SignedF32Array;
    use crate::data::matrix::F32Matrix;
    use crate::math::f32::linalg::dot;
    use crate::math::f32::linalg::mat_mul;
    use crate::math::f32::linalg::transpose;
    use crate::matrix::ops::LinAlgMode;

    fn matrix(cols: usize, rows: usize, values: &[f32]) -> F32Matrix {
        F32Matrix::new(cols, rows, Array::new(values.to_vec()))
    }

    fn assert_matrix(matrix: &F32Matrix, expected_cols: usize, expected: &[&[f32]]) {
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
    fn mat_mul_scalar_handles_a_row_vector_times_a_column_vector() {
        let result = mat_mul(
            matrix(3, 1, &[1.5, -2.0, 4.0]),
            matrix(1, 3, &[2.0, -3.0, 0.5]),
            LinAlgMode::Normal,
        );

        assert_matrix(&result, 1, &[&[11.0]]);
    }

    #[test]
    fn mat_mul_blas_handles_a_row_vector_times_a_column_vector() {
        let result = mat_mul(
            matrix(3, 1, &[1.5, -2.0, 4.0]),
            matrix(1, 3, &[2.0, -3.0, 0.5]),
            LinAlgMode::Blas,
        );

        assert_matrix(&result, 1, &[&[11.0]]);
    }

    #[test]
    fn mat_mul_scalar_handles_zero_inner_dimension() {
        let result = mat_mul(matrix(0, 2, &[]), matrix(3, 0, &[]), LinAlgMode::Normal);

        assert_matrix(&result, 3, &[&[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0]]);
    }

    #[test]
    fn mat_mul_blas_handles_zero_inner_dimension() {
        let result = mat_mul(matrix(0, 2, &[]), matrix(3, 0, &[]), LinAlgMode::Blas);

        assert_matrix(&result, 3, &[&[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0]]);
    }

    #[test]
    fn mat_mul_scalar_and_blas_agree_for_fractional_and_negative_values() {
        let left = matrix(
            4,
            3,
            &[
                1.25, -2.0, 0.5, 3.0, -4.5, 2.25, 6.0, 0.0, -1.5, 2.0, 3.5, -5.0,
            ],
        );
        let right = matrix(2, 4, &[-1.0, 2.5, 3.0, -4.0, 0.25, -2.0, 5.5, 1.0]);

        let scalar = mat_mul(left.clone(), right.clone(), LinAlgMode::Normal);
        let blas = mat_mul(left, right, LinAlgMode::Blas);

        assert_eq!(scalar.cols(), blas.cols());
        assert_eq!(scalar.rows(), blas.rows());
        for row in 0..scalar.rows() {
            for col in 0..scalar.cols() {
                assert_eq!(scalar.get_or_else(col, row), blas.get_or_else(col, row));
            }
        }
    }

    #[test]
    #[should_panic]
    fn mat_mul_rejects_par_blas_mode_until_implemented() {
        mat_mul(
            matrix(1, 1, &[1.0]),
            matrix(1, 1, &[1.0]),
            LinAlgMode::ParBlas,
        );
    }

    #[test]
    fn transpose_scalar_transposes_a_rectangular_matrix() {
        let result = transpose(
            matrix(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            LinAlgMode::Normal,
        );

        assert_matrix(&result, 2, &[&[1.0, 4.0], &[2.0, 5.0], &[3.0, 6.0]]);
    }

    #[test]
    fn transpose_blas_transposes_a_rectangular_matrix() {
        let result = transpose(
            matrix(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            LinAlgMode::Blas,
        );

        assert_matrix(&result, 2, &[&[1.0, 4.0], &[2.0, 5.0], &[3.0, 6.0]]);
    }

    #[test]
    fn transpose_scalar_transposes_a_square_matrix() {
        let result = transpose(matrix(2, 2, &[1.0, 2.0, 3.0, 4.0]), LinAlgMode::Normal);

        assert_matrix(&result, 2, &[&[1.0, 3.0], &[2.0, 4.0]]);
    }

    #[test]
    fn transpose_blas_transposes_a_square_matrix() {
        let result = transpose(matrix(2, 2, &[1.0, 2.0, 3.0, 4.0]), LinAlgMode::Blas);

        assert_matrix(&result, 2, &[&[1.0, 3.0], &[2.0, 4.0]]);
    }

    #[test]
    fn transpose_scalar_handles_zero_rows() {
        let result = transpose(matrix(3, 0, &[]), LinAlgMode::Normal);

        assert_matrix(&result, 0, &[&[], &[], &[]]);
    }

    #[test]
    fn transpose_scalar_handles_zero_columns() {
        let result = transpose(matrix(0, 3, &[]), LinAlgMode::Normal);

        assert_matrix(&result, 3, &[]);
    }

    #[test]
    fn transpose_blas_handles_zero_rows() {
        let result = transpose(matrix(3, 0, &[]), LinAlgMode::Blas);

        assert_matrix(&result, 0, &[&[], &[], &[]]);
    }

    #[test]
    fn transpose_blas_handles_zero_columns() {
        let result = transpose(matrix(0, 3, &[]), LinAlgMode::Blas);

        assert_matrix(&result, 3, &[]);
    }

    #[test]
    #[should_panic]
    fn transpose_rejects_par_blas_mode_until_implemented() {
        transpose(matrix(1, 1, &[1.0]), LinAlgMode::ParBlas);
    }

    #[test]
    fn transpose_scalar_handles_large_matrix_with_large_values() {
        let cols = 257;
        let rows = 263;
        let values: Vec<f32> = (0..cols * rows)
            .map(|index| match index % 4 {
                0 => 1.0e30,
                1 => -2.0e30,
                2 => 3.0e29,
                _ => -4.0e29,
            })
            .collect();
        let input = matrix(cols, rows, &values);

        let result = transpose(input, LinAlgMode::Normal);

        assert_eq!(result.cols(), rows);
        assert_eq!(result.rows(), cols);
        for row in 0..rows {
            for col in 0..cols {
                assert_eq!(
                    result.get_or_else(row, col),
                    Some(&values[row * cols + col]),
                    "incorrect transposed value at row {col}, column {row}"
                );
            }
        }
    }

    #[test]
    fn transpose_blas_handles_large_matrix_with_large_values() {
        let cols = 257;
        let rows = 263;
        let values: Vec<f32> = (0..cols * rows)
            .map(|index| match index % 4 {
                0 => 1.0e30,
                1 => -2.0e30,
                2 => 3.0e29,
                _ => -4.0e29,
            })
            .collect();
        let input = matrix(cols, rows, &values);

        let result = transpose(input, LinAlgMode::Blas);

        assert_eq!(result.cols(), rows);
        assert_eq!(result.rows(), cols);
        for row in 0..rows {
            for col in 0..cols {
                assert_eq!(
                    result.get_or_else(row, col),
                    Some(&values[row * cols + col]),
                    "incorrect transposed value at row {col}, column {row}"
                );
            }
        }
    }

    #[test]
    #[should_panic = "attempt to multiply with overflow"]
    fn transpose_scalar_rejects_maximum_column_dimension() {
        let input = F32Matrix::empty(usize::MAX, 2);

        transpose(input, LinAlgMode::Normal);
    }

    #[test]
    #[should_panic = "attempt to multiply with overflow"]
    fn transpose_blas_rejects_maximum_column_dimension() {
        let input = F32Matrix::empty(usize::MAX, 2);

        transpose(input, LinAlgMode::Blas);
    }

    #[test]
    #[should_panic]
    fn transpose_scalar_rejects_maximum_row_dimension() {
        let input = F32Matrix::empty(2, usize::MAX);

        transpose(input, LinAlgMode::Normal);
    }

    #[test]
    #[should_panic]
    fn transpose_blas_rejects_maximum_row_dimension() {
        let input = F32Matrix::empty(2, usize::MAX);

        transpose(input, LinAlgMode::Blas);
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
        F32Matrix::empty(usize::MAX, 2);
    }

    #[test]
    fn dot_scalar_multiplies_and_sums_vectors() {
        let result = dot(
            SignedF32Array::new(vec![1.0, 2.0, 3.0]),
            SignedF32Array::new(vec![4.0, 5.0, 6.0]),
            LinAlgMode::Normal,
        );

        assert_eq!(result, 32.0_f64);
    }

    #[test]
    fn dot_scalar_handles_single_element_vectors() {
        let result = dot(
            SignedF32Array::new(vec![-2.5]),
            SignedF32Array::new(vec![4.0]),
            LinAlgMode::Normal,
        );

        assert_eq!(result, -10.0_f64);
    }

    #[test]
    fn dot_scalar_handles_mixed_signs_and_fractions() {
        let result = dot(
            SignedF32Array::new(vec![-2.0, 0.5, -4.0]),
            SignedF32Array::new(vec![-3.0, 1.5, 2.0]),
            LinAlgMode::Normal,
        );

        assert_eq!(result, -1.25_f64);
    }

    #[test]
    fn dot_scalar_returns_zero_for_empty_vectors() {
        let result = dot(
            SignedF32Array::new(vec![]),
            SignedF32Array::new(vec![]),
            LinAlgMode::Normal,
        );

        assert_eq!(result, 0.0_f64);
    }

    #[test]
    fn dot_scalar_returns_zero_when_one_vector_is_zero() {
        let result = dot(
            SignedF32Array::new(vec![0.0, 0.0, 0.0]),
            SignedF32Array::new(vec![4.0, -5.0, 6.0]),
            LinAlgMode::Normal,
        );

        assert_eq!(result, 0.0_f64);
    }

    #[test]
    fn dot_scalar_accumulates_in_f64() {
        let result = dot(
            SignedF32Array::new(vec![16_777_216.0, 1.0, -16_777_216.0]),
            SignedF32Array::new(vec![1.0, 1.0, 1.0]),
            LinAlgMode::Normal,
        );

        assert_eq!(result, 1.0_f64);
    }

    #[test]
    fn dot_scalar_multiplies_in_f64() {
        let result = dot(
            SignedF32Array::new(vec![4097.0]),
            SignedF32Array::new(vec![4097.0]),
            LinAlgMode::Normal,
        );

        assert_eq!(result, 16_785_409.0_f64);
    }

    #[test]
    fn dot_scalar_avoids_f32_product_overflow() {
        let result = dot(
            SignedF32Array::new(vec![f32::MAX]),
            SignedF32Array::new(vec![2.0]),
            LinAlgMode::Normal,
        );

        assert_eq!(result, f64::from(f32::MAX) * 2.0);
    }

    #[test]
    fn dot_scalar_handles_long_cancellation_without_f32_accumulation() {
        let mut left = Vec::with_capacity(1025);
        let mut right = Vec::with_capacity(1025);
        for index in 0..1024 {
            left.push(if index % 2 == 0 {
                16_777_216.0
            } else {
                -16_777_216.0
            });
            right.push(1.0);
        }
        left.push(3.0);
        right.push(1.0);

        assert_eq!(
            dot(
                SignedF32Array::new(left),
                SignedF32Array::new(right),
                LinAlgMode::Normal
            ),
            3.0
        );
    }

    #[test]
    fn dot_scalar_propagates_special_values() {
        assert!(
            dot(
                SignedF32Array::new(vec![f32::NAN]),
                SignedF32Array::new(vec![1.0]),
                LinAlgMode::Normal,
            )
            .is_nan()
        );
        assert_eq!(
            dot(
                SignedF32Array::new(vec![f32::INFINITY]),
                SignedF32Array::new(vec![2.0]),
                LinAlgMode::Normal,
            ),
            f64::INFINITY
        );
    }

    #[test]
    #[should_panic(expected = "assertion `left == right` failed")]
    fn dot_scalar_rejects_a_shorter_right_vector() {
        dot(
            SignedF32Array::new(vec![1.0, 2.0]),
            SignedF32Array::new(vec![1.0]),
            LinAlgMode::Normal,
        );
    }

    #[test]
    #[should_panic(expected = "assertion `left == right` failed")]
    fn dot_scalar_rejects_a_longer_right_vector() {
        dot(
            SignedF32Array::new(vec![1.0]),
            SignedF32Array::new(vec![1.0, 2.0]),
            LinAlgMode::Normal,
        );
    }

    #[test]
    fn dot_blas_multiplies_and_sums_contiguous_vectors() {
        let result = dot(
            SignedF32Array::new(vec![1.0, 2.0, 3.0]),
            SignedF32Array::new(vec![4.0, 5.0, 6.0]),
            LinAlgMode::Blas,
        );

        assert_eq!(result, 32.0_f64);
    }

    #[test]
    fn dot_blas_accumulates_f32_inputs_in_f64() {
        let result = dot(
            SignedF32Array::new(vec![16_777_216.0, 1.0]),
            SignedF32Array::new(vec![1.0, 1.0]),
            LinAlgMode::Blas,
        );

        assert_eq!(result, 16_777_217.0_f64);
    }

    #[test]
    fn dot_blas_returns_zero_for_empty_vectors() {
        let result = dot(
            SignedF32Array::new(vec![]),
            SignedF32Array::new(vec![]),
            LinAlgMode::Blas,
        );

        assert_eq!(result, 0.0_f64);
    }

    #[test]
    #[should_panic]
    fn dot_blas_rejects_vectors_with_different_lengths() {
        dot(
            SignedF32Array::new(vec![1.0]),
            SignedF32Array::new(vec![1.0, 2.0]),
            LinAlgMode::Blas,
        );
    }

    #[test]
    fn dot_blas_propagates_special_values() {
        assert!(
            dot(
                SignedF32Array::new(vec![f32::NAN]),
                SignedF32Array::new(vec![1.0]),
                LinAlgMode::Blas,
            )
            .is_nan()
        );
        assert_eq!(
            dot(
                SignedF32Array::new(vec![f32::INFINITY]),
                SignedF32Array::new(vec![2.0]),
                LinAlgMode::Blas,
            ),
            f64::INFINITY
        );
    }

    #[test]
    #[should_panic]
    fn dot_rejects_par_blas_mode_until_implemented() {
        dot(
            SignedF32Array::new(vec![1.0]),
            SignedF32Array::new(vec![1.0]),
            LinAlgMode::ParBlas,
        );
    }
}
