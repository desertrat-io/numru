#[cfg(test)]
mod tests {
    use crate::data::array::SignedF32Array;
    use crate::math::f32::linalg::dot;
    use crate::matrix::ops::LinAlgMode;

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
}
