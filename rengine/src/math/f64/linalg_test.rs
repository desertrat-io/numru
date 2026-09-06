#[cfg(test)]
mod tests {
    use crate::data::array::SignedF64Array;
    use crate::math::f64::linalg::dot;
    use crate::matrix::ops::Mode;

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
