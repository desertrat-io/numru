// TODO: Codex experimental test code, evaluate for usefulness and correctness.
#[cfg(test)]
mod tests {
    use crate::data::array::SignedIntArray;
    use crate::math::stats::VarianceType;
    use crate::math::stats_i32::{std, var};
    use crate::matrix::ops::Mode;

    fn assert_variances(values: &[i32], expected_sample: i32, expected_population: i32) {
        assert_eq!(
            var(
                SignedIntArray::new(values.to_vec()),
                Mode::Normal,
                VarianceType::Sample,
            ),
            expected_sample,
        );
        assert_eq!(
            var(
                SignedIntArray::new(values.to_vec()),
                Mode::Normal,
                VarianceType::Population,
            ),
            expected_population,
        );
    }

    fn assert_standard_deviations(values: &[i32], expected_sample: i32, expected_population: i32) {
        assert_eq!(
            std(
                SignedIntArray::new(values.to_vec()),
                Mode::Normal,
                VarianceType::Sample,
            ),
            expected_sample,
        );
        assert_eq!(
            std(
                SignedIntArray::new(values.to_vec()),
                Mode::Normal,
                VarianceType::Population,
            ),
            expected_population,
        );
    }

    #[test]
    fn variance_normal_empty_vector_returns_i32_min() {
        assert_variances(&[], i32::MIN, i32::MIN);
    }

    #[test]
    fn variance_normal_single_element_vector_returns_i32_min() {
        assert_variances(&[42], i32::MIN, i32::MIN);
    }

    #[test]
    fn variance_normal_two_values() {
        assert_variances(&[1, 3], 2, 1);
    }

    #[test]
    fn standard_deviation_normal_two_sorted_values() {
        assert_standard_deviations(&[1, 3], 1, 1);
    }

    #[test]
    fn variance_normal_three_ascending_values() {
        assert_variances(&[1, 2, 3], 1, 0);
    }

    #[test]
    fn variance_and_standard_deviation_normal_three_unsorted_values() {
        assert_variances(&[3, 1, 2], 1, 0);
        assert_standard_deviations(&[3, 1, 2], 1, 0);
    }

    #[test]
    fn variance_normal_negative_values() {
        assert_variances(&[-3, -1], 2, 1);
    }

    #[test]
    fn variance_normal_mixed_small_and_large_signed_values() {
        assert_variances(&[-100, -2, -1, 0, 1, 2, 100], 3_335, 2_858);
    }

    #[test]
    fn variance_normal_eleven_values() {
        assert_variances(&[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], 11, 10);
    }

    #[test]
    fn variance_normal_twenty_values() {
        assert_variances(
            &[
                -19, -9, -8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 19,
            ],
            68,
            64,
        );
    }

    #[test]
    fn variance_normal_identical_values() {
        assert_variances(&[7, 7, 7], 0, 0);
    }

    #[test]
    fn variance_and_standard_deviation_normal_large_sorted_vector() {
        let values: Vec<i32> = (0..100)
            .flat_map(|value| std::iter::repeat(value).take(100))
            .collect();

        assert_eq!(values.len(), 10_000);
        assert_variances(&values, 833, 833);
        assert_standard_deviations(&values, 29, 29);
    }

    #[test]
    fn variance_and_standard_deviation_normal_large_unsorted_vector() {
        let values: Vec<i32> = (0..100).cycle().take(10_000).collect();

        assert_eq!(values.len(), 10_000);
        assert_variances(&values, 833, 833);
        assert_standard_deviations(&values, 29, 29);
    }

    #[test]
    #[should_panic]
    fn variance_normal_panics_when_sum_overflows_i32() {
        let _ = var(
            SignedIntArray::new(vec![i32::MAX, i32::MAX]),
            Mode::Normal,
            VarianceType::Sample,
        );
    }

    #[test]
    #[should_panic]
    fn variance_normal_panics_when_squared_deviation_overflows_i32() {
        let _ = var(
            SignedIntArray::new(vec![i32::MAX, i32::MIN]),
            Mode::Normal,
            VarianceType::Population,
        );
    }
}
