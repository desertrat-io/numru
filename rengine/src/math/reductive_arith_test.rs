#[cfg(test)]
mod tests {
    use crate::data::array::SignedIntArray;
    use crate::math::reductive_arith::{max, mean, min, sum};
    use crate::matrix::ops::Mode;

    const EMPTY: &[i32] = &[];
    const THREE_ASC: &[i32] = &[1, 2, 3];

    const FOUR_ASC: &[i32] = &[1, 2, 3, 4];
    const FOUR_NEG: &[i32] = &[-1, -2, -3, -4];

    const FIVE_ASC: &[i32] = &[1, 2, 3, 4, 5];
    const FIVE_WITH_MAX: &[i32] = &[1, 2, 10, 3, 4];

    const SIX_WITH_MIN: &[i32] = &[1, 2, 3, 0, 4, 5];
    const SIX_WITH_MAX: &[i32] = &[1, 2, 10, 3, 4, 5];

    const SEVEN_ASC: &[i32] = &[1, 2, 3, 4, 5, 6, 7];

    const EIGHT_WITH_MIN: &[i32] = &[1, 2, 3, 0, 4, 5, 1, 10];
    const EIGHT_WITH_MAX: &[i32] = &[1, 2, 10, 3, 4, 5, 6, 7];

    fn signed_int_vector(values: &[i32]) -> SignedIntArray {
        SignedIntArray::new(values.to_vec())
    }

    #[test]
    fn test_reductive_scalar_sum_32() {
        let vector = signed_int_vector(THREE_ASC);
        let result = sum(vector, Mode::Normal);
        assert_eq!(result, 6);
        let vector = signed_int_vector(EMPTY);
        let result = sum(vector, Mode::Normal);
        assert_eq!(result, 0);
        let vector = signed_int_vector(FOUR_NEG);
        let result = sum(vector, Mode::Normal);
        assert_eq!(result, -10);
    }

    #[test]
    fn signed_int_neon_1_sums_full_vector() {
        let vector = signed_int_vector(FOUR_ASC);
        let result = sum(vector, Mode::Neon);
        assert_eq!(result, 10);
    }

    #[test]
    fn signed_int_neon_1_sums_full_vector_with_negative() {
        let vector = signed_int_vector(FOUR_NEG);
        let result = sum(vector, Mode::Neon);
        assert_eq!(result, -10);
    }

    #[test]
    fn signed_int_neon_1_sums_remainder() {
        let vector = signed_int_vector(FIVE_ASC);
        let result = sum(vector, Mode::Neon);
        assert_eq!(result, 15);
    }

    #[test]
    fn signed_int_par_1_sums_full_vector() {
        let vector = signed_int_vector(FOUR_ASC);
        let result = sum(vector, Mode::ParNeon);
        assert_eq!(result, 10);
    }

    #[test]
    fn signed_int_par_1_sums_remainder() {
        let vector = signed_int_vector(SEVEN_ASC);
        let result = sum(vector, Mode::ParNeon);
        assert_eq!(result, 28);
    }

    #[test]
    fn test_reductive_scalar_min() {
        let vector = signed_int_vector(THREE_ASC);
        let result = min(vector, Mode::Normal);
        assert_eq!(result, 1);
    }

    #[test]
    fn test_reductive_neon_min() {
        let vector = signed_int_vector(FOUR_ASC);
        let result = min(vector, Mode::Neon);
        assert_eq!(result, 1);
    }

    #[test]
    fn test_reductive_neon_min_with_remainder() {
        let vector = signed_int_vector(SIX_WITH_MIN);
        let result = min(vector, Mode::Neon);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_reductive_neon_par_min() {
        let vector = signed_int_vector(SIX_WITH_MIN);
        let result = min(vector, Mode::ParNeon);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_reductive_neon_par_min_with_remainder() {
        let vector = signed_int_vector(EIGHT_WITH_MIN);
        let result = min(vector, Mode::ParNeon);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_reductive_scalar_max() {
        let vector = signed_int_vector(THREE_ASC);
        let result = max(vector, Mode::Normal);
        assert_eq!(result, 3);
    }

    #[test]
    fn test_reduction_neon_max() {
        let vector = signed_int_vector(FIVE_WITH_MAX);
        let result = max(vector, Mode::Neon);
        assert_eq!(result, 10);
    }

    #[test]
    fn test_reduction_neon_max_with_remainder() {
        let vector = signed_int_vector(EIGHT_WITH_MAX);
        let result = max(vector, Mode::Neon);
        assert_eq!(result, 10);
    }

    #[test]
    fn test_reduction_neon_par_max() {
        let vector = signed_int_vector(SIX_WITH_MAX);
        let result = max(vector, Mode::ParNeon);
        assert_eq!(result, 10);
    }

    #[test]
    fn test_reduction_neon_par_max_with_remainder() {
        let vector = signed_int_vector(EIGHT_WITH_MAX);
        let result = max(vector, Mode::ParNeon);
        assert_eq!(result, 10);
    }

    #[test]
    fn test_reductive_scalar_mean() {
        let vector = signed_int_vector(THREE_ASC);
        let result = mean(vector, Mode::Normal);
        assert_eq!(result, 2);
    }

    #[test]
    #[should_panic(expected = "Divide by zero risk: vector length is 0")]
    fn test_reductive_scalar_mean_empty_vec() {
        let vector = signed_int_vector(EMPTY);
        let result = mean(vector, Mode::Normal);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_reductive_neon_mean_with_remainder() {
        let vector = signed_int_vector(FIVE_ASC);
        let result = mean(vector, Mode::Neon);
        assert_eq!(result, 3);
    }

    #[test]
    fn test_reductive_par_neon_mean_with_remainder() {
        let vector = signed_int_vector(SEVEN_ASC);
        let result = mean(vector, Mode::ParNeon);
        assert_eq!(result, 4);
    }

    #[test]
    #[should_panic(expected = "Divide by zero risk: vector length is 0")]
    fn test_reductive_mean_neon_mean_empty_vec() {
        let vector = signed_int_vector(EMPTY);
        let result = mean(vector, Mode::Neon);
        assert_eq!(result, 0);
    }

    #[test]
    #[should_panic(expected = "Divide by zero risk: vector length is 0")]
    fn test_reductive_mean_par_neon_mean_empty_vec() {
        let vector = signed_int_vector(EMPTY);
        let result = mean(vector, Mode::ParNeon);
        assert_eq!(result, 0);
    }
}
