// TODO: Codex experimental test code, evaluate for usefulness and correctness
#[cfg(test)]
mod tests {
    use crate::data::array::SignedIntArray;
    use crate::math::reductive_arg_i32::argmin;
    use crate::math::reductive_arith_i32::{max, mean, min, sum};
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

    const SINGLE_I32_MAX: &[i32] = &[i32::MAX];
    const I32_MAX_WITH_SMALLER_VALUE: &[i32] = &[i32::MAX, 0];
    const I32_MIN_AT_MIDDLE: &[i32] = &[7, i32::MIN, -9, 0];
    const DUPLICATE_MIN: &[i32] = &[5, -3, 2, -3];

    const PAR_CHUNK_SIZE: usize = 4096;

    fn range_vector(len: usize) -> SignedIntArray {
        SignedIntArray::new((0..len as i32).collect())
    }
    fn signed_int_vector(values: &[i32]) -> SignedIntArray {
        SignedIntArray::new(values.to_vec())
    }

    #[test]
    fn reductive_scalar_sum_32() {
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
    fn signed_int_par_1_sums_empty_vector() {
        let vector = signed_int_vector(EMPTY);
        let result = sum(vector, Mode::ParNeon);
        assert_eq!(result, 0);
    }

    #[test]
    fn signed_int_par_1_sums_remainder() {
        let vector = signed_int_vector(SEVEN_ASC);
        let result = sum(vector, Mode::ParNeon);
        assert_eq!(result, 28);
    }

    #[test]
    fn signed_int_par_1_sums_exact_parallel_chunk() {
        let vector = range_vector(PAR_CHUNK_SIZE);
        let result = sum(vector, Mode::ParNeon);
        assert_eq!(result, (0..PAR_CHUNK_SIZE as i32).sum::<i32>());
    }

    #[test]
    fn signed_int_par_1_sums_parallel_chunk_with_remainder() {
        let vector = range_vector(PAR_CHUNK_SIZE + 1);
        let result = sum(vector, Mode::ParNeon);
        assert_eq!(result, (0..=PAR_CHUNK_SIZE as i32).sum::<i32>());
    }

    #[test]
    fn signed_int_par_1_sums_multiple_parallel_chunks() {
        let vector = range_vector(PAR_CHUNK_SIZE * 2 + 3);
        let result = sum(vector, Mode::ParNeon);
        assert_eq!(result, (0..(PAR_CHUNK_SIZE * 2 + 3) as i32).sum::<i32>());
    }

    #[test]
    fn reductive_scalar_min() {
        let vector = signed_int_vector(THREE_ASC);
        let result = min(vector, Mode::Normal);
        assert_eq!(result, 1);
    }

    #[test]
    #[should_panic]
    fn reductive_scalar_min_empty_vec_panics() {
        let vector = signed_int_vector(EMPTY);
        min(vector, Mode::Normal);
    }

    #[test]
    #[should_panic]
    fn reductive_neon_min_empty_vec_panics() {
        let vector = signed_int_vector(EMPTY);
        min(vector, Mode::Neon);
    }

    #[test]
    fn reductive_par_neon_min_empty_vec_returns_identity() {
        let vector = signed_int_vector(EMPTY);
        assert_eq!(min(vector, Mode::ParNeon), i32::MAX);
    }

    #[test]
    fn reductive_neon_par_min_across_parallel_chunks() {
        let mut values: Vec<i32> = (0..(PAR_CHUNK_SIZE * 2 + 1) as i32).collect();
        values[PAR_CHUNK_SIZE + 7] = -99;

        let vector = signed_int_vector(&values);
        assert_eq!(min(vector, Mode::ParNeon), -99);
    }

    #[test]
    fn reductive_scalar_argmin() {
        let vector = signed_int_vector(SIX_WITH_MIN);
        let result = argmin(vector, Mode::Normal);
        assert_eq!(result, 3);
    }

    #[test]
    fn reductive_scalar_argmin_single_i32_max() {
        let vector = signed_int_vector(SINGLE_I32_MAX);
        let result = argmin(vector, Mode::Normal);
        assert_eq!(result, 0);
    }

    #[test]
    fn reductive_scalar_argmin_i32_max_with_smaller_value() {
        let vector = signed_int_vector(I32_MAX_WITH_SMALLER_VALUE);
        let result = argmin(vector, Mode::Normal);
        assert_eq!(result, 1);
    }

    #[test]
    fn reductive_scalar_argmin_i32_min() {
        let vector = signed_int_vector(I32_MIN_AT_MIDDLE);
        let result = argmin(vector, Mode::Normal);
        assert_eq!(result, 1);
    }

    #[test]
    fn reductive_scalar_argmin_returns_first_duplicate_min() {
        let vector = signed_int_vector(DUPLICATE_MIN);
        let result = argmin(vector, Mode::Normal);
        assert_eq!(result, 1);
    }

    #[test]
    fn reductive_neon_min() {
        let vector = signed_int_vector(FOUR_ASC);
        let result = min(vector, Mode::Neon);
        assert_eq!(result, 1);
    }

    #[test]
    fn reductive_neon_min_with_remainder() {
        let vector = signed_int_vector(SIX_WITH_MIN);
        let result = min(vector, Mode::Neon);
        assert_eq!(result, 0);
    }

    #[test]
    fn reductive_neon_par_min() {
        let vector = signed_int_vector(SIX_WITH_MIN);
        let result = min(vector, Mode::ParNeon);
        assert_eq!(result, 0);
    }

    #[test]
    fn reductive_neon_par_min_with_remainder() {
        let vector = signed_int_vector(EIGHT_WITH_MIN);
        let result = min(vector, Mode::ParNeon);
        assert_eq!(result, 0);
    }

    #[test]
    fn reductive_neon_par_min_in_final_parallel_remainder() {
        let mut values: Vec<i32> = (0..(PAR_CHUNK_SIZE * 2 + 3) as i32).collect();
        values[PAR_CHUNK_SIZE * 2 + 2] = -99;

        let vector = signed_int_vector(&values);
        assert_eq!(min(vector, Mode::ParNeon), -99);
    }

    #[test]
    fn reductive_scalar_max() {
        let vector = signed_int_vector(THREE_ASC);
        let result = max(vector, Mode::Normal);
        assert_eq!(result, 3);
    }

    #[test]
    #[should_panic]
    fn reductive_scalar_max_empty_vec_panics() {
        let vector = signed_int_vector(EMPTY);
        max(vector, Mode::Normal);
    }

    #[test]
    fn reduction_neon_max() {
        let vector = signed_int_vector(FIVE_WITH_MAX);
        let result = max(vector, Mode::Neon);
        assert_eq!(result, 10);
    }

    #[test]
    #[should_panic]
    fn reductive_neon_max_empty_vec_panics() {
        let vector = signed_int_vector(EMPTY);
        max(vector, Mode::Neon);
    }

    #[test]
    fn reduction_neon_max_with_remainder() {
        let vector = signed_int_vector(EIGHT_WITH_MAX);
        let result = max(vector, Mode::Neon);
        assert_eq!(result, 10);
    }

    #[test]
    fn reduction_neon_par_max() {
        let vector = signed_int_vector(SIX_WITH_MAX);
        let result = max(vector, Mode::ParNeon);
        assert_eq!(result, 10);
    }

    #[test]
    fn reduction_neon_par_max_with_remainder() {
        let vector = signed_int_vector(EIGHT_WITH_MAX);
        let result = max(vector, Mode::ParNeon);
        assert_eq!(result, 10);
    }

    #[test]
    fn reductive_par_neon_max_empty_vec_returns_identity() {
        let vector = signed_int_vector(EMPTY);
        assert_eq!(max(vector, Mode::ParNeon), i32::MIN);
    }

    #[test]
    fn reduction_neon_par_max_across_parallel_chunks() {
        let mut values: Vec<i32> = vec![0; PAR_CHUNK_SIZE * 2 + 1];
        values[PAR_CHUNK_SIZE + 7] = 99;

        let vector = signed_int_vector(&values);
        assert_eq!(max(vector, Mode::ParNeon), 99);
    }

    #[test]
    fn reduction_neon_par_max_in_final_parallel_remainder() {
        let mut values: Vec<i32> = vec![0; PAR_CHUNK_SIZE * 2 + 3];
        values[PAR_CHUNK_SIZE * 2 + 2] = 99;

        let vector = signed_int_vector(&values);
        assert_eq!(max(vector, Mode::ParNeon), 99);
    }

    #[test]
    fn reductive_scalar_mean() {
        let vector = signed_int_vector(THREE_ASC);
        let result = mean(vector, Mode::Normal);
        assert_eq!(result, 2);
    }

    #[test]
    #[should_panic(expected = "Divide by zero risk: vector length is 0")]
    fn reductive_scalar_mean_empty_vec() {
        let vector = signed_int_vector(EMPTY);
        let result = mean(vector, Mode::Normal);
        assert_eq!(result, 0);
    }

    #[test]
    fn reductive_neon_mean_with_remainder() {
        let vector = signed_int_vector(FIVE_ASC);
        let result = mean(vector, Mode::Neon);
        assert_eq!(result, 3);
    }

    #[test]
    fn reductive_par_neon_mean_with_remainder() {
        let vector = signed_int_vector(SEVEN_ASC);
        let result = mean(vector, Mode::ParNeon);
        assert_eq!(result, 4);
    }

    #[test]
    #[should_panic(expected = "Divide by zero risk: vector length is 0")]
    fn reductive_mean_neon_mean_empty_vec() {
        let vector = signed_int_vector(EMPTY);
        let result = mean(vector, Mode::Neon);
        assert_eq!(result, 0);
    }

    #[test]
    #[should_panic(expected = "Divide by zero risk: vector length is 0")]
    fn reductive_mean_par_neon_mean_empty_vec() {
        let vector = signed_int_vector(EMPTY);
        let result = mean(vector, Mode::ParNeon);
        assert_eq!(result, 0);
    }

    #[test]
    fn reductive_par_neon_mean_across_parallel_chunks() {
        let vector = range_vector(PAR_CHUNK_SIZE * 2 + 3);
        let result = mean(vector, Mode::ParNeon);

        let len = (PAR_CHUNK_SIZE * 2 + 3) as i32;
        let expected = (0..len).sum::<i32>() / len;
        assert_eq!(result, expected);
    }
}
