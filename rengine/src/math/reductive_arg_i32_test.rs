// TODO: Codex experimental test code, evaluate for usefulness and correctness
#[cfg(test)]
mod tests {
    use crate::data::array::SignedIntArray;
    use crate::math::reductive_arg_i32::{argmax, argmin};
    use crate::matrix::ops::Mode;

    const EMPTY: &[i32] = &[];
    const SINGLE: &[i32] = &[42];
    const ASCENDING: &[i32] = &[-3, -2, -1, 0, 1, 2, 3];
    const DESCENDING: &[i32] = &[3, 2, 1, 0, -1, -2, -3];
    const MIN_IN_MIDDLE: &[i32] = &[7, -4, 0, -9, 3];
    const MAX_IN_MIDDLE: &[i32] = &[-7, 4, 0, 9, -3];
    const DUPLICATE_MIN: &[i32] = &[5, -3, 2, -3, 8];
    const DUPLICATE_MAX: &[i32] = &[-5, 3, 2, 3, -8];
    const I32_MIN_AT_MIDDLE: &[i32] = &[10, i32::MIN, -1, 0];
    const I32_MAX_AT_MIDDLE: &[i32] = &[-10, i32::MAX, 1, 0];
    const ALL_I32_MAX: &[i32] = &[i32::MAX, i32::MAX];
    const ALL_I32_MIN: &[i32] = &[i32::MIN, i32::MIN];
    const NEON_ARGMIN_EXACT_CHUNK: &[i32] = &[12, 11, 10, 9, 8, 7, 6, 5, -4, 3, 2, 1, 0, 4, 13, 14];
    const NEON_ARGMAX_EXACT_CHUNK: &[i32] = &[
        -12, -11, -10, -9, -8, -7, -6, -5, 4, -3, -2, -1, 0, -4, 13, -14,
    ];
    const NEON_ARGMIN_WITH_REMAINDER: &[i32] = &[
        8, 7, 6, 5, 4, 3, 2, 1, 0, 9, 10, 11, 12, 13, 14, 15, -3, 16, 17,
    ];
    const NEON_ARGMAX_WITH_REMAINDER: &[i32] = &[
        -8, -7, -6, -5, -4, -3, -2, -1, 0, -9, -10, -11, -12, -13, -14, -15, 3, -16, -17,
    ];
    const NEON_DUPLICATE_MIN: &[i32] = &[
        9, 8, 7, 6, 5, 4, 3, -2, 1, 0, -1, -2, 10, 11, 12, 13, -2, 14, 15,
    ];
    const NEON_DUPLICATE_MAX: &[i32] = &[
        -9, -8, -7, -6, -5, -4, -3, 2, -1, 0, 1, 2, -10, -11, -12, -13, 2, -14, -15,
    ];
    const NEON_ARGMIN_EXACT_CHUNK_NONZERO_LANE: &[i32] =
        &[12, 11, 10, 9, 8, 7, 6, 5, 4, 3, -4, 1, 0, 2, 13, 14];
    const NEON_I32_MIN_EXACT_CHUNK: &[i32] =
        &[12, 11, 10, 9, 8, 7, 6, 5, 4, 3, i32::MIN, 1, 0, 2, 13, 14];
    const NEON_I32_MAX_EXACT_CHUNK: &[i32] = &[
        -12,
        -11,
        -10,
        -9,
        -8,
        -7,
        -6,
        -5,
        -4,
        -3,
        i32::MAX,
        -1,
        0,
        -2,
        -13,
        -14,
    ];

    fn signed_int_vector(values: &[i32]) -> SignedIntArray {
        SignedIntArray::new(values.to_vec())
    }

    #[test]
    fn argmin_returns_index_of_min_value() {
        let vector = signed_int_vector(MIN_IN_MIDDLE);
        assert_eq!(argmin(vector, Mode::Normal), 3);
    }

    #[test]
    fn argmax_returns_index_of_max_value() {
        let vector = signed_int_vector(MAX_IN_MIDDLE);
        assert_eq!(argmax(vector, Mode::Normal), 3);
    }

    #[test]
    fn argmin_returns_zero_for_single_value() {
        let vector = signed_int_vector(SINGLE);
        assert_eq!(argmin(vector, Mode::Normal), 0);
    }

    #[test]
    fn argmax_returns_zero_for_single_value() {
        let vector = signed_int_vector(SINGLE);
        assert_eq!(argmax(vector, Mode::Normal), 0);
    }

    #[test]
    fn argmin_returns_first_index_for_ascending_values() {
        let vector = signed_int_vector(ASCENDING);
        assert_eq!(argmin(vector, Mode::Normal), 0);
    }

    #[test]
    fn argmax_returns_last_index_for_ascending_values() {
        let vector = signed_int_vector(ASCENDING);
        assert_eq!(argmax(vector, Mode::Normal), 6);
    }

    #[test]
    fn argmin_returns_last_index_for_descending_values() {
        let vector = signed_int_vector(DESCENDING);
        assert_eq!(argmin(vector, Mode::Normal), 6);
    }

    #[test]
    fn argmax_returns_first_index_for_descending_values() {
        let vector = signed_int_vector(DESCENDING);
        assert_eq!(argmax(vector, Mode::Normal), 0);
    }

    #[test]
    fn argmin_returns_first_duplicate_min_index() {
        let vector = signed_int_vector(DUPLICATE_MIN);
        assert_eq!(argmin(vector, Mode::Normal), 1);
    }

    #[test]
    fn argmax_returns_first_duplicate_max_index() {
        let vector = signed_int_vector(DUPLICATE_MAX);
        assert_eq!(argmax(vector, Mode::Normal), 1);
    }

    #[test]
    fn argmin_handles_i32_min() {
        let vector = signed_int_vector(I32_MIN_AT_MIDDLE);
        assert_eq!(argmin(vector, Mode::Normal), 1);
    }

    #[test]
    fn argmax_handles_i32_max() {
        let vector = signed_int_vector(I32_MAX_AT_MIDDLE);
        assert_eq!(argmax(vector, Mode::Normal), 1);
    }

    #[test]
    fn argmin_returns_zero_for_empty_input() {
        let vector = signed_int_vector(EMPTY);
        assert_eq!(argmin(vector, Mode::Normal), 0);
    }

    #[test]
    fn argmax_returns_zero_for_empty_input() {
        let vector = signed_int_vector(EMPTY);
        assert_eq!(argmax(vector, Mode::Normal), 0);
    }

    #[test]
    fn argmin_returns_zero_when_all_values_equal_initial_edge_value() {
        let vector = signed_int_vector(ALL_I32_MAX);
        assert_eq!(argmin(vector, Mode::Normal), 0);
    }

    #[test]
    fn argmax_returns_zero_when_all_values_equal_initial_edge_value() {
        let vector = signed_int_vector(ALL_I32_MIN);
        assert_eq!(argmax(vector, Mode::Normal), 0);
    }

    #[test]
    fn argmin_neon_returns_index_from_exact_chunk() {
        let vector = signed_int_vector(NEON_ARGMIN_EXACT_CHUNK);
        assert_eq!(argmin(vector, Mode::Neon), 8);
    }

    #[test]
    fn argmax_neon_returns_index_from_exact_chunk() {
        let vector = signed_int_vector(NEON_ARGMAX_EXACT_CHUNK);
        assert_eq!(argmax(vector, Mode::Neon), 14);
    }

    #[test]
    fn argmin_neon_checks_remainder_after_full_chunk() {
        let vector = signed_int_vector(NEON_ARGMIN_WITH_REMAINDER);
        assert_eq!(argmin(vector, Mode::Neon), 16);
    }

    #[test]
    fn argmax_neon_checks_remainder_after_full_chunk() {
        let vector = signed_int_vector(NEON_ARGMAX_WITH_REMAINDER);
        assert_eq!(argmax(vector, Mode::Neon), 16);
    }

    #[test]
    fn argmin_neon_returns_first_duplicate_min_index() {
        let vector = signed_int_vector(NEON_DUPLICATE_MIN);
        assert_eq!(argmin(vector, Mode::Neon), 7);
    }

    #[test]
    fn argmax_neon_returns_first_duplicate_max_index() {
        let vector = signed_int_vector(NEON_DUPLICATE_MAX);
        assert_eq!(argmax(vector, Mode::Neon), 7);
    }

    #[test]
    #[should_panic]
    fn argmin_neon_empty_input_panics() {
        let vector = signed_int_vector(EMPTY);
        argmin(vector, Mode::Neon);
    }

    #[test]
    #[should_panic]
    fn argmax_neon_empty_input_panics() {
        let vector = signed_int_vector(EMPTY);
        argmax(vector, Mode::Neon);
    }

    #[test]
    fn argmin_neon_handles_input_shorter_than_chunk() {
        let vector = signed_int_vector(MIN_IN_MIDDLE);
        assert_eq!(argmin(vector, Mode::Neon), 3);
    }

    #[test]
    fn argmax_neon_handles_input_shorter_than_chunk() {
        let vector = signed_int_vector(MAX_IN_MIDDLE);
        assert_eq!(argmax(vector, Mode::Neon), 3);
    }

    #[test]
    fn argmin_neon_compares_all_lanes_in_exact_chunk() {
        let vector = signed_int_vector(NEON_ARGMIN_EXACT_CHUNK_NONZERO_LANE);
        assert_eq!(argmin(vector, Mode::Neon), 10);
    }

    #[test]
    fn argmin_neon_handles_i32_min() {
        let vector = signed_int_vector(NEON_I32_MIN_EXACT_CHUNK);
        assert_eq!(argmin(vector, Mode::Neon), 10);
    }

    #[test]
    fn argmax_neon_handles_i32_max() {
        let vector = signed_int_vector(NEON_I32_MAX_EXACT_CHUNK);
        assert_eq!(argmax(vector, Mode::Neon), 10);
    }

    #[test]
    fn argmax_neon_par_handles_i32_max() {
        let vector = signed_int_vector(NEON_I32_MAX_EXACT_CHUNK);
        assert_eq!(argmax(vector, Mode::ParNeon), 10);
    }

    #[test]
    fn argmin_neon_par_returns_index_from_exact_chunk() {
        let vector = signed_int_vector(NEON_ARGMIN_EXACT_CHUNK);
        assert_eq!(argmin(vector, Mode::ParNeon), 8);
    }

    #[test]
    fn argmax_neon_par_returns_index_from_exact_chunk() {
        let vector = signed_int_vector(NEON_ARGMAX_EXACT_CHUNK);
        assert_eq!(argmax(vector, Mode::ParNeon), 14);
    }

    #[test]
    fn argmin_neon_par_checks_remainder_after_full_chunk() {
        let vector = signed_int_vector(NEON_ARGMIN_WITH_REMAINDER);
        assert_eq!(argmin(vector, Mode::ParNeon), 16);
    }

    #[test]
    fn argmax_neon_par_checks_remainder_after_full_chunk() {
        let vector = signed_int_vector(NEON_ARGMAX_WITH_REMAINDER);
        assert_eq!(argmax(vector, Mode::ParNeon), 16);
    }

    #[test]
    fn argmin_neon_par_returns_first_duplicate_min_index() {
        let vector = signed_int_vector(NEON_DUPLICATE_MIN);
        assert_eq!(argmin(vector, Mode::ParNeon), 7);
    }

    #[test]
    fn argmax_neon_par_returns_first_duplicate_max_index() {
        let vector = signed_int_vector(NEON_DUPLICATE_MAX);
        assert_eq!(argmax(vector, Mode::ParNeon), 7);
    }

    #[test]
    #[should_panic]
    fn argmin_neon_par_empty_input_panics() {
        let vector = signed_int_vector(EMPTY);
        argmin(vector, Mode::ParNeon);
    }

    #[test]
    #[should_panic]
    fn argmax_neon_test_empty_input_panics() {
        let vector = signed_int_vector(EMPTY);
        argmax(vector, Mode::ParNeon);
    }

    #[test]
    fn argmin_neon_par_andles_input_shorter_than_chunk() {
        let vector = signed_int_vector(MIN_IN_MIDDLE);
        assert_eq!(argmin(vector, Mode::ParNeon), 3);
    }

    #[test]
    fn argmax_neon_par_handles_input_shorter_than_chunk() {
        let vector = signed_int_vector(MAX_IN_MIDDLE);
        assert_eq!(argmax(vector, Mode::ParNeon), 3);
    }

    #[test]
    fn argmin_neon_par_compares_all_lanes_in_exact_chunk() {
        let vector = signed_int_vector(NEON_ARGMIN_EXACT_CHUNK_NONZERO_LANE);
        assert_eq!(argmin(vector, Mode::ParNeon), 10);
    }

    #[test]
    fn argmin_neon_par_handles_i32_min() {
        let vector = signed_int_vector(NEON_I32_MIN_EXACT_CHUNK);
        assert_eq!(argmin(vector, Mode::ParNeon), 10);
    }
}
