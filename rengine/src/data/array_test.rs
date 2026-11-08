use crate::data::array::{
    ArrayExtras, BoolArray, NumericArrayExtras, SignedF32Array, SignedF64Array, SignedInt32Array,
};

mod array_tests {
    use super::*;

    #[test]
    fn signed_f32_zeros_fills_requested_length() {
        for len in [1, 2, 7, 64, 1025] {
            let array = SignedF32Array::zeros(len);

            assert_eq!(array.len(), len);
            assert_eq!(array.slice(), vec![0.0_f32; len].as_slice());
        }
    }

    #[test]
    fn signed_f32_zeros_accepts_zero_length() {
        let array = SignedF32Array::zeros(0);

        assert_eq!(array.len(), 0);
        assert!(array.slice().is_empty());
    }

    #[test]
    fn signed_f32_ones_fills_requested_length() {
        for len in [1, 2, 7, 64, 1025] {
            let array = SignedF32Array::ones(len);

            assert_eq!(array.len(), len);
            assert_eq!(array.slice(), vec![1.0_f32; len].as_slice());
        }
    }

    #[test]
    fn signed_f32_ones_accepts_zero_length() {
        let array = SignedF32Array::ones(0);

        assert_eq!(array.len(), 0);
        assert!(array.slice().is_empty());
    }

    #[test]
    fn signed_f64_zeros_fills_requested_length() {
        for len in [1, 2, 7, 64, 1025] {
            let array = SignedF64Array::zeros(len);

            assert_eq!(array.len(), len);
            assert_eq!(array.slice(), vec![0.0_f64; len].as_slice());
        }
    }

    #[test]
    fn signed_f64_zeros_accepts_zero_length() {
        let array = SignedF64Array::zeros(0);

        assert_eq!(array.len(), 0);
        assert!(array.slice().is_empty());
    }

    #[test]
    fn signed_f64_ones_fills_requested_length() {
        for len in [1, 2, 7, 64, 1025] {
            let array = SignedF64Array::ones(len);

            assert_eq!(array.len(), len);
            assert_eq!(array.slice(), vec![1.0_f64; len].as_slice());
        }
    }

    #[test]
    fn signed_f64_ones_accepts_zero_length() {
        let array = SignedF64Array::ones(0);

        assert_eq!(array.len(), 0);
        assert!(array.slice().is_empty());
    }

    #[test]
    fn bool_array_zeros_fills_requested_length() {
        for len in [1, 2, 7, 64, 1025] {
            let array = BoolArray::zeros(len);

            assert_eq!(array.len(), len);
            assert_eq!(array.slice(), vec![false; len].as_slice());
        }
    }

    #[test]
    fn bool_array_zeros_accepts_zero_length() {
        let array = BoolArray::zeros(0);

        assert_eq!(array.len(), 0);
        assert!(array.slice().is_empty());
    }

    #[test]
    fn bool_array_ones_fills_requested_length() {
        for len in [1, 2, 7, 64, 1025] {
            let array = BoolArray::ones(len);

            assert_eq!(array.len(), len);
            assert_eq!(array.slice(), vec![true; len].as_slice());
        }
    }

    #[test]
    fn bool_array_ones_accepts_zero_length() {
        let array = BoolArray::ones(0);

        assert_eq!(array.len(), 0);
        assert!(array.slice().is_empty());
    }

    #[test]
    fn signed_int32_zeros_fills_requested_length() {
        for len in [1, 2, 7, 64, 1025] {
            let array = SignedInt32Array::zeros(len);

            assert_eq!(array.len(), len);
            assert_eq!(array.slice(), vec![0_i32; len].as_slice());
        }
    }

    #[test]
    fn signed_int32_zeros_accepts_zero_length() {
        let array = SignedInt32Array::zeros(0);

        assert_eq!(array.len(), 0);
        assert!(array.slice().is_empty());
    }

    #[test]
    fn signed_int32_ones_fills_requested_length() {
        for len in [1, 2, 7, 64, 1025] {
            let array = SignedInt32Array::ones(len);

            assert_eq!(array.len(), len);
            assert_eq!(array.slice(), vec![1_i32; len].as_slice());
        }
    }

    #[test]
    fn signed_int32_ones_accepts_zero_length() {
        let array = SignedInt32Array::ones(0);

        assert_eq!(array.len(), 0);
        assert!(array.slice().is_empty());
    }

    #[test]
    fn signed_f32_fill_accepts_zero_length() {
        for value in [
            f32::MIN,
            f32::MAX,
            f32::MIN_POSITIVE,
            f32::EPSILON,
            f32::from_bits(1),
            -0.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ] {
            let array = SignedF32Array::fill(0, value);

            assert_eq!(array.len(), 0);
            assert!(array.slice().is_empty());
        }
    }

    #[test]
    fn signed_f32_fill_repeats_value_for_requested_length() {
        for len in [1, 2, 7, 64, 1025] {
            for value in [-3.5_f32, 0.0, 2.25] {
                let array = SignedF32Array::fill(len, value);

                assert_eq!(array.len(), len);
                assert_eq!(array.slice(), vec![value; len].as_slice());
            }
        }
    }

    #[test]
    fn signed_f32_fill_preserves_boundary_values() {
        for value in [
            f32::MIN,
            f32::MAX,
            f32::MIN_POSITIVE,
            f32::EPSILON,
            f32::from_bits(1),
            -0.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ] {
            let array = SignedF32Array::fill(7, value);

            assert_eq!(array.len(), 7);
            assert_eq!(array.slice().len(), 7);
            for actual in array.slice() {
                assert_eq!(actual.to_bits(), value.to_bits());
            }
        }
    }

    #[test]
    fn signed_f64_fill_accepts_zero_length() {
        for value in [
            f64::MIN,
            f64::MAX,
            f64::MIN_POSITIVE,
            f64::EPSILON,
            f64::from_bits(1),
            -0.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ] {
            let array = SignedF64Array::fill(0, value);

            assert_eq!(array.len(), 0);
            assert!(array.slice().is_empty());
        }
    }

    #[test]
    fn signed_f64_fill_repeats_value_for_requested_length() {
        for len in [1, 2, 7, 64, 1025] {
            for value in [-3.5_f64, 0.0, 2.25] {
                let array = SignedF64Array::fill(len, value);

                assert_eq!(array.len(), len);
                assert_eq!(array.slice(), vec![value; len].as_slice());
            }
        }
    }

    #[test]
    fn signed_f64_fill_preserves_boundary_values() {
        for value in [
            f64::MIN,
            f64::MAX,
            f64::MIN_POSITIVE,
            f64::EPSILON,
            f64::from_bits(1),
            -0.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ] {
            let array = SignedF64Array::fill(7, value);

            assert_eq!(array.len(), 7);
            assert_eq!(array.slice().len(), 7);
            for actual in array.slice() {
                assert_eq!(actual.to_bits(), value.to_bits());
            }
        }
    }

    #[test]
    fn signed_int32_fill_accepts_zero_length() {
        for value in [i32::MIN, i32::MAX] {
            let array = SignedInt32Array::fill(0, value);

            assert_eq!(array.len(), 0);
            assert!(array.slice().is_empty());
        }
    }

    #[test]
    fn signed_int32_fill_repeats_value_for_requested_length() {
        for len in [1, 2, 7, 64, 1025] {
            for value in [-42_i32, 0, 17] {
                let array = SignedInt32Array::fill(len, value);

                assert_eq!(array.len(), len);
                assert_eq!(array.slice(), vec![value; len].as_slice());
            }
        }
    }

    #[test]
    fn signed_int32_fill_preserves_boundary_values() {
        for value in [i32::MIN, i32::MAX] {
            let array = SignedInt32Array::fill(7, value);

            assert_eq!(array.len(), 7);
            assert_eq!(array.slice().len(), 7);
            assert_eq!(array.slice(), &[value; 7]);
        }
    }

    #[test]
    fn bool_array_fill_accepts_zero_length() {
        for value in [false, true] {
            let array = BoolArray::fill(0, value);

            assert_eq!(array.len(), 0);
            assert!(array.slice().is_empty());
        }
    }

    #[test]
    fn bool_array_fill_repeats_value_for_requested_length() {
        for len in [1, 2, 7, 64, 1025] {
            for value in [false, true] {
                let array = BoolArray::fill(len, value);

                assert_eq!(array.len(), len);
                assert_eq!(array.slice(), vec![value; len].as_slice());
            }
        }
    }

    #[test]
    fn bool_array_fill_preserves_boundary_values() {
        for value in [false, true] {
            let array = BoolArray::fill(7, value);

            assert_eq!(array.len(), 7);
            assert_eq!(array.slice().len(), 7);
            assert_eq!(array.slice(), &[value; 7]);
        }
    }

    #[test]
    fn signed_f32_range_positive_step_produces_expected_sequence() {
        for exclusive in [true, false] {
            let array = SignedF32Array::range(0.0, 10.0, 1.5, exclusive);

            assert_eq!(array.slice(), &[0.0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0]);
            assert_eq!(array.len(), 7);
            assert_eq!(array.len(), array.slice().len());
        }
    }

    #[test]
    fn signed_f32_range_negative_step_produces_expected_sequence() {
        for exclusive in [true, false] {
            let array = SignedF32Array::range(10.0, 0.0, -1.5, exclusive);

            assert_eq!(array.slice(), &[10.0, 8.5, 7.0, 5.5, 4.0, 2.5, 1.0]);
            assert_eq!(array.len(), 7);
            assert_eq!(array.len(), array.slice().len());
        }
    }

    #[test]
    fn signed_f32_range_end_mode_controls_exact_endpoint_in_both_directions() {
        for (start, end, step, expected_exclusive, expected_inclusive) in [
            (0.0, 3.0, 1.5, [0.0, 1.5], [0.0, 1.5, 3.0]),
            (3.0, 0.0, -1.5, [3.0, 1.5], [3.0, 1.5, 0.0]),
        ] {
            let exclusive = SignedF32Array::range(start, end, step, true);
            let inclusive = SignedF32Array::range(start, end, step, false);

            assert_eq!(exclusive.slice(), &expected_exclusive);
            assert_eq!(exclusive.len(), 2);
            assert_eq!(exclusive.len(), exclusive.slice().len());
            assert_eq!(inclusive.slice(), &expected_inclusive);
            assert_eq!(inclusive.len(), 3);
            assert_eq!(inclusive.len(), inclusive.slice().len());
        }
    }

    #[test]
    fn signed_f32_range_equal_bounds_obey_end_mode() {
        for step in [-1.5, 1.5] {
            let exclusive = SignedF32Array::range(2.0, 2.0, step, true);
            let inclusive = SignedF32Array::range(2.0, 2.0, step, false);

            assert!(exclusive.slice().is_empty());
            assert_eq!(exclusive.len(), 0);
            assert_eq!(inclusive.slice(), &[2.0]);
            assert_eq!(inclusive.len(), 1);
        }
    }

    #[test]
    fn signed_f32_range_step_pointing_away_from_end_is_empty() {
        for exclusive in [true, false] {
            for (start, end, step) in [(0.0, 3.0, -1.0), (3.0, 0.0, 1.0)] {
                let array = SignedF32Array::range(start, end, step, exclusive);

                assert!(array.slice().is_empty());
                assert_eq!(array.len(), 0);
            }
        }
    }

    #[test]
    fn signed_f32_range_step_larger_than_distance_keeps_only_start() {
        for exclusive in [true, false] {
            for (start, end, step) in [(2.0, 3.0, 4.0), (2.0, 1.0, -4.0)] {
                let array = SignedF32Array::range(start, end, step, exclusive);

                assert_eq!(array.slice(), &[2.0]);
                assert_eq!(array.len(), 1);
            }
        }
    }

    #[test]
    #[should_panic(expected = "Invalid step value, 0 not allowed")]
    fn signed_f32_range_rejects_positive_zero_step() {
        SignedF32Array::range(0.0, 3.0, 0.0, true);
    }

    #[test]
    #[should_panic(expected = "Invalid step value, 0 not allowed")]
    fn signed_f32_range_rejects_negative_zero_step() {
        SignedF32Array::range(0.0, 3.0, -0.0, false);
    }

    #[test]
    fn signed_f32_range_uses_index_based_floating_point_values() {
        // These are f32 results for start + i * step, rather than repeated addition.
        let expected = [
            0.0_f32, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.90000004, 1.0,
        ];
        for exclusive in [true, false] {
            let array = SignedF32Array::range(0.0, 1.0, 0.1, exclusive);
            let expected_len = if exclusive { 10 } else { 11 };
            let expected = &expected[..expected_len];

            // Length describes generated values, regardless of the capacity estimate.
            assert_eq!(array.len(), expected.len());
            assert_eq!(array.slice().len(), expected.len());
            for (actual, expected) in array.slice().iter().zip(expected) {
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
        }
    }

    #[test]
    fn signed_f32_range_rejects_non_finite_inputs() {
        for exclusive in [true, false] {
            for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                for (start, end, step) in [
                    (invalid, 3.0, 1.0),
                    (0.0, invalid, 1.0),
                    (0.0, 3.0, invalid),
                ] {
                    assert!(
                        std::panic::catch_unwind(|| {
                            SignedF32Array::range(start, end, step, exclusive)
                        })
                        .is_err(),
                        "expected rejection for ({start}, {end}, {step}, {exclusive})"
                    );
                }
            }
        }
    }

    #[test]
    fn signed_f64_range_positive_step_produces_expected_sequence() {
        for exclusive in [true, false] {
            let array = SignedF64Array::range(0.0, 10.0, 1.5, exclusive);

            assert_eq!(array.slice(), &[0.0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0]);
            assert_eq!(array.len(), 7);
            assert_eq!(array.len(), array.slice().len());
        }
    }

    #[test]
    fn signed_f64_range_negative_step_produces_expected_sequence() {
        for exclusive in [true, false] {
            let array = SignedF64Array::range(10.0, 0.0, -1.5, exclusive);

            assert_eq!(array.slice(), &[10.0, 8.5, 7.0, 5.5, 4.0, 2.5, 1.0]);
            assert_eq!(array.len(), 7);
            assert_eq!(array.len(), array.slice().len());
        }
    }

    #[test]
    fn signed_f64_range_end_mode_controls_exact_endpoint_in_both_directions() {
        for (start, end, step, expected_exclusive, expected_inclusive) in [
            (0.0, 3.0, 1.5, [0.0, 1.5], [0.0, 1.5, 3.0]),
            (3.0, 0.0, -1.5, [3.0, 1.5], [3.0, 1.5, 0.0]),
        ] {
            let exclusive = SignedF64Array::range(start, end, step, true);
            let inclusive = SignedF64Array::range(start, end, step, false);

            assert_eq!(exclusive.slice(), &expected_exclusive);
            assert_eq!(exclusive.len(), 2);
            assert_eq!(exclusive.len(), exclusive.slice().len());
            assert_eq!(inclusive.slice(), &expected_inclusive);
            assert_eq!(inclusive.len(), 3);
            assert_eq!(inclusive.len(), inclusive.slice().len());
        }
    }

    #[test]
    fn signed_f64_range_equal_bounds_obey_end_mode() {
        for step in [-1.5, 1.5] {
            let exclusive = SignedF64Array::range(2.0, 2.0, step, true);
            let inclusive = SignedF64Array::range(2.0, 2.0, step, false);

            assert!(exclusive.slice().is_empty());
            assert_eq!(exclusive.len(), 0);
            assert_eq!(inclusive.slice(), &[2.0]);
            assert_eq!(inclusive.len(), 1);
        }
    }

    #[test]
    fn signed_f64_range_step_pointing_away_from_end_is_empty() {
        for exclusive in [true, false] {
            for (start, end, step) in [(0.0, 3.0, -1.0), (3.0, 0.0, 1.0)] {
                let array = SignedF64Array::range(start, end, step, exclusive);

                assert!(array.slice().is_empty());
                assert_eq!(array.len(), 0);
            }
        }
    }

    #[test]
    fn signed_f64_range_step_larger_than_distance_keeps_only_start() {
        for exclusive in [true, false] {
            for (start, end, step) in [(2.0, 3.0, 4.0), (2.0, 1.0, -4.0)] {
                let array = SignedF64Array::range(start, end, step, exclusive);

                assert_eq!(array.slice(), &[2.0]);
                assert_eq!(array.len(), 1);
            }
        }
    }

    #[test]
    #[should_panic(expected = "Invalid step value, 0 not allowed")]
    fn signed_f64_range_rejects_positive_zero_step() {
        SignedF64Array::range(0.0, 3.0, 0.0, true);
    }

    #[test]
    #[should_panic(expected = "Invalid step value, 0 not allowed")]
    fn signed_f64_range_rejects_negative_zero_step() {
        SignedF64Array::range(0.0, 3.0, -0.0, false);
    }

    #[test]
    fn signed_f64_range_uses_index_based_floating_point_values() {
        // These are f64 results for start + i * step, rather than repeated addition.
        let expected = [
            0.0_f64,
            0.1,
            0.2,
            0.30000000000000004,
            0.4,
            0.5,
            0.6000000000000001,
            0.7000000000000001,
            0.8,
            0.9,
            1.0,
        ];
        for exclusive in [true, false] {
            let array = SignedF64Array::range(0.0, 1.0, 0.1, exclusive);
            let expected_len = if exclusive { 10 } else { 11 };
            let expected = &expected[..expected_len];

            // Length describes generated values, regardless of the capacity estimate.
            assert_eq!(array.len(), expected.len());
            assert_eq!(array.slice().len(), expected.len());
            for (actual, expected) in array.slice().iter().zip(expected) {
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
        }
    }

    #[test]
    fn signed_f64_range_rejects_non_finite_inputs() {
        for exclusive in [true, false] {
            for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                for (start, end, step) in [
                    (invalid, 3.0, 1.0),
                    (0.0, invalid, 1.0),
                    (0.0, 3.0, invalid),
                ] {
                    assert!(
                        std::panic::catch_unwind(|| {
                            SignedF64Array::range(start, end, step, exclusive)
                        })
                        .is_err(),
                        "expected rejection for ({start}, {end}, {step}, {exclusive})"
                    );
                }
            }
        }
    }
}
