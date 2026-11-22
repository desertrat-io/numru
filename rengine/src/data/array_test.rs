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

    #[test]
    fn signed_f32_linspace_includes_both_endpoints() {
        let array = SignedF32Array::linspace(0.0, 1.0, 5);
        assert_eq!(array.len(), 5);
        assert_eq!(array.slice(), &[0.0, 0.25, 0.5, 0.75, 1.0]);
    }

    #[test]
    fn signed_f32_linspace_zero_count_is_empty() {
        for (start, end) in [(0.0, 1.0), (1.0, -1.0), (2.0, 2.0)] {
            let array = SignedF32Array::linspace(start, end, 0);
            assert_eq!(array.len(), 0);
            assert!(array.slice().is_empty());
        }
    }

    #[test]
    fn signed_f32_linspace_one_count_preserves_start() {
        for start in [
            -3.5,
            0.0,
            -0.0,
            f32::MIN,
            f32::MAX,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
        ] {
            let array = SignedF32Array::linspace(start, 7.0, 1);
            assert_eq!(array.len(), 1);
            assert_eq!(array.slice()[0].to_bits(), start.to_bits());
        }
    }

    #[test]
    fn signed_f32_linspace_two_count_returns_endpoints() {
        for (start, end) in [(2.0, 9.0), (9.0, 2.0), (-4.0, 3.0)] {
            let array = SignedF32Array::linspace(start, end, 2);
            assert_eq!(array.slice(), &[start, end]);
        }
    }

    #[test]
    fn signed_f32_linspace_descends_and_crosses_zero() {
        assert_eq!(
            SignedF32Array::linspace(1.0, -1.0, 5).slice(),
            &[1.0, 0.5, 0.0, -0.5, -1.0]
        );
        assert_eq!(
            SignedF32Array::linspace(-1.0, 1.0, 5).slice(),
            &[-1.0, -0.5, 0.0, 0.5, 1.0]
        );
    }

    #[test]
    fn signed_f32_linspace_equal_bounds_repeat_value() {
        for value in [-2.5, 0.0, f32::MIN, f32::MAX] {
            for n in [2, 5, 64] {
                let array = SignedF32Array::linspace(value, value, n);
                assert_eq!(array.len(), n);
                assert!(array.slice().iter().all(|&actual| actual == value));
            }
        }
    }

    #[test]
    fn signed_f32_linspace_fractional_spacing_with_even_count() {
        let array = SignedF32Array::linspace(0.0, 1.0, 4);
        let expected: [f32; 4] = [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0];
        assert_eq!(array.len(), expected.len());
        for (&actual, expected) in array.slice().iter().zip(expected) {
            assert!((actual - expected).abs() <= 4.0 * f32::EPSILON);
        }
    }

    #[test]
    fn signed_f32_linspace_preserves_exact_endpoints_despite_rounding() {
        for (start, end) in [(-1.0, 0.1), (0.1, -1.0)] {
            let array = SignedF32Array::linspace(start, end, 4);
            assert_eq!(array.slice()[0], start);
            assert_eq!(array.slice()[3], end);
        }
    }

    #[test]
    fn signed_f32_linspace_has_requested_length_and_contiguous_storage() {
        for n in [0, 1, 2, 3, 16, 65, 1025] {
            let array = SignedF32Array::linspace(-2.0, 2.0, n);
            let values = array.slice();
            assert_eq!(array.len(), n);
            assert_eq!(values.len(), n);
            for (i, value) in values.iter().enumerate() {
                assert_eq!(value as *const f32, values.as_ptr().wrapping_add(i));
            }
        }
    }

    #[test]
    fn signed_f32_linspace_preserves_subnormal_samples() {
        let tiny = f32::from_bits(1);
        let array = SignedF32Array::linspace(0.0, tiny * 4.0, 5);
        assert_eq!(
            array.slice(),
            &[0.0, tiny, tiny * 2.0, tiny * 3.0, tiny * 4.0]
        );
    }

    // All expected samples are finite and representable. These regressions
    // expose intermediate overflow without defining non-finite input policy.
    #[test]
    fn signed_f32_linspace_extreme_opposite_bounds_do_not_overflow_subtraction() {
        let max = f32::MAX;
        for (start, end) in [(-max, max), (max, -max)] {
            let array = SignedF32Array::linspace(start, end, 3);
            assert_eq!(array.slice(), &[start, 0.0, end]);
        }
    }

    #[test]
    fn signed_f32_linspace_extreme_distance_does_not_overflow_multiplication() {
        let max = f32::MAX;
        for end in [max, -max] {
            let array = SignedF32Array::linspace(0.0, end, 5);
            let expected = [0.0, end / 4.0, end / 2.0, (end / 4.0) * 3.0, end];
            assert_eq!(array.len(), expected.len());
            for (&actual, expected) in array.slice().iter().zip(expected) {
                assert!(actual.is_finite(), "non-finite sample: {actual}");
                // Normalize before subtracting to avoid overflow in the assertion.
                assert!((actual / max - expected / max).abs() <= 4.0 * f32::EPSILON);
            }
            assert_eq!(array.slice()[4], end);
        }
    }

    #[test]
    #[should_panic(expected = "capacity overflow")]
    fn signed_f32_linspace_rejects_unrepresentable_allocation_size() {
        // Vec rejects this layout before allocation; do not attempt an OOM test.
        SignedF32Array::linspace(0.0, 1.0, usize::MAX);
    }

    #[test]
    fn signed_f64_linspace_includes_both_endpoints() {
        let array = SignedF64Array::linspace(0.0, 1.0, 5);
        assert_eq!(array.len(), 5);
        assert_eq!(array.slice(), &[0.0, 0.25, 0.5, 0.75, 1.0]);
    }

    #[test]
    fn signed_f64_linspace_zero_count_is_empty() {
        for (start, end) in [(0.0, 1.0), (1.0, -1.0), (2.0, 2.0)] {
            let array = SignedF64Array::linspace(start, end, 0);
            assert_eq!(array.len(), 0);
            assert!(array.slice().is_empty());
        }
    }

    #[test]
    fn signed_f64_linspace_one_count_preserves_start() {
        for start in [
            -3.5,
            0.0,
            -0.0,
            f64::MIN,
            f64::MAX,
            f64::MIN_POSITIVE,
            f64::from_bits(1),
        ] {
            let array = SignedF64Array::linspace(start, 7.0, 1);
            assert_eq!(array.len(), 1);
            assert_eq!(array.slice()[0].to_bits(), start.to_bits());
        }
    }

    #[test]
    fn signed_f64_linspace_two_count_returns_endpoints() {
        for (start, end) in [(2.0, 9.0), (9.0, 2.0), (-4.0, 3.0)] {
            let array = SignedF64Array::linspace(start, end, 2);
            assert_eq!(array.slice(), &[start, end]);
        }
    }

    #[test]
    fn signed_f64_linspace_descends_and_crosses_zero() {
        assert_eq!(
            SignedF64Array::linspace(1.0, -1.0, 5).slice(),
            &[1.0, 0.5, 0.0, -0.5, -1.0]
        );
        assert_eq!(
            SignedF64Array::linspace(-1.0, 1.0, 5).slice(),
            &[-1.0, -0.5, 0.0, 0.5, 1.0]
        );
    }

    #[test]
    fn signed_f64_linspace_equal_bounds_repeat_value() {
        for value in [-2.5, 0.0, f64::MIN, f64::MAX] {
            for n in [2, 5, 64] {
                let array = SignedF64Array::linspace(value, value, n);
                assert_eq!(array.len(), n);
                assert!(array.slice().iter().all(|&actual| actual == value));
            }
        }
    }

    #[test]
    fn signed_f64_linspace_fractional_spacing_with_even_count() {
        let array = SignedF64Array::linspace(0.0, 1.0, 4);
        let expected: [f64; 4] = [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0];
        assert_eq!(array.len(), expected.len());
        for (&actual, expected) in array.slice().iter().zip(expected) {
            assert!((actual - expected).abs() <= 4.0 * f64::EPSILON);
        }
    }

    #[test]
    fn signed_f64_linspace_preserves_exact_endpoints_despite_rounding() {
        for (start, end) in [(-1.0, 0.1), (0.1, -1.0)] {
            let array = SignedF64Array::linspace(start, end, 4);
            assert_eq!(array.slice()[0], start);
            assert_eq!(array.slice()[3], end);
        }
    }

    #[test]
    fn signed_f64_linspace_has_requested_length_and_contiguous_storage() {
        for n in [0, 1, 2, 3, 16, 65, 1025] {
            let array = SignedF64Array::linspace(-2.0, 2.0, n);
            let values = array.slice();
            assert_eq!(array.len(), n);
            assert_eq!(values.len(), n);
            for (i, value) in values.iter().enumerate() {
                assert_eq!(value as *const f64, values.as_ptr().wrapping_add(i));
            }
        }
    }

    #[test]
    fn signed_f64_linspace_preserves_subnormal_samples() {
        let tiny = f64::from_bits(1);
        let array = SignedF64Array::linspace(0.0, tiny * 4.0, 5);
        assert_eq!(
            array.slice(),
            &[0.0, tiny, tiny * 2.0, tiny * 3.0, tiny * 4.0]
        );
    }

    // All expected samples are finite and representable. These regressions
    // expose intermediate overflow without defining non-finite input policy.
    #[test]
    fn signed_f64_linspace_extreme_opposite_bounds_do_not_overflow_subtraction() {
        let max = f64::MAX;
        for (start, end) in [(-max, max), (max, -max)] {
            let array = SignedF64Array::linspace(start, end, 3);
            assert_eq!(array.slice(), &[start, 0.0, end]);
        }
    }

    #[test]
    fn signed_f64_linspace_extreme_distance_does_not_overflow_multiplication() {
        let max = f64::MAX;
        for end in [max, -max] {
            let array = SignedF64Array::linspace(0.0, end, 5);
            let expected = [0.0, end / 4.0, end / 2.0, (end / 4.0) * 3.0, end];
            assert_eq!(array.len(), expected.len());
            for (&actual, expected) in array.slice().iter().zip(expected) {
                assert!(actual.is_finite(), "non-finite sample: {actual}");
                // Normalize before subtracting to avoid overflow in the assertion.
                assert!((actual / max - expected / max).abs() <= 4.0 * f64::EPSILON);
            }
            assert_eq!(array.slice()[4], end);
        }
    }

    #[test]
    #[should_panic(expected = "capacity overflow")]
    fn signed_f64_linspace_rejects_unrepresentable_allocation_size() {
        // Vec rejects this layout before allocation; do not attempt an OOM test.
        SignedF64Array::linspace(0.0, 1.0, usize::MAX);
    }
}

macro_rules! reshape_tests {
    ($module:ident, $array:ty, $values:expr, $replacement:expr) => {
        mod $module {
            use super::*;

            #[test]
            fn copies_elements_in_order_into_independent_storage() {
                let values = $values;
                let original = <$array>::new(values.to_vec());
                let mut reshaped = <$array>::reshape_cp(&original, &[2, 3]);

                assert_eq!(reshaped.shape(), &[2, 3]);
                assert_eq!(reshaped.slice(), &values);
                reshaped.mut_slice()[0] = $replacement;
                assert_eq!(original.shape(), &[6]);
                assert_eq!(original.slice(), &values);
            }

            #[test]
            fn reshapes_empty_1d_array() {
                let original = <$array>::new(vec![]);
                assert_eq!(original.shape(), &[0]);

                let reshaped = <$array>::reshape_cp(&original, &[0]);

                assert_eq!(reshaped.shape(), &[0]);
                assert!(reshaped.slice().is_empty());
            }

            #[test]
            #[should_panic(expected = "shape overflow")]
            fn rejects_shape_product_overflow() {
                let original = <$array>::new(($values).to_vec());
                <$array>::reshape_cp(&original, &[usize::MAX, 2]);
            }

            #[test]
            #[should_panic(expected = "Invalid shape")]
            fn rejects_element_count_mismatch() {
                let original = <$array>::new(($values).to_vec());
                <$array>::reshape_cp(&original, &[2, 2]);
            }

            #[test]
            #[should_panic]
            fn rejects_zero_length_shape_for_nonempty_array() {
                let original = <$array>::new(($values).to_vec());
                <$array>::reshape_cp(&original, &[0]);
            }
        }
    };
}

reshape_tests!(reshape_i32, SignedInt32Array, [-3, -2, -1, 0, 1, 2], 9);
reshape_tests!(
    reshape_bool,
    BoolArray,
    [false, true, true, false, true, false],
    true
);
reshape_tests!(
    reshape_f32,
    SignedF32Array,
    [-1.5, 0.0, 0.5, 1.0, 2.0, 3.0],
    9.0
);
reshape_tests!(
    reshape_f64,
    SignedF64Array,
    [-1.5, 0.0, 0.5, 1.0, 2.0, 3.0],
    9.0
);
