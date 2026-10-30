use crate::data::array::{
    ArrayExtras, BoolArray, SignedF32Array, SignedF64Array, SignedInt32Array,
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
}
