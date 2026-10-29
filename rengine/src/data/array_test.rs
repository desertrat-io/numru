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
}
