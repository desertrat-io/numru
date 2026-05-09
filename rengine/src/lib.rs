pub mod math {
    pub mod arith;
    pub mod trig;
}

pub mod data {
    pub mod array;
}
pub mod utils {
    pub mod optypes;
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::array::Array;
    use crate::math::arith::Mode;

    // repetitive tests for repetitive operations, but will
    // detect any variance in the arithmetic algorithm
    fn get_fixture() -> (Array, Array) {
        (
            Array::new(vec![1.0, 2.0, 3.0]),
            Array::new(vec![4.0, 5.0, 6.0]),
        )
    }

    // experimental assertion for accuracy of floating points
    // that have had unary operations done on them
    fn assert_close_slice(actual: &[f32], expected: &[f32]) {
        const EPSILON: f32 = 0.000_001;

        assert_eq!(actual.len(), expected.len());
        for i in 0..actual.len() {
            assert!(
                (actual[i] - expected[i]).abs() <= EPSILON,
                "expected index {i} to be close to {}, got {}",
                expected[i],
                actual[i],
            );
        }
    }

    fn get_unary_fixture() -> Array {
        Array::new(vec![-4.0, -1.0, 0.0, 1.0, 4.0, 9.0])
    }

    #[test]
    fn test_abs_32() {
        let result = math::arith::abs(get_unary_fixture(), Mode::Normal);
        assert_eq!(result.slice(), [4.0, 1.0, 0.0, 1.0, 4.0, 9.0]);
    }

    #[test]
    fn test_abs_32_neon() {
        let result = math::arith::abs(get_unary_fixture(), Mode::Neon);
        assert_eq!(result.slice(), [4.0, 1.0, 0.0, 1.0, 4.0, 9.0]);
    }

    #[test]
    fn test_abs_32_par_neon() {
        let result = math::arith::abs(get_unary_fixture(), Mode::ParNeon);
        assert_eq!(result.slice(), [4.0, 1.0, 0.0, 1.0, 4.0, 9.0]);
    }

    #[test]
    fn test_neg_32() {
        let result = math::arith::neg(get_unary_fixture(), Mode::Normal);
        assert_eq!(result.slice(), [4.0, 1.0, -0.0, -1.0, -4.0, -9.0]);
    }

    #[test]
    fn test_neg_32_neon() {
        let result = math::arith::neg(get_unary_fixture(), Mode::Neon);
        assert_eq!(result.slice(), [4.0, 1.0, -0.0, -1.0, -4.0, -9.0]);
    }

    #[test]
    fn test_neg_32_par_neon() {
        let result = math::arith::neg(get_unary_fixture(), Mode::ParNeon);
        assert_eq!(result.slice(), [4.0, 1.0, -0.0, -1.0, -4.0, -9.0]);
    }

    #[test]
    fn test_sqrt_32() {
        let input = Array::new(vec![0.0, 1.0, 2.0, 4.0, 9.0, 16.0]);
        let result = math::arith::sqrt(input, Mode::Normal);
        assert_close_slice(result.slice(), &[0.0, 1.0, 2.0_f32.sqrt(), 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_sqrt_32_neon() {
        let input = Array::new(vec![0.0, 1.0, 2.0, 4.0, 9.0, 16.0]);
        let result = math::arith::sqrt(input, Mode::Neon);
        assert_close_slice(result.slice(), &[0.0, 1.0, 2.0_f32.sqrt(), 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_sqrt_32_par_neon() {
        let input = Array::new(vec![0.0, 1.0, 2.0, 4.0, 9.0, 16.0]);
        let result = math::arith::sqrt(input, Mode::ParNeon);
        assert_close_slice(result.slice(), &[0.0, 1.0, 2.0_f32.sqrt(), 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_sqrt_32_negative_is_nan() {
        let input = Array::new(vec![-1.0]);
        let result = math::arith::sqrt(input, Mode::Normal);
        assert!(result.slice()[0].is_nan());
    }

    #[test]
    fn test_log_32() {
        let input = Array::new(vec![1.0, 10.0, 100.0, 1000.0]);
        let result = math::arith::log(input, Mode::Normal);
        assert_close_slice(result.slice(), &[0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_log_32_zero_is_negative_infinity() {
        let input = Array::new(vec![0.0]);
        let result = math::arith::log(input, Mode::Normal);
        assert_eq!(result.slice()[0], f32::NEG_INFINITY);
    }

    #[test]
    fn test_log_32_negative_is_nan() {
        let input = Array::new(vec![-1.0]);
        let result = math::arith::log(input, Mode::Normal);
        assert!(result.slice()[0].is_nan());
    }

    #[test]
    fn test_exp_32() {
        let input = Array::new(vec![0.0, 1.0, 2.0]);
        let result = math::arith::exp(input, Mode::Normal);
        assert_close_slice(result.slice(), &[1.0, 1.0_f32.exp(), 2.0_f32.exp()]);
    }

    // TODO: remove when this is actually working

    #[test]
    #[should_panic(expected = "exp only supported in scalar mode for now")]
    fn test_exp_32_neon_panics() {
        let input = Array::new(vec![1.0]);
        math::arith::exp(input, Mode::Neon);
    }

    #[test]
    #[should_panic(expected = "exp only supported in scalar mode for now")]
    fn test_exp_32_par_neon_panics() {
        let input = Array::new(vec![1.0]);
        math::arith::exp(input, Mode::ParNeon);
    }

    #[test]
    fn test_add_32() {
        let (left, right) = get_fixture();
        let result = math::arith::add(left, right, Mode::Normal);
        assert_eq!(result.slice(), [5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_add_32_neon() {
        let (left, right) = get_fixture();
        let result = math::arith::add(left, right, Mode::Neon);
        assert_eq!(result.slice(), [5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_add_32_neon_par() {
        let (left, right) = get_fixture();
        let result = math::arith::add(left, right, Mode::ParNeon);
        assert_eq!(result.slice(), [5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_sub_32() {
        let (left, right) = get_fixture();
        let result = math::arith::sub(left, right, Mode::Normal);
        assert_eq!(result.slice(), [-3.0, -3.0, -3.0]);
    }

    #[test]
    fn test_sub_32_neon() {
        let (left, right) = get_fixture();
        let result = math::arith::sub(left, right, Mode::Neon);
        assert_eq!(result.slice(), [-3.0, -3.0, -3.0]);
    }

    #[test]
    fn test_sub_32_par_neon() {
        let (left, right) = get_fixture();
        let result = math::arith::sub(left, right, Mode::ParNeon);
        assert_eq!(result.slice(), [-3.0, -3.0, -3.0]);
    }

    #[test]
    fn test_mul_32() {
        let (left, right) = get_fixture();
        let result = math::arith::mul(left, right, Mode::Normal);
        assert_eq!(result.slice(), [4.0, 10.0, 18.0]);
    }

    #[test]
    fn test_mul_32_neon() {
        let (left, right) = get_fixture();
        let result = math::arith::mul(left, right, Mode::Neon);
        assert_eq!(result.slice(), [4.0, 10.0, 18.0]);
    }

    #[test]
    fn test_mul_32_par_neon() {
        let (left, right) = get_fixture();
        let result = math::arith::mul(left, right, Mode::ParNeon);
        assert_eq!(result.slice(), [4.0, 10.0, 18.0]);
    }

    #[test]
    fn test_div_32() {
        let (left, right) = get_fixture();
        let result = math::arith::div(left, right, Mode::Normal);
        assert_eq!(result.slice(), [0.25, 0.4, 0.5]);
    }

    #[test]
    fn test_div_32_neon() {
        let (left, right) = get_fixture();
        let result = math::arith::div(left, right, Mode::Neon);
        assert_eq!(result.slice(), [0.25, 0.4, 0.5]);
    }

    #[test]
    fn test_div_32_par_neon() {
        let (left, right) = get_fixture();
        let result = math::arith::div(left, right, Mode::ParNeon);
        assert_eq!(result.slice(), [0.25, 0.4, 0.5]);
    }

    #[test]
    #[should_panic]
    fn test_div_32_by_zero() {
        let left = Array::new(vec![0.0]);
        let right = Array::new(vec![0.0]);
        math::arith::div(left, right, Mode::Normal);
    }

    // allow simple and par neon ONLY
    #[test]
    fn test_add_mul_32() {
        let left = Array::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let middle = Array::new(vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
        let right = Array::new(vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let result = math::arith::add_mul(left, middle, right, Mode::Normal);
        assert_eq!(result.slice(), [41.0, 57.0, 75.0, 95.0, 117.0, 141.0]);
    }

    #[test]
    fn test_add_mul_32_neon() {
        let left = Array::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let middle = Array::new(vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
        let right = Array::new(vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let result = math::arith::add_mul(left, middle, right, Mode::Neon);
        assert_eq!(result.slice(), [41.0, 57.0, 75.0, 95.0, 117.0, 141.0]);
    }

    #[test]
    fn test_add_mul_32_par_neon() {
        let left = Array::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let middle = Array::new(vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
        let right = Array::new(vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let result = math::arith::add_mul(left, middle, right, Mode::ParNeon);
        assert_eq!(result.slice(), [41.0, 57.0, 75.0, 95.0, 117.0, 141.0]);
    }
}
