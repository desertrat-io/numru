#[cfg(test)]
mod tests {
    use crate::data::array::SignedIntArray;
    use crate::math::reductive_arith::{max, min, sum};
    use crate::matrix::ops::Mode;

    #[test]
    fn test_reductive_scalar_add_32() {
        let vector = SignedIntArray::new(vec![1, 2, 3]);
        let result = sum(vector, Mode::Normal);
        assert_eq!(result, 6);
        let vector = SignedIntArray::new(vec![]);
        let result = sum(vector, Mode::Normal);
        assert_eq!(result, 0);
        let vector = SignedIntArray::new(vec![-1, -2, -3, -4]);
        let result = sum(vector, Mode::Normal);
        assert_eq!(result, -10);
    }

    #[test]
    fn signed_int_neon_1_sums_full_vector() {
        let vector = SignedIntArray::new(vec![1, 2, 3, 4]);
        let result = sum(vector, Mode::Neon);
        assert_eq!(result, 10);
    }

    #[test]
    fn signed_int_neon_1_sums_full_vector_with_negative() {
        let vector = SignedIntArray::new(vec![-1, -2, -3, -4]);
        let result = sum(vector, Mode::Neon);
        assert_eq!(result, -10);
    }

    #[test]
    fn signed_int_neon_1_sums_remainder() {
        let vector = SignedIntArray::new(vec![1, 2, 3, 4, 5]);
        let result = sum(vector, Mode::Neon);
        assert_eq!(result, 15);
    }

    #[test]
    fn signed_int_par_1_sums_full_vector() {
        let vector = SignedIntArray::new(vec![1, 2, 3, 4]);
        let result = sum(vector, Mode::ParNeon);
        assert_eq!(result, 10);
    }

    #[test]
    fn signed_int_par_1_sums_remainder() {
        let vector = SignedIntArray::new(vec![1, 2, 3, 4, 5, 6, 7]);
        let result = sum(vector, Mode::ParNeon);
        assert_eq!(result, 28);
    }

    #[test]
    fn test_reductive_scalar_min() {
        let vector = SignedIntArray::new(vec![1, 2, 3]);
        let result = min(vector, Mode::Normal);
        assert_eq!(result, 1);
    }

    #[test]
    fn test_reductive_neon_min() {
        let vector = SignedIntArray::new(vec![1, 2, 3, 4]);
        let result = min(vector, Mode::Neon);
        assert_eq!(result, 1);
    }

    #[test]
    fn test_reductive_neon_min_with_remainder() {
        let vector = SignedIntArray::new(vec![1, 2, 3, 0, 4, 5]);
        let result = min(vector, Mode::Neon);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_reductive_neon_par_min() {
        let vector = SignedIntArray::new(vec![1, 2, 3, 0, 4, 5]);
        let result = min(vector, Mode::ParNeon);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_reductive_neon_par_min_with_remainder() {
        let vector = SignedIntArray::new(vec![1, 2, 3, 0, 4, 5, 1, 10]);
        let result = min(vector, Mode::ParNeon);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_reductive_scalar_max() {
        let vector = SignedIntArray::new(vec![1, 2, 3]);
        let result = max(vector, Mode::Normal);
        assert_eq!(result, 3);
    }

    #[test]
    fn test_reduction_neon_max() {
        let vector = SignedIntArray::new(vec![1, 2, 10, 3, 4]);
        let result = max(vector, Mode::Neon);
        assert_eq!(result, 10);
    }

    #[test]
    fn test_reduction_neon_max_with_remainder() {
        let vector = SignedIntArray::new(vec![1, 2, 10, 3, 4, 5, 6, 7]);
        let result = max(vector, Mode::Neon);
        assert_eq!(result, 10);
    }

    #[test]
    fn test_reduction_neon_par_max() {
        let vector = SignedIntArray::new(vec![1, 2, 10, 3, 4, 5]);
        let result = max(vector, Mode::ParNeon);
        assert_eq!(result, 10);
    }

    #[test]
    fn test_reduction_neon_par_max_with_remainder() {
        let vector = SignedIntArray::new(vec![1, 2, 10, 3, 4, 5, 6, 7]);
        let result = max(vector, Mode::ParNeon);
        assert_eq!(result, 10);
    }
}
