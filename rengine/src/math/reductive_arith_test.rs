#[cfg(test)]
mod tests {
    use crate::data::array::IntArray;
    use crate::math::reductive_arith::reductive_sum;
    use crate::matrix::ops::Mode;

    #[test]
    fn test_reductive_scalar_add_32() {
        let vector = IntArray::new(vec![1, 2, 3]);
        let result = reductive_sum(vector, Mode::Normal);
        assert_eq!(result, 6);
        let vector = IntArray::new(vec![]);
        let result = reductive_sum(vector, Mode::Normal);
        assert_eq!(result, 0);
    }

    #[test]
    fn signed_int_neon_1_sums_full_vector() {
        let vector = IntArray::new(vec![1, 2, 3, 4]);
        let result = reductive_sum(vector, Mode::Neon);
        assert_eq!(result, 10);
    }

    #[test]
    fn signed_int_neon_1_sums_remainder() {
        let vector = IntArray::new(vec![1, 2, 3, 4, 5]);
        let result = reductive_sum(vector, Mode::Neon);
        assert_eq!(result, 15);
    }

    #[test]
    fn signed_int_par_1_sums_full_vector() {
        let vector = IntArray::new(vec![1, 2, 3, 4]);
        let result = reductive_sum(vector, Mode::ParNeon);
        assert_eq!(result, 10);
    }

    #[test]
    fn signed_int_par_1_sums_remainder() {
        let vector = IntArray::new(vec![1, 2, 3, 4, 5]);
        let result = reductive_sum(vector, Mode::ParNeon);
        assert_eq!(result, 15);
    }
}
