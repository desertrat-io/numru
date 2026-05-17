#[cfg(test)]
mod tests {
    use crate::data::array::Array;
    use crate::logic::compare;
    use crate::matrix::ops::Mode;

    fn get_compare_fixture() -> (Array, Array) {
        (
            Array::new(vec![3.0, 2.0, 2.0, -1.0, 0.0]),
            Array::new(vec![2.0, 2.0, 3.0, -2.0, 1.0]),
        )
    }

    #[test]
    fn test_gt_32() {
        let (left, right) = get_compare_fixture();
        let result = compare::gt(left, right, Mode::Normal);
        assert_eq!(result.slice(), [1.0, 0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_gt_32_neon() {
        let (left, right) = get_compare_fixture();
        let result = compare::gt(left, right, Mode::Neon);
        assert_eq!(result.slice(), [1.0, 0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_gt_32_par_neon() {
        let (left, right) = get_compare_fixture();
        let result = compare::gt(left, right, Mode::ParNeon);
        assert_eq!(result.slice(), [1.0, 0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_eq_32() {
        let (left, right) = get_compare_fixture();
        let result = compare::eq(left, right, Mode::Normal);
        assert_eq!(result.slice(), [0.0, 1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_eq_32_neon() {
        let (left, right) = get_compare_fixture();
        let result = compare::eq(left, right, Mode::Neon);
        assert_eq!(result.slice(), [0.0, 1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_eq_32_par_neon_panics() {
        let (left, right) = get_compare_fixture();
        let result = compare::eq(left, right, Mode::ParNeon);
        assert_eq!(result.slice(), [0.0, 1.0, 0.0, 0.0, 0.0]);
    }
}
