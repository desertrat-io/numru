#[cfg(test)]
mod tests {
    use crate::data::array::BoolArray;
    use crate::logic::boolean;
    use crate::matrix::ops::Mode;

    #[test]
    fn test_not_32_neon() {
        let left = BoolArray::new(vec![true, false, true, false, true]);
        let result = boolean::not(left, Mode::Neon);
        assert_eq!(result.slice(), vec![false, true, false, true, false]);
    }

    #[test]
    fn test_not_32_par_neon() {
        let left = BoolArray::new(vec![true, false, true, false, true]);
        let result = boolean::not(left, Mode::ParNeon);
        assert_eq!(result.slice(), vec![false, true, false, true, false]);
    }

    #[test]
    fn test_and_32_neon() {
        let left = BoolArray::new(vec![true, false, true, false, true]);
        let right = BoolArray::new(vec![true, true, false, true, false]);
        let result = boolean::and(left, right, Mode::Neon);
        assert_eq!(result.slice(), vec![true, false, false, false, false]);
    }

    #[test]
    fn test_and_32_par_neon() {
        let left = BoolArray::new(vec![true, false, true, false, true]);
        let right = BoolArray::new(vec![true, true, false, true, false]);
        let result = boolean::and(left, right, Mode::ParNeon);
        assert_eq!(result.slice(), vec![true, false, false, false, false]);
    }
}
