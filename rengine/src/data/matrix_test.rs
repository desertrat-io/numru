#[cfg(test)]
mod tests {
    use crate::data::array::Array;
    use crate::data::matrix::Matrix;

    #[test]
    fn reads_row_major_elements() {
        let matrix = Matrix::new(3, 2, Array::new(vec![1_i32, 2, 3, 4, 5, 6]));

        assert_eq!(matrix.get_or_else(0, 0), Some(&1));
        assert_eq!(matrix.get_or_else(2, 0), Some(&3));
        assert_eq!(matrix.get_or_else(0, 1), Some(&4));
        assert_eq!(matrix.get_or_else(2, 1), Some(&6));
    }

    #[test]
    fn returns_each_row_as_a_contiguous_slice() {
        let matrix = Matrix::new(3, 2, Array::new(vec![1_i32, 2, 3, 4, 5, 6]));

        assert_eq!(matrix.row(0), Some(&[1, 2, 3][..]));
        assert_eq!(matrix.row(1), Some(&[4, 5, 6][..]));
    }

    #[test]
    fn row_returns_none_when_out_of_bounds() {
        let matrix = Matrix::new(2, 2, Array::new(vec![1_u8, 2, 3, 4]));

        assert_eq!(matrix.row(2), None);
        assert_eq!(matrix.row(usize::MAX), None);
    }

    #[test]
    fn zero_sized_matrices_have_no_rows() {
        let no_rows = Matrix::new(3, 0, Array::<i32>::new(Vec::new()));
        let no_columns = Matrix::new(0, 3, Array::<i32>::new(Vec::new()));

        assert_eq!(no_rows.rows(), 0);
        assert_eq!(no_rows.row(0), None);
        assert_eq!(no_columns.cols(), 0);
        assert_eq!(no_columns.row(0), Some(&[][..]));
        assert_eq!(no_columns.row(3), None);
    }

    #[test]
    fn reports_dimensions() {
        let matrix = Matrix::new(4, 3, Array::<f64>::default_padded(12));

        assert_eq!(matrix.cols(), 4);
        assert_eq!(matrix.rows(), 3);
    }

    #[test]
    fn out_of_bounds_coordinates_return_none() {
        let matrix = Matrix::new(2, 2, Array::new(vec![1_u8, 2, 3, 4]));

        assert_eq!(matrix.get_or_else(2, 0), None);
        assert_eq!(matrix.get_or_else(0, 2), None);
        assert_eq!(matrix.get_or_else(usize::MAX, 0), None);
        assert_eq!(matrix.get_or_else(0, usize::MAX), None);
    }

    #[test]
    #[should_panic(expected = "matrix dimensions do not match data length")]
    fn rejects_dimension_and_storage_mismatch() {
        let _ = Matrix::new(3, 3, Array::new(vec![0_i32; 8]));
    }

    #[test]
    #[should_panic(expected = "matrix dimensions overflow")]
    fn rejects_dimension_product_overflow() {
        let _ = Matrix::new(usize::MAX, 2, Array::<i32>::new(Vec::new()));
    }

    #[test]
    fn supports_floating_point_elements() {
        let matrix = Matrix::new(2, 1, Array::new(vec![1.5_f32, -2.25]));

        assert_eq!(matrix.get_or_else(0, 0), Some(&1.5));
        assert_eq!(matrix.get_or_else(1, 0), Some(&-2.25));
    }

    #[test]
    fn supports_boolean_elements() {
        let matrix = Matrix::new(2, 1, Array::new(vec![true, false]));

        assert_eq!(matrix.get_or_else(0, 0), Some(&true));
        assert_eq!(matrix.get_or_else(1, 0), Some(&false));
    }
}
