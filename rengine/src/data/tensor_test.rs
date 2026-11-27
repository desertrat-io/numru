use crate::data::array::{BoolArray, SignedF32Array, SignedF64Array, SignedInt32Array};
use crate::data::tensor::{
    BoolArray as BoolTensor, SignedF32Tensor, SignedF64Tensor, SignedInt32Tensor,
};

#[test]
fn creates_f32_scalar() {
    let values = [1.25_f32];
    let tensor = SignedF32Tensor::new(SignedF32Array::new(values.to_vec()), &[]);

    assert_eq!(tensor.rank(), 0);
    assert!(tensor.shape().is_empty());
}

#[test]
fn creates_f64_vector() {
    let values = [-2.5_f64, 0.0, 3.5];
    let tensor = SignedF64Tensor::new(SignedF64Array::new(values.to_vec()), &[3]);

    assert_eq!(tensor.rank(), 1);
    assert_eq!(tensor.shape(), [3]);
}

#[test]
fn creates_i32_matrix() {
    let values = [i32::MIN, -1, 0, 1, 2, i32::MAX];
    let tensor = SignedInt32Tensor::new(SignedInt32Array::new(values.to_vec()), &[2, 3]);

    assert_eq!(tensor.rank(), 2);
    assert_eq!(tensor.shape(), [2, 3]);
}

#[test]
fn creates_rank_four_bool_tensor_with_singleton_axes() {
    let values = [true, false, false, true];
    let tensor = BoolTensor::new(BoolArray::new(values.to_vec()), &[1, 2, 1, 2]);

    assert_eq!(tensor.rank(), 4);
    assert_eq!(tensor.shape(), [1, 2, 1, 2]);
}

const RANK_TWELVE_SHAPE: [usize; 12] = [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3];

#[test]
fn creates_rank_twelve_f32_tensor() {
    let data = SignedF32Array::new(vec![1.25, -2.5, 0.0, 3.75, 4.0, 5.5]);
    let tensor = SignedF32Tensor::new(data, &RANK_TWELVE_SHAPE);

    assert_eq!(tensor.shape(), RANK_TWELVE_SHAPE);
    assert_eq!(tensor.rank(), 12);
}

#[test]
fn creates_rank_twelve_f64_tensor() {
    let data = SignedF64Array::new(vec![1.25, -2.5, 0.0, 3.75, 4.0, 5.5]);
    let tensor = SignedF64Tensor::new(data, &RANK_TWELVE_SHAPE);

    assert_eq!(tensor.shape(), RANK_TWELVE_SHAPE);
    assert_eq!(tensor.rank(), 12);
}

#[test]
fn creates_rank_twelve_i32_tensor() {
    let data = SignedInt32Array::new(vec![1, 2, 3, 4, 5, 6]);
    let tensor = SignedInt32Tensor::new(data, &RANK_TWELVE_SHAPE);

    assert_eq!(tensor.shape(), RANK_TWELVE_SHAPE);
    assert_eq!(tensor.rank(), 12);
}

#[test]
fn creates_rank_twelve_bool_tensor() {
    let data = BoolArray::new(vec![true, false, false, true, true, false]);
    let tensor = BoolTensor::new(data, &RANK_TWELVE_SHAPE);

    assert_eq!(tensor.shape(), RANK_TWELVE_SHAPE);
    assert_eq!(tensor.rank(), 12);
}

#[test]
fn creates_empty_vector() {
    let tensor = SignedF32Tensor::new(SignedF32Array::new(vec![]), &[0]);

    assert_eq!(tensor.rank(), 1);
    assert_eq!(tensor.shape(), [0]);
}

#[test]
fn preserves_rank_of_empty_tensors() {
    for shape in [[0, 2, 3], [2, 0, 3], [2, 3, 0], [0, 0, 0]] {
        let tensor = BoolTensor::new(BoolArray::new(vec![]), &shape);

        assert_eq!(tensor.shape(), shape);
        assert_eq!(tensor.rank(), shape.len(), "shape {shape:?}");
    }
}

#[test]
fn zero_axis_makes_an_otherwise_overflowing_shape_empty() {
    for shape in [[0, usize::MAX, 2], [usize::MAX, 2, 0]] {
        let tensor = SignedInt32Tensor::new(SignedInt32Array::new(vec![]), &shape);

        assert_eq!(tensor.shape(), shape);
    }
}

#[test]
#[should_panic(expected = "Invalid shape for tensor")]
fn rejects_scalar_with_no_data() {
    SignedF32Tensor::new(SignedF32Array::new(vec![]), &[]);
}

#[test]
#[should_panic(expected = "Invalid shape for tensor")]
fn rejects_scalar_with_extra_data() {
    BoolTensor::new(BoolArray::new(vec![true, false]), &[]);
}

#[test]
#[should_panic(expected = "Invalid shape for tensor")]
fn rejects_empty_data_for_nonempty_shape() {
    SignedF64Tensor::new(SignedF64Array::new(vec![]), &[1, 1]);
}

#[test]
#[should_panic(expected = "Invalid shape for tensor")]
fn rejects_data_for_zero_sized_shape() {
    SignedInt32Tensor::new(SignedInt32Array::new(vec![1]), &[usize::MAX, 2, 0]);
}

#[test]
#[should_panic(expected = "Invalid shape for tensor")]
fn rejects_data_shorter_than_shape_product() {
    SignedF32Tensor::new(SignedF32Array::new(vec![0.0; 5]), &[2, 3]);
}

#[test]
#[should_panic(expected = "Invalid shape for tensor")]
fn rejects_data_longer_than_shape_product() {
    SignedF64Tensor::new(SignedF64Array::new(vec![0.0; 7]), &[2, 3]);
}

#[test]
#[should_panic(expected = "Invalid shape for tensor")]
fn largest_representable_product_is_a_length_mismatch() {
    SignedInt32Tensor::new(SignedInt32Array::new(vec![]), &[usize::MAX, 1]);
}

#[test]
#[should_panic(expected = "shape overflow")]
fn rejects_product_that_would_wrap_to_zero() {
    BoolTensor::new(BoolArray::new(vec![]), &[usize::MAX / 2 + 1, 2]);
}

#[test]
#[should_panic(expected = "shape overflow")]
fn rejects_overflow_in_a_later_axis_with_nonempty_data() {
    SignedF64Tensor::new(SignedF64Array::new(vec![1.0]), &[1, usize::MAX, 2]);
}

#[test]
fn flatten_preserves_logical_row_major_order_across_ranks() {
    let values = [101, 102, 103, 201, 202, 203, 301, 302, 303, 401, 402, 403];
    let shapes: &[&[usize]] = &[&[3, 4], &[2, 2, 3], &[2, 1, 2, 3]];

    for &shape in shapes {
        let tensor = SignedInt32Tensor::new(SignedInt32Array::new(values.to_vec()), shape);
        let flattened: SignedInt32Array = tensor.flatten();

        assert_eq!(flattened.slice(), values, "shape {shape:?}");
        assert_eq!(tensor.shape(), shape);
        assert_eq!(tensor.rank(), shape.len());
        assert_eq!(tensor.flatten().slice(), values, "shape {shape:?}");
    }
}

#[test]
fn flatten_returns_one_element_for_a_scalar() {
    let tensor = SignedF64Tensor::new(SignedF64Array::new(vec![-2.5]), &[]);
    let flattened: SignedF64Array = tensor.flatten();

    assert_eq!(flattened.slice(), [-2.5]);
    assert!(tensor.shape().is_empty());
}

#[test]
fn flatten_returns_an_empty_array_for_a_zero_sized_axis() {
    let tensor = BoolTensor::new(BoolArray::new(vec![]), &[2, 0, 3]);
    let flattened: BoolArray = tensor.flatten();

    assert!(flattened.slice().is_empty());
    assert_eq!(tensor.shape(), [2, 0, 3]);
}

#[test]
fn flatten_copies_an_already_flat_tensor() {
    let values = [-9, 0, 4, 17];
    let tensor = SignedInt32Tensor::new(SignedInt32Array::new(values.to_vec()), &[4]);
    let mut flattened: SignedInt32Array = tensor.flatten();
    assert_eq!(flattened.slice(), values);

    flattened.mut_slice()[0] = 99;
    let second_copy: SignedInt32Array = tensor.flatten();

    assert_eq!(tensor.shape(), [4]);
    assert_eq!(second_copy.slice(), values);
    assert_ne!(flattened.slice().as_ptr(), second_copy.slice().as_ptr());

    drop(tensor);
    assert_eq!(flattened.slice(), [99, 0, 4, 17]);
    assert_eq!(second_copy.slice(), values);
}

#[test]
#[should_panic(expected = "shape overflow")]
fn rejects_overflow_that_would_wrap_to_the_buffer_length() {
    // The shape product is usize::MAX + 3, which wraps to two.
    SignedInt32Tensor::new(
        SignedInt32Array::new(vec![10, 20]),
        &[usize::MAX / 2 + 2, 2],
    );
}

#[test]
fn flatten_returns_an_empty_array_for_an_empty_vector() {
    let tensor = SignedF32Tensor::new(SignedF32Array::new(vec![]), &[0]);
    let flattened: SignedF32Array = tensor.flatten();

    assert!(flattened.slice().is_empty());
    assert_eq!(tensor.shape(), [0]);
    assert_eq!(tensor.rank(), 1);
}

#[test]
fn flatten_handles_empty_shapes_with_otherwise_overflowing_extents() {
    for shape in [[0, usize::MAX, 2], [usize::MAX, 0, 2], [usize::MAX, 2, 0]] {
        let tensor = SignedInt32Tensor::new(SignedInt32Array::new(vec![]), &shape);
        let flattened: SignedInt32Array = tensor.flatten();

        assert!(flattened.slice().is_empty(), "shape {shape:?}");
        assert_eq!(tensor.shape(), shape);
        assert_eq!(tensor.rank(), shape.len());
    }
}

#[test]
fn flatten_preserves_order_for_rank_twelve() {
    let values = [11, 12, 13, 21, 22, 23];
    let tensor = SignedInt32Tensor::new(SignedInt32Array::new(values.to_vec()), &RANK_TWELVE_SHAPE);
    let flattened: SignedInt32Array = tensor.flatten();

    assert_eq!(flattened.slice(), values);
    assert_eq!(tensor.shape(), RANK_TWELVE_SHAPE);
    assert_eq!(tensor.rank(), 12);
}

fn assert_flattened_copy<T: Default + Clone + PartialEq + std::fmt::Debug>(
    tensor: &crate::data::tensor::Tensor<T>,
    shape: &[usize],
    values: &[T],
    replacement: T,
) {
    let mut flattened = tensor.flatten();

    assert_eq!(flattened.shape(), [values.len()]);
    assert_eq!(flattened.slice(), values);
    flattened.mut_slice()[0] = replacement;

    assert_eq!(tensor.flatten().slice(), values);
    assert_eq!(tensor.shape(), shape);
    assert_eq!(tensor.rank(), shape.len());
}

#[test]
fn flatten_copies_high_rank_f32_tensor() {
    let shape = [2, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 2];
    let values = [-3.5, -2.0, -1.25, 0.0, 0.5, 1.0, 1.25, 2.0, 3.5, 4.0, 5.5, 8.0];
    let tensor = SignedF32Tensor::new(SignedF32Array::new(values.to_vec()), &shape);

    assert_flattened_copy(&tensor, &shape, &values, 99.0);
}

#[test]
fn flatten_copies_high_rank_f64_tensor() {
    let shape = [1, 2, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 2];
    let values = [
        -f64::MAX,
        -2.5,
        -1.0,
        0.0,
        f64::EPSILON,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        f64::MAX,
    ];
    let tensor = SignedF64Tensor::new(SignedF64Array::new(values.to_vec()), &shape);

    assert_flattened_copy(&tensor, &shape, &values, 99.0);
}

#[test]
fn flatten_copies_high_rank_i32_tensor() {
    let shape = [2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2];
    let values = [i32::MIN, -41, -7, 0, 2, 8, 31, i32::MAX];
    let tensor = SignedInt32Tensor::new(SignedInt32Array::new(values.to_vec()), &shape);

    assert_flattened_copy(&tensor, &shape, &values, 99);
}

#[test]
fn flatten_copies_high_rank_bool_tensor() {
    let shape = [1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 3];
    let values = [
        true, false, false, true, true, false, false, false, true, false, true, true,
    ];
    let tensor = BoolTensor::new(BoolArray::new(values.to_vec()), &shape);

    assert_flattened_copy(&tensor, &shape, &values, false);
}

#[test]
fn flatten_preserves_owned_matrix_elements() {
    use crate::data::array::Array;
    use crate::data::matrix::Matrix;
    use crate::data::tensor::Tensor;

    let matrices = Array::new(vec![
        Matrix::new(3, 2, Array::new(vec![1, 2, 3, 4, 5, 6])),
        Matrix::new(2, 1, Array::new(vec![7, 8])),
    ]);
    let shape = [1, 2, 1];
    let tensor = Tensor::new(matrices, &shape);
    let mut flattened: Array<Matrix<i32>> = tensor.flatten();

    assert_eq!(flattened.shape(), [2]);
    assert_eq!(tensor.shape(), shape);
    assert_eq!(tensor.rank(), 3);
    let [first, second] = flattened.mut_slice() else {
        panic!("expected two matrices");
    };
    assert_eq!((first.rows(), first.cols()), (2, 3));
    assert_eq!(first.row(0), Some([1, 2, 3].as_slice()));
    assert_eq!(first.row(1), Some([4, 5, 6].as_slice()));
    assert_eq!((second.rows(), second.cols()), (1, 2));
    assert_eq!(second.slice(), [7, 8]);

    first.slice_mut()[0] = 99;
    assert_eq!(tensor.flatten().slice()[0].slice(), [1, 2, 3, 4, 5, 6]);
}

#[test]
fn flatten_preserves_owned_array_elements() {
    use crate::data::array::Array;
    use crate::data::tensor::Tensor;

    let arrays = Array::new(vec![
        Array::new(vec![1.25, -2.5]),
        Array::new(vec![]),
        Array::new(vec![3.75]),
    ]);
    let shape = [1, 3, 1, 1, 1];
    let tensor = Tensor::new(arrays, &shape);
    let mut flattened: Array<Array<f64>> = tensor.flatten();

    assert_eq!(flattened.shape(), [3]);
    assert_eq!(tensor.shape(), shape);
    assert_eq!(tensor.rank(), 5);
    let [first, empty, last] = flattened.mut_slice() else {
        panic!("expected three arrays");
    };
    assert_eq!(first.slice(), [1.25, -2.5]);
    assert_eq!(first.shape(), [2]);
    assert!(empty.slice().is_empty());
    assert_eq!(empty.shape(), [0]);
    assert_eq!(last.slice(), [3.75]);
    assert_eq!(last.shape(), [1]);

    first.mut_slice()[0] = 99.0;
    assert_eq!(tensor.flatten().slice()[0].slice(), [1.25, -2.5]);
}
