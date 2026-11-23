use crate::data::array::Array;
use crate::data::tensor::{BoolArray, SignedF32Tensor, SignedF64Tensor, SignedInt32Tensor};

#[test]
fn creates_f32_scalar() {
    let values = [1.25_f32];
    let tensor = SignedF32Tensor::new(Array::new(values.to_vec()), &[]);

    assert_eq!(tensor.rank(), 0);
    assert!(tensor.shape().is_empty());
}

#[test]
fn creates_f64_vector() {
    let values = [-2.5_f64, 0.0, 3.5];
    let tensor = SignedF64Tensor::new(Array::new(values.to_vec()), &[3]);

    assert_eq!(tensor.rank(), 1);
    assert_eq!(tensor.shape(), [3]);
}

#[test]
fn creates_i32_matrix() {
    let values = [i32::MIN, -1, 0, 1, 2, i32::MAX];
    let tensor = SignedInt32Tensor::new(Array::new(values.to_vec()), &[2, 3]);

    assert_eq!(tensor.rank(), 2);
    assert_eq!(tensor.shape(), [2, 3]);
}

#[test]
fn creates_rank_four_bool_tensor_with_singleton_axes() {
    let values = [true, false, false, true];
    let tensor = BoolArray::new(Array::new(values.to_vec()), &[1, 2, 1, 2]);

    assert_eq!(tensor.rank(), 4);
    assert_eq!(tensor.shape(), [1, 2, 1, 2]);
}

const RANK_TWELVE_SHAPE: [usize; 12] = [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3];

#[test]
fn creates_rank_twelve_f32_tensor() {
    let data = Array::new(vec![1.25, -2.5, 0.0, 3.75, 4.0, 5.5]);
    let tensor = SignedF32Tensor::new(data, &RANK_TWELVE_SHAPE);

    assert_eq!(tensor.shape(), RANK_TWELVE_SHAPE);
    assert_eq!(tensor.rank(), 12);
}

#[test]
fn creates_rank_twelve_f64_tensor() {
    let data = Array::new(vec![1.25, -2.5, 0.0, 3.75, 4.0, 5.5]);
    let tensor = SignedF64Tensor::new(data, &RANK_TWELVE_SHAPE);

    assert_eq!(tensor.shape(), RANK_TWELVE_SHAPE);
    assert_eq!(tensor.rank(), 12);
}

#[test]
fn creates_rank_twelve_i32_tensor() {
    let data = Array::new(vec![1, 2, 3, 4, 5, 6]);
    let tensor = SignedInt32Tensor::new(data, &RANK_TWELVE_SHAPE);

    assert_eq!(tensor.shape(), RANK_TWELVE_SHAPE);
    assert_eq!(tensor.rank(), 12);
}

#[test]
fn creates_rank_twelve_bool_tensor() {
    let data = Array::new(vec![true, false, false, true, true, false]);
    let tensor = BoolArray::new(data, &RANK_TWELVE_SHAPE);

    assert_eq!(tensor.shape(), RANK_TWELVE_SHAPE);
    assert_eq!(tensor.rank(), 12);
}

#[test]
fn creates_empty_vector() {
    let tensor = SignedF32Tensor::new(Array::new(vec![]), &[0]);

    assert_eq!(tensor.rank(), 1);
    assert_eq!(tensor.shape(), [0]);
}

#[test]
fn preserves_rank_of_empty_tensors() {
    for shape in [[0, 2, 3], [2, 0, 3], [2, 3, 0], [0, 0, 0]] {
        let tensor = BoolArray::new(Array::new(vec![]), &shape);

        assert_eq!(tensor.shape(), shape);
        assert_eq!(tensor.rank(), shape.len(), "shape {shape:?}");
    }
}

#[test]
fn zero_axis_makes_an_otherwise_overflowing_shape_empty() {
    for shape in [[0, usize::MAX, 2], [usize::MAX, 2, 0]] {
        let tensor = SignedInt32Tensor::new(Array::new(vec![]), &shape);

        assert_eq!(tensor.shape(), shape);
    }
}

#[test]
#[should_panic(expected = "Invalid shape for tensor")]
fn rejects_scalar_with_no_data() {
    SignedF32Tensor::new(Array::new(vec![]), &[]);
}

#[test]
#[should_panic(expected = "Invalid shape for tensor")]
fn rejects_scalar_with_extra_data() {
    BoolArray::new(Array::new(vec![true, false]), &[]);
}

#[test]
#[should_panic(expected = "Invalid shape for tensor")]
fn rejects_empty_data_for_nonempty_shape() {
    SignedF64Tensor::new(Array::new(vec![]), &[1, 1]);
}

#[test]
#[should_panic(expected = "Invalid shape for tensor")]
fn rejects_data_for_zero_sized_shape() {
    SignedInt32Tensor::new(Array::new(vec![1]), &[usize::MAX, 2, 0]);
}

#[test]
#[should_panic(expected = "Invalid shape for tensor")]
fn rejects_data_shorter_than_shape_product() {
    SignedF32Tensor::new(Array::new(vec![0.0; 5]), &[2, 3]);
}

#[test]
#[should_panic(expected = "Invalid shape for tensor")]
fn rejects_data_longer_than_shape_product() {
    SignedF64Tensor::new(Array::new(vec![0.0; 7]), &[2, 3]);
}

#[test]
#[should_panic(expected = "Invalid shape for tensor")]
fn largest_representable_product_is_a_length_mismatch() {
    SignedInt32Tensor::new(Array::new(vec![]), &[usize::MAX, 1]);
}

#[test]
#[should_panic(expected = "shape overflow")]
fn rejects_product_that_would_wrap_to_zero() {
    BoolArray::new(Array::new(vec![]), &[usize::MAX / 2 + 1, 2]);
}

#[test]
#[should_panic(expected = "shape overflow")]
fn rejects_overflow_in_a_later_axis_with_nonempty_data() {
    SignedF64Tensor::new(Array::new(vec![1.0]), &[1, usize::MAX, 2]);
}
