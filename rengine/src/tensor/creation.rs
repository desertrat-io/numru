use ndarray::iter::Axes;
use ndarray::ArrayD;

// unless something insane happens, no need to support tensors more than order...255
// same for axes
struct Tensor<'a> {
    data: ArrayD<f64>,
    order: u8,
    axes: Axes<'a, u8>,
}
