use crate::data::array::Array;

// don't use the Default trait, it can result in Tensors we don't allow
#[derive(Debug, Clone)]
pub struct Tensor<T> {
    data: Array<T>,
    shape: Vec<usize>,
    rank: usize,
}

impl<T: Default + Clone> Tensor<T> {
    pub fn new(data: Array<T>, shape: &[usize]) -> Self {
        let actual_length = data.len();
        if actual_length == 0 && shape.contains(&0) {
            return Self {
                data,
                shape: shape.to_vec(),
                rank: shape.len(),
            };
        }
        if !Tensor::valid_shape(&data, shape) {
            panic!("Invalid shape for tensor")
        }

        Self {
            data,
            shape: shape.to_vec(),
            rank: shape.len(),
        }
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn flatten(&self) -> Array<T> {
        // the backing Array is already in row major order
        // so cloning it and returning that performs the flatten operation for free
        self.data.clone()
    }

    fn valid_shape(data: &Array<T>, shape: &[usize]) -> bool {
        let actual_length = data.len();
        if actual_length > 0 && shape.contains(&0) {
            return false;
        }
        let mut final_length: usize = 1;
        for dim in shape {
            final_length = final_length.checked_mul(*dim).expect("shape overflow");
        }

        final_length == actual_length
    }
}

pub type SignedF32Tensor = Tensor<f32>;
pub type SignedF64Tensor = Tensor<f64>;
pub type SignedInt32Tensor = Tensor<i32>;
pub type BoolArray = Tensor<bool>;
