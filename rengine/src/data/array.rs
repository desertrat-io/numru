pub const CONTIGUOUS_STRIDE: i32 = 1;

#[derive(Debug, Clone, Default)]
pub struct Array<T> {
    data: Vec<T>,
    len: usize,
}

impl<T: Default + Clone> Array<T> {
    pub fn new(data: Vec<T>) -> Self {
        let len = data.len();
        Self { data, len }
    }

    pub fn slice(&self) -> &[T] {
        &self.data
    }

    pub fn mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn default_padded(len: usize) -> Self {
        Self::new(vec![T::default(); len])
    }
}

pub type SignedF32Array = Array<f32>;
pub type SignedF64Array = Array<f64>;
pub type BoolArray = Array<bool>;
pub type SignedIntArray = Array<i32>;
