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

pub trait ArrayExtras {
    fn zeros(len: usize) -> Self;

    fn ones(len: usize) -> Self;
}

macro_rules! arr_extras {
    ($x:ty, $y:expr, $z:expr) => {
        impl ArrayExtras for $x {
            fn zeros(len: usize) -> Self {
                Array::new(vec![$y; len])
            }

            fn ones(len: usize) -> Self {
                Array::new(vec![$z; len])
            }
        }
    };
}

pub type SignedF32Array = Array<f32>;
arr_extras!(SignedF32Array, 0.0_f32, 1.0_f32);
pub type SignedF64Array = Array<f64>;
arr_extras!(SignedF64Array, 0.0_f64, 1.0_f64);
pub type BoolArray = Array<bool>;
arr_extras!(BoolArray, false, true);
pub type SignedInt32Array = Array<i32>;
arr_extras!(SignedInt32Array, 0_i32, 1_i32);

#[cfg(test)]
#[path = "array_test.rs"]
mod array_test;
