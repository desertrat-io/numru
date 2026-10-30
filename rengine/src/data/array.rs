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

pub trait ArrayExtras<T> {
    fn zeros(len: usize) -> Self;

    fn ones(len: usize) -> Self;

    fn fill(len: usize, value: T) -> Self;
}

// small little hack to decorate typed zeros and ones functions onto the arrays
macro_rules! arr_decorator {
    ($array_type:ty, $inner_type:ty, $zero_val:expr, $ones_val:expr) => {
        // all of these return the same array type we're decorating, no need for weird
        // type management
        impl ArrayExtras<$inner_type> for $array_type {
            fn zeros(len: usize) -> Self {
                Array::new(vec![$zero_val; len])
            }

            fn ones(len: usize) -> Self {
                Array::new(vec![$ones_val; len])
            }

            fn fill(len: usize, value: $inner_type) -> Self {
                Array::new(vec![value; len])
            }
        }
    };
}

pub type SignedF32Array = Array<f32>;
arr_decorator!(SignedF32Array, f32, 0.0_f32, 1.0_f32);
pub type SignedF64Array = Array<f64>;
arr_decorator!(SignedF64Array, f64, 0.0_f64, 1.0_f64);
//TODO: consider bit packing for bools, may need special handling.
pub type BoolArray = Array<bool>;
arr_decorator!(BoolArray, bool, false, true);
pub type SignedInt32Array = Array<i32>;
arr_decorator!(SignedInt32Array, i32, 0_i32, 1_i32);
