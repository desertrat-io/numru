use std::usize;

pub const CONTIGUOUS_STRIDE: i32 = 1;

#[derive(Debug, Clone, Default)]
pub struct Array<T> {
    // TODO: was reading around, consider Box<[T]> instead of Vec<T> for memory utilization optimization
    data: Vec<T>,
}

impl<T: Default + Clone> Array<T> {
    pub fn new(data: Vec<T>) -> Self {
        Self { data }
    }

    pub fn slice(&self) -> &[T] {
        &self.data
    }

    pub fn mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    pub fn len(&self) -> usize {
        self.data.len()
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

pub trait NumericArrayExtras<T>: ArrayExtras<T> {
    fn range(start: T, end: T, step: T, exclusive: bool) -> Self;
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
macro_rules! step_check {
    ($step:expr) => {
        if $step == 0.0 {
            panic!("Invalid step value, 0 not allowed")
        }
    };
}
macro_rules! float_arr_decorator {
    ($array_type:ty, $inner_type:ty) => {
        impl NumericArrayExtras<$inner_type> for $array_type {
            fn range(
                start: $inner_type,
                end: $inner_type,
                step: $inner_type,
                exclusive: bool,
            ) -> Self {
                step_check!(step);
                assert!(start.is_finite() && end.is_finite() && step.is_finite());

                // this is for floats only, go ahead and expand everything to f64 regardless of base type
                // capacity has an edge case that uses floor + 1 or ceil for inclusive and exclusive respectively

                let capacity = if exclusive {
                    ((end as f64 - start as f64) / step as f64).ceil()
                } else {
                    ((end as f64 - start as f64) / step as f64).floor() + 1.0
                };

                if !capacity.is_finite() {
                    panic!("Cannot allocate space for infinite or overflowed capacity for vector");
                }

                let mut accumulator: Vec<$inner_type> = Vec::with_capacity(capacity as usize);
                let mut i: $inner_type = 0.0;
                loop {
                    let val: $inner_type = start + i * step;
                    // we can safely skip step == 0 since step_check already does that for us

                    // check exclusive
                    if (exclusive) {
                        if (val >= end && step > 0.0) || (val <= end && step < 0.0) {
                            break;
                        }
                    }
                    // then inclusive
                    else {
                        if (val > end && step > 0.0) || (val < end && step < 0.0) {
                            break;
                        }
                    }
                    accumulator.push(val as $inner_type);
                    i += 1.0;
                }
                Array::new(accumulator)
            }
        }
    };
}

pub type SignedF32Array = Array<f32>;
arr_decorator!(SignedF32Array, f32, 0.0_f32, 1.0_f32);
float_arr_decorator!(SignedF32Array, f32);
pub type SignedF64Array = Array<f64>;
arr_decorator!(SignedF64Array, f64, 0.0_f64, 1.0_f64);
float_arr_decorator!(SignedF64Array, f64);
//TODO: consider bit packing for bools, may need special handling.
pub type BoolArray = Array<bool>;
arr_decorator!(BoolArray, bool, false, true);
pub type SignedInt32Array = Array<i32>;
arr_decorator!(SignedInt32Array, i32, 0_i32, 1_i32);
