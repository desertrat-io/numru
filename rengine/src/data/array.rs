pub const CONTIGUOUS_STRIDE: i32 = 1;

#[derive(Debug, Clone, Default)]
pub struct Array<T> {
    // TODO: was reading around, consider Box<[T]> instead of Vec<T> for memory utilization optimization
    data: Vec<T>,
    shape: Vec<usize>,
}

impl<T: Default + Clone> Array<T> {
    pub fn new(data: Vec<T>) -> Self {
        // using this constructor can only create a 1d vector, so the shape is a single scalar vector representing the length
        let shape = vec![data.len()];
        Self { data, shape }
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

    pub fn shape(&self) -> &[usize] {
        self.shape.as_slice()
    }

    pub fn default_padded(len: usize) -> Self {
        Self::new(vec![T::default(); len])
    }

    pub fn reshape_cp(original: &Array<T>, shape: &[usize]) -> Self {
        // 0 element 1d vector
        let actual_length = original.len();
        if actual_length == 0 && shape.contains(&0) {
            return Self {
                data: vec![],
                shape: shape.to_vec(),
            };
        }
        if actual_length > 0 && shape.contains(&0) {
            panic!("Invalid shape, vector is not empty")
        }
        let mut final_length: usize = 1;
        for dim in shape {
            final_length = final_length.checked_mul(*dim).expect("shape overflow");
        }

        if final_length != actual_length {
            panic!("Invalid shape, vector length does not match shape requirements")
        }
        Self {
            data: original.data.clone(),
            shape: shape.to_vec(),
        }
    }
}

pub trait ArrayExtras<T> {
    fn zeros(len: usize) -> Self;

    fn ones(len: usize) -> Self;

    fn fill(len: usize, value: T) -> Self;
}

pub trait NumericArrayExtras<T>: ArrayExtras<T> {
    fn range(start: T, end: T, step: T, exclusive: bool) -> Self;

    fn linspace(start: T, end: T, num_elements: usize) -> Self;
}

macro_rules! step_check {
    ($step:expr) => {
        if $step == 0.0 {
            panic!("Invalid step value, 0 not allowed")
        }
    };
}

// not needed for all array types, restrict it to only those that want to implement it, plus the
// representation of "zero" and "one" is different. avoid extra allocations and casts when unecessary
// let the compiler optimize
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

macro_rules! int_arr_decorator {
    ($array_type:ty, $inner_type:ty) => {
        impl NumericArrayExtras<$inner_type> for $array_type {
            fn range(
                start: $inner_type,
                end: $inner_type,
                step: $inner_type,
                exclusive: bool,
            ) -> Self {
                todo!()
            }
            fn linspace(start: $inner_type, end: $inner_type, num_elements: usize) -> Self {
                todo!()
            }
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
                // TODO: wouldn't a for loop be easier to read? Maybe more casts but I think maintainability is better
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

            fn linspace(start: $inner_type, end: $inner_type, num_elements: usize) -> Self {
                // explicitly handling 0 and 1 for num_elements. negative num_elements don't make sense
                let start = start as f64;
                let end = end as f64;
                let result = if num_elements == 0 {
                    Array::new(vec![])
                } else if num_elements == 1 {
                    Array::new(vec![start as $inner_type])
                } else {
                    let reciprocal = 1.0 / (num_elements as f64 - 1.0);
                    let end_gap = end - start;
                    let is_gap_finite = end_gap.is_finite();
                    let mut accumulator: Vec<$inner_type> = Vec::with_capacity(num_elements);
                    // need to handle some overflows, so we'll be explicit with first and end appending
                    // thus only calculating the intermediate
                    accumulator.push(start as $inner_type);
                    // explicit push of start and end, just loop over intermediate
                    for i in 1..num_elements - 1 {
                        let diff_delta = reciprocal * i as f64;
                        if is_gap_finite {
                            accumulator.push((start + diff_delta * end_gap) as $inner_type);
                        } else {
                            accumulator.push(
                                ((1.0 - diff_delta) * start + diff_delta * end) as $inner_type,
                            );
                        }
                    }
                    accumulator.push(end as $inner_type);
                    Array::new(accumulator)
                };

                result
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
// could also remove the need for a whole decorator if a bool is now numeric? maybe
pub type BoolArray = Array<bool>;
arr_decorator!(BoolArray, bool, false, true);
pub type SignedInt32Array = Array<i32>;
arr_decorator!(SignedInt32Array, i32, 0_i32, 1_i32);
int_arr_decorator!(SignedInt32Array, i32);
