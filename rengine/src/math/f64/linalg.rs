use crate::data::array::{CONTIGUOUS_STRIDE, SignedF64Array};
use crate::matrix::ops::Mode;
use blas::ddot;


pub fn dot(left_vector: SignedF64Array, right_vector: SignedF64Array, mode: Mode) -> f64 {
    assert_eq!(left_vector.len(), right_vector.len());
    let result;
    match mode {
        Mode::Normal => result = dot_blas(left_vector.slice(), right_vector.slice()),
        _ => todo!(),
    };
    result
}

fn dot_blas(left_vector: &[f64], right_vector: &[f64]) -> f64 {
    let mut sum = 0.0;
    let lim = i32::MAX as usize;
    // certain blas implementations return f64 for f32 operations, and I do believe that's how
    // apple's accelerate works? Not all implementations work this way
    #[cfg(feature = "blas-apple")]
    unsafe {
        // this will cleanly divide the vector into parts that fit inside the total length
        // no iterator overhead, just the chunks
        for (left_chunk, right_chunk) in left_vector.chunks(lim).zip(right_vector.chunks(lim)) {
            sum += ddot(left_chunk.len() as i32, left_chunk, CONTIGUOUS_STRIDE, right_chunk, CONTIGUOUS_STRIDE);
        }
    };
    sum
}
