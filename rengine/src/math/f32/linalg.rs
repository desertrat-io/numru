use crate::data::array::{CONTIGUOUS_STRIDE, SignedF32Array};
use crate::matrix::ops::LinAlgMode;
use blas::dsdot;

pub fn dot(left_vector: SignedF32Array, right_vector: SignedF32Array, mode: LinAlgMode) -> f64 {
    assert_eq!(left_vector.len(), right_vector.len());
    let result;
    match mode {
        LinAlgMode::Normal => result = dot_scalar(left_vector.slice(), right_vector.slice()),
        LinAlgMode::Blas => result = dot_blas(left_vector.slice(), right_vector.slice()),
        _ => todo!(),
    };
    result
}

fn dot_scalar(left_vector: &[f32], right_vector: &[f32]) -> f64 {
    let mut result: f64 = 0.;

    for i in 0..left_vector.len() {
        result += left_vector[i] as f64 * right_vector[i] as f64;
    }

    result
}
fn dot_blas(left_vector: &[f32], right_vector: &[f32]) -> f64 {
    let mut sum = 0.0;
    let lim = i32::MAX as usize;
    // certain blas implementations return f64 for f32 operations, and I do believe that's how
    // apple's accelerate works? Not all implementations work this way
    #[cfg(feature = "blas-apple")]
    unsafe {
        // this will cleanly divide the vector into parts that fit inside the total length
        // no iterator overhead, just the chunks
        for (left_chunk, right_chunk) in left_vector.chunks(lim).zip(right_vector.chunks(lim)) {
            sum += dsdot(left_chunk.len() as i32, left_chunk, CONTIGUOUS_STRIDE, right_chunk, CONTIGUOUS_STRIDE);
        }
    };
    sum
}

// TODO: Benchmarking needed for parallel, not needed yet
fn dot_blas_par(left_vector: &[f32], right_vector: &[f32]) -> f64 {
    #[cfg(feature = "blas-apple")]
    todo!()
}
