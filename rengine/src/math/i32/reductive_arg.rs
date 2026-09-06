use crate::data::array::SignedIntArray;
use crate::matrix::ops::{Mode, PAR_CHUNK_SIZE};
use rayon::iter::IndexedParallelIterator;
use rayon::iter::ParallelIterator;
use rayon::prelude::ParallelSlice;
use std::arch::aarch64::{
    int32x4_t, uint32x4_t, vaddq_s32, vbslq_s32, vcgtq_s32, vcltq_s32, vdupq_n_s32, vgetq_lane_s32,
    vld1q_s32,
};

trait ArgReductionOp {
    const EDGE_VALUE: i32;

    fn reduce(newer: int32x4_t, current: int32x4_t) -> uint32x4_t;

    fn scalar_reduce(newer: i32, current: i32) -> bool;
}

struct ArgMin;

impl ArgReductionOp for ArgMin {
    const EDGE_VALUE: i32 = i32::MAX;

    #[inline(always)]
    fn reduce(newer: int32x4_t, current: int32x4_t) -> uint32x4_t {
        unsafe { vcltq_s32(newer, current) }
    }

    #[inline(always)]
    fn scalar_reduce(newer: i32, current: i32) -> bool {
        newer < current
    }
}

struct ArgMax;

impl ArgReductionOp for ArgMax {
    const EDGE_VALUE: i32 = i32::MIN;

    #[inline(always)]
    fn reduce(newer: int32x4_t, current: int32x4_t) -> uint32x4_t {
        unsafe { vcgtq_s32(newer, current) }
    }

    #[inline(always)]
    fn scalar_reduce(newer: i32, current: i32) -> bool {
        newer > current
    }
}

/// convergence helper function for arg reductions, do not
/// use elsewhere
fn reduce_pair<R: ArgReductionOp>(
    lhs_values: int32x4_t,
    lhs_indices: int32x4_t,
    rhs_values: int32x4_t,
    rhs_indices: int32x4_t,
) -> (int32x4_t, int32x4_t) {
    let mask = R::reduce(rhs_values, lhs_values);
    unsafe {
        (
            vbslq_s32(mask, rhs_values, lhs_values),
            vbslq_s32(mask, rhs_indices, lhs_indices),
        )
    }
}

fn arg_scalar<R: ArgReductionOp>(vector: &[i32]) -> u32 {
    let mut index: usize = 0;
    let mut current_index: usize = 0;
    let mut current = R::EDGE_VALUE;
    for _ in vector {
        let v_current = vector[index];
        if R::scalar_reduce(v_current, current) {
            current_index = index;
            current = v_current;
        }
        index += 1;
    }
    current_index as u32
}

/// Very opinionated arg reduction algorithm
/// Returns the index of the value in the vector that is the "best"
/// according to the heuristic defined by the reduction type passed in
/// by the caller
fn arg_neon<R: ArgReductionOp>(vector: &[i32]) -> usize {
    assert!(!vector.is_empty());
    unsafe {
        // first go ahead and allocate lanes with the edge value
        // so that we start with the "opposite" of best
        let mut lane_vals_0 = vdupq_n_s32(R::EDGE_VALUE);
        let mut lane_vals_1 = vdupq_n_s32(R::EDGE_VALUE);
        let mut lane_vals_2 = vdupq_n_s32(R::EDGE_VALUE);
        let mut lane_vals_3 = vdupq_n_s32(R::EDGE_VALUE);

        // create a pointer for each index. the ord positional of the index
        // across all 16 total chunks is the number contained at a given
        // point in the loaded 4 long signed 32 bit values in the 128 bit vector
        // register. returns a pointer to the value
        let mut index_0 = vld1q_s32([0, 1, 2, 3].as_ptr());
        let mut index_1 = vld1q_s32([4, 5, 6, 7].as_ptr());
        let mut index_2 = vld1q_s32([8, 9, 10, 11].as_ptr());
        let mut index_3 = vld1q_s32([12, 13, 14, 15].as_ptr());

        // let these accumulators have a starting value for further reduction
        let mut current_index_0 = index_0;
        let mut current_index_1 = index_1;
        let mut current_index_2 = index_2;
        let mut current_index_3 = index_3;

        let mask_val = 16;
        let mask = vdupq_n_s32(mask_val);
        let len = vector.len();

        let mem_chunks = len / 16;

        // gives us a view into the vector
        let ptr = vector.as_ptr();
        for chunk in 0..mem_chunks {
            // go ahead and grab an offset into the "chunk" we setup
            // each addition to the pointer just points at the next signed 32 bit int
            // and extracts the value from the register
            let idx = chunk * 16;
            let x0 = vld1q_s32(ptr.add(idx));
            let x1 = vld1q_s32(ptr.add(idx + 4));
            let x2 = vld1q_s32(ptr.add(idx + 8));
            let x3 = vld1q_s32(ptr.add(idx + 12));

            // perform a reduction operation with the correct
            // NEON instruction
            let m0 = R::reduce(x0, lane_vals_0);
            let m1 = R::reduce(x1, lane_vals_1);
            let m2 = R::reduce(x2, lane_vals_2);
            let m3 = R::reduce(x3, lane_vals_3);

            // extract the computed value from that lane
            lane_vals_0 = vbslq_s32(m0, x0, lane_vals_0);
            lane_vals_1 = vbslq_s32(m1, x1, lane_vals_1);
            lane_vals_2 = vbslq_s32(m2, x2, lane_vals_2);
            lane_vals_3 = vbslq_s32(m3, x3, lane_vals_3);

            // extract the index associated with that value
            current_index_0 = vbslq_s32(m0, index_0, current_index_0);
            current_index_1 = vbslq_s32(m1, index_1, current_index_1);
            current_index_2 = vbslq_s32(m2, index_2, current_index_2);
            current_index_3 = vbslq_s32(m3, index_3, current_index_3);

            // finally, assign to the vector register the new best index for each
            // position, and add the value of the mask pointer to it
            // to move the window down
            index_0 = vaddq_s32(index_0, mask);
            index_1 = vaddq_s32(index_1, mask);
            index_2 = vaddq_s32(index_2, mask);
            index_3 = vaddq_s32(index_3, mask);
        }

        // naming convention is such that 01 is the first and second index in a lane
        // 23 is the third and fourth, etc etc
        // these three passes combine the best values and indices from each lane
        // into a final values
        let (value01, index01) =
            reduce_pair::<R>(lane_vals_0, current_index_0, lane_vals_1, current_index_1);
        let (value23, index23) =
            reduce_pair::<R>(lane_vals_2, current_index_2, lane_vals_3, current_index_3);
        let (values, indicies) = reduce_pair::<R>(value01, index01, value23, index23);

        // load 4 general purpose registers, each with one of the values
        let vals = [
            vgetq_lane_s32::<0>(values),
            vgetq_lane_s32::<1>(values),
            vgetq_lane_s32::<2>(values),
            vgetq_lane_s32::<3>(values),
        ];

        // same, but with indicies
        let indices = [
            vgetq_lane_s32::<0>(indicies),
            vgetq_lane_s32::<1>(indicies),
            vgetq_lane_s32::<2>(indicies),
            vgetq_lane_s32::<3>(indicies),
        ];

        // before starting the loop, the first load
        // into the result variables assumes the first index and value
        // are the "best"
        let mut best_value = vals[0];
        let mut best_index = indices[0] as usize;

        // reduce the values in the general purpose registers and record the results
        // keeping in mind "best" may have changed
        // this uses the scalar reduction due to these values not being in the vector register
        // and are thus plucked individually
        for i in 1..4 {
            if R::scalar_reduce(vals[i], best_value) {
                best_value = vals[i];
                best_index = indices[i] as usize;
            }
        }

        // remainder reduction for anything still left in the primary vector
        for i in (mem_chunks * 16)..len {
            let value = vector[i];
            if R::scalar_reduce(value, best_value) {
                best_value = value;
                best_index = i;
            }
        }

        best_index
    }
}

fn arg_neon_par<R: ArgReductionOp>(vector: &[i32]) -> u32 {
    vector
        .par_chunks(PAR_CHUNK_SIZE)
        .enumerate()
        .map(|(chunk_num, chunk)| {
            let iteration_index = arg_neon::<R>(chunk);
            let accumulative_index = chunk_num * PAR_CHUNK_SIZE + iteration_index;
            (chunk[iteration_index], accumulative_index)
        })
        .reduce_with(|left, right| {
            if R::scalar_reduce(left.0, right.0) {
                right
            } else {
                left
            }
        })
        .unwrap()
        .1 as u32
}

pub fn argmax(vector: SignedIntArray, mode: Mode) -> u32 {
    match mode {
        Mode::Normal => arg_scalar::<ArgMax>(vector.slice()),
        Mode::Neon => arg_neon::<ArgMax>(vector.slice()) as u32,
        Mode::ParNeon => arg_neon_par::<ArgMax>(vector.slice()),
    }
}

pub fn argmin(vector: SignedIntArray, mode: Mode) -> u32 {
    match mode {
        Mode::Normal => arg_scalar::<ArgMin>(vector.slice()),
        Mode::Neon => arg_neon::<ArgMin>(vector.slice()) as u32,
        Mode::ParNeon => arg_neon_par::<ArgMin>(vector.slice()),
    }
}
