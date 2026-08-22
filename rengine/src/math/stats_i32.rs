use crate::data::array::SignedIntArray;
use crate::math::stats::VarianceType;
use crate::matrix::ops::Mode;
trait StatsI32Op {
    const STARTING_MEAN: i32;
    fn calc(vector: &[i32], mean: i32, mode: Mode) -> i32;

    fn calc_scalar(vector: &[i32], mean: i32) -> i32;
}

struct SampleVar;
impl StatsI32Op for SampleVar {
    const STARTING_MEAN: i32 = 0;
    fn calc(vector: &[i32], mean: i32, mode: Mode) -> i32 {
        // save for when ndarray enters the picture
        todo!()
    }

    #[inline(always)]
    fn calc_scalar(vector: &[i32], mean: i32) -> i32 {
        assert_ne!(vector.len(), 1);
        let mut acculmulator = 0;
        for val in vector {
            let dist_from_mean = mean - val;
            acculmulator += i32::pow(dist_from_mean, 2);
        }
        acculmulator / (vector.len() - 1) as i32
    }
}

struct PopulationVar;
impl StatsI32Op for PopulationVar {
    const STARTING_MEAN: i32 = 1;
    fn calc(vector: &[i32], mean: i32, mode: Mode) -> i32 {
        todo!()
    }

    fn calc_scalar(vector: &[i32], mean: i32) -> i32 {
        assert_ne!(vector.len(), 1);
        let mut acculmulator = 0;
        for val in vector {
            let dist_from_mean = mean - val;
            acculmulator += i32::pow(dist_from_mean, 2);
        }
        acculmulator / vector.len() as i32
    }
}

fn var_scalar<R: StatsI32Op>(vector: &[i32], starting_mean: i32) -> i32 {
    R::calc_scalar(vector, starting_mean)
}

// To be done later
fn var_neon<R: StatsI32Op>(vector: &[i32], starting_mean: i32, mode: Mode) -> i32 {
    todo!()
}

/// Since this is a non floating point operation, we have to round the naswer
/// Rounding abides by f32 rounding rules
pub fn std(vector: SignedIntArray, mode: Mode, sample_type: VarianceType) -> i32 {
    let variance = match mode {
        Mode::Normal => var(vector, mode, sample_type) as f32,
        _ => unimplemented!()
    };
    variance.sqrt().round() as i32
}

pub fn var(vector: SignedIntArray, mode: Mode, sample_type: VarianceType) -> i32 {
    if vector.len() < 2 {
        return i32::MIN;
    }
    // TODO: I think this can be done in 1 pass instead of 2 so this instruction can be removed
    // and put closer to implementation
    let sum = crate::math::reductive_arith_i32::sum(&vector, mode);
    let initial_mean = sum / vector.len() as i32;
    match (mode, sample_type) {
        (Mode::Normal, VarianceType::Sample) => {
            var_scalar::<SampleVar>(vector.slice(), initial_mean)
        }
        (Mode::Neon, VarianceType::Sample) => {
            var_neon::<SampleVar>(vector.slice(), initial_mean, mode)
        }
        (_, VarianceType::Sample) => { unimplemented!() }
        (Mode::Normal, VarianceType::Population) => var_scalar::<PopulationVar>(vector.slice(), initial_mean),
        (Mode::Neon, VarianceType::Population) => var_neon::<PopulationVar>(vector.slice(), initial_mean, mode),
        (_, VarianceType::Population) => unimplemented!(),
    }
}
