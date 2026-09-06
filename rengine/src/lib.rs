#[cfg(feature = "blas-apple")]
use blas_src as _;
pub mod math {
    pub mod f32 {
        pub mod arith;
        mod arith_test;
        pub mod linalg;
        mod linalg_test;
    }

    pub mod f64 {
        pub mod linalg;
        mod linalg_test;
    }
    pub mod i32 {
        pub mod reductive_arg;
        mod reductive_arg_test;
        pub mod reductive_arith;
        mod reductive_arith_test;
        pub mod stats;
        mod stats_test;
    }
    pub mod generic {
        pub mod stats;
    }
}

pub mod data {
    pub mod array;
}

pub mod logic {
    pub mod boolean;
    mod boolean_test;
    pub mod compare;
    mod compare_test;
}

pub mod matrix {
    pub mod ops;
}
