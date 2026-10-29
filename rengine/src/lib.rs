#[cfg(feature = "blas-apple")]
use blas_src as _;
pub mod math {
    pub mod f32 {
        pub mod arith;
        #[cfg(test)]
        mod arith_test;
        pub mod linalg;
        #[cfg(test)]
        mod linalg_test;
    }

    pub mod f64 {
        pub mod linalg;
        #[cfg(test)]
        mod linalg_test;
    }
    pub mod i32 {
        pub mod reductive_arg;
        #[cfg(test)]
        mod reductive_arg_test;
        pub mod reductive_arith;
        #[cfg(test)]
        mod reductive_arith_test;
        pub mod stats;
        #[cfg(test)]
        mod stats_test;
    }
    pub mod generic {
        pub mod arith;
        pub mod stats;
    }
}

pub mod data {
    pub mod array;
    #[cfg(test)]
    mod array_test;
    pub mod matrix;
    #[cfg(test)]
    mod matrix_test;
}

pub mod logic {
    pub mod boolean;
    #[cfg(test)]
    mod boolean_test;
    pub mod compare;
    #[cfg(test)]
    mod compare_test;
}

pub mod matrix {
    pub mod ops;
}
