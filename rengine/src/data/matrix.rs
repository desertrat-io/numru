use crate::data::array::Array;
use blas::{c32, c64};

#[derive(Debug, Clone, Default)]
pub struct Matrix<T> {
    data: Array<T>,
    cols: usize,
    rows: usize,
}

impl<T: Default + Clone> Matrix<T> {
    pub fn new(cols: usize, rows: usize, data: Array<T>) -> Self {
        assert_ne!(cols, usize::MAX, "matrix dimensions overflow");
        assert_ne!(rows, usize::MAX, "matrix dimensions overflow");
        assert!((cols * rows) < usize::MAX - 1, "matrix dimensions overflow");
        assert_eq!(
            cols * rows,
            data.len(),
            "matrix dimensions do not match data length"
        );
        Self { data, cols, rows }
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn slice(&self) -> &[T] {
        self.data.slice()
    }

    pub fn slice_mut(&mut self) -> &mut [T] {
        self.data.mut_slice()
    }

    pub fn empty(cols: usize, rows: usize) -> Self {
        Self::new(cols, rows, Array::<T>::default_padded(cols * rows))
    }

    pub fn row(&self, row: usize) -> Option<&[T]> {
        if row >= self.rows {
            return None;
        }
        unsafe { Some(self.row_unchecked(row)) }
    }

    pub unsafe fn row_unchecked(&self, row: usize) -> &[T] {
        let start = row * self.cols;
        let end = start + self.cols;
        &self.data.slice()[start..end]
    }

    pub unsafe fn row_unchecked_mut(&mut self, row: usize) -> &mut [T] {
        let start = row * self.cols;
        let end = start + self.cols;
        &mut self.data.mut_slice()[start..end]
    }

    pub fn get(&mut self, row: usize, col: usize) -> &mut T {
        &mut self.data.mut_slice()[row * self.cols + col]
    }
    pub fn get_or_else(&self, col: usize, row: usize) -> Option<&T> {
        // no need to check the length, you can't create a matrix at all if they aren't valid
        if col >= self.cols || row >= self.rows {
            return None;
        }
        Some(&self.data.slice()[row * self.cols + col])
    }
}

pub type BoolMatrix = Matrix<bool>;
pub type F32Matrix = Matrix<f32>;
pub type F64Matrix = Matrix<f64>;
pub type C32Matrix = Matrix<c32>;
pub type C64Matrix = Matrix<c64>;
