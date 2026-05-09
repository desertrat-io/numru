#[derive(Clone, Debug)]
pub struct Array {
    data: Vec<f32>,
    len: usize,
}
impl Array {
    pub fn new(data: Vec<f32>) -> Self {
        let len = data.len();
        Self { data, len }
    }

    pub fn zero_padded(len: usize) -> Self {
        Self {
            data: vec![0.0; len],
            len,
        }
    }

    pub fn slice(&self) -> &[f32] {
        &self.data
    }

    pub fn mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }

    pub fn len(&self) -> usize {
        self.len
    }
}
