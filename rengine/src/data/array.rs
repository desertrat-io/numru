// TODO: Update these so it's just a generic vector, no need to hardcode the type. This is for
// todo list organization

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

#[derive(Clone, Debug)]
pub struct BoolArray {
    data: Vec<bool>,
    len: usize,
}

impl BoolArray {
    pub fn new(data: Vec<bool>) -> Self {
        let len = data.len();
        Self { data, len }
    }

    pub fn slice(&self) -> &[bool] {
        &self.data
    }

    pub fn mut_slice(&mut self) -> &mut [bool] {
        &mut self.data
    }

    pub fn len(&self) -> usize {
        self.len
    }
}

#[derive(Clone, Debug)]
pub struct SignedIntArray {
    data: Vec<i32>,
    len: usize,
}

impl SignedIntArray {
    pub fn new(data: Vec<i32>) -> Self {
        let len = data.len();
        Self { data, len }
    }

    pub fn slice(&self) -> &[i32] {
        &self.data
    }

    pub fn mut_slice(&mut self) -> &mut [i32] {
        &mut self.data
    }

    pub fn len(&self) -> usize {
        self.len
    }
}
