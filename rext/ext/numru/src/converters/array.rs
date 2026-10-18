use crate::converters::converter::NumruArrayConverter;
use magnus::{Error, RArray};
use rengine::data::array::{BoolArray, SignedF32Array, SignedF64Array, SignedInt32Array};

// TODO: turn this into macros
pub struct SignedInt32Converter;

impl NumruArrayConverter for SignedInt32Converter {
    type ConvertDataStructure = SignedInt32Array;

    // TODO: This loop could get slow for huge arrays, reconsider conversion mechanism
    // however into_iter uses a fast implementation via the magnus crate
    fn convert(value: RArray) -> Result<Self::ConvertDataStructure, Error> {
        // not checking length here, the engine does this automatically, only barebones
        // conversions and passthru should be in this ffi
        // so we don't use if value.len() >= i32::MAX or anything of the sort
        let single_dim_array = value.to_vec::<i32>()?;
        Ok(SignedInt32Array::new(single_dim_array))
    }
}

pub struct SignedFloat32Converter;

impl NumruArrayConverter for SignedFloat32Converter {
    type ConvertDataStructure = SignedF32Array;

    fn convert(value: RArray) -> Result<Self::ConvertDataStructure, Error> {
        let single_dim_array = value.to_vec::<f32>()?;
        Ok(SignedF32Array::new(single_dim_array))
    }
}

pub struct SignedFloat64Converter;

impl NumruArrayConverter for SignedFloat64Converter {
    type ConvertDataStructure = SignedF64Array;

    fn convert(value: RArray) -> Result<Self::ConvertDataStructure, Error> {
        let single_dim_array = value.to_vec::<f64>()?;
        Ok(SignedF64Array::new(single_dim_array))
    }
}

pub struct BoolArrayConverter;

impl NumruArrayConverter for BoolArrayConverter {
    type ConvertDataStructure = BoolArray;

    fn convert(value: RArray) -> Result<Self::ConvertDataStructure, Error> {
        let single_dim_array = value.to_vec::<bool>()?;
        Ok(BoolArray::new(single_dim_array))
    }
}
