use crate::converters::array::SignedInt32Converter;
use crate::converters::converter::{NumruArrayConverter, NumruSymbolConverter};
use crate::converters::mode::ModeConverter;
use magnus::{Error, Module, RArray, RModule, Symbol, function, prelude::*};
use rengine::math::i32::reductive_arith::sum as numru_sum;

pub fn init(parent: RModule) -> Result<(), Error> {
    let i32_module = parent.define_module("I32")?;
    i32_module.define_singleton_method("sum", function!(sum, 2))?;
    Ok(())
}
pub fn sum(array: RArray, mode: Symbol) -> Result<i32, Error> {
    let mode = ModeConverter::convert(mode)?;
    let vector = SignedInt32Converter::convert(array)?;
    let result = numru_sum(&vector, mode);
    Ok(result)
}
