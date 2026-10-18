use magnus::{Error, RArray, Symbol};

pub trait NumruSymbolConverter {
    type ConvertedResult;
    fn convert(value: Symbol) -> Result<Self::ConvertedResult, Error>;
}

pub trait NumruArrayConverter {
    type ConvertDataStructure;

    fn convert(value: RArray) -> Result<Self::ConvertDataStructure, Error>;
}
