use crate::converters::converter::NumruSymbolConverter;
use magnus::{Error, Ruby, Symbol};
use rengine::matrix::ops::Mode;

pub struct ModeConverter;
impl NumruSymbolConverter for ModeConverter {
    type ConvertedResult = Mode;
    fn convert(value: Symbol) -> Result<Self::ConvertedResult, Error> {
        let name = value.name()?.to_string();
        match name.as_str() {
            "normal" => Ok(Mode::Normal),
            "neon" => Ok(Mode::Neon),
            "par_neon" => Ok(Mode::ParNeon),
            _ => Err(Error::new(
                Ruby::get_with(value).exception_arg_error(),
                format!("Invalid computation type: :{name}"),
            )),
        }
    }
}
