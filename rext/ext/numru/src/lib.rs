pub mod converters;
pub mod math;
use magnus::{Error, Ruby};

#[magnus::init(name = "numru")]
fn init(ruby: &Ruby) -> Result<(), Error> {
    let numru = ruby.define_module("Numru")?;
    math::init(numru)?;
    Ok(())
}
