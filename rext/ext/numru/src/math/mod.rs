pub mod reductive;
use magnus::{Error, RModule, prelude::*};

pub fn init(parent: RModule) -> Result<(), Error> {
    let math = parent.define_module("Math")?;
    reductive::init(math)?;
    Ok(())
}
