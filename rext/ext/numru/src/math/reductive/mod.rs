pub mod i32;

use magnus::{Error, RModule, prelude::*};

pub fn init(parent: RModule) -> Result<(), Error> {
    let reductive = parent.define_module("Reductive")?;
    i32::init(reductive)?;
    Ok(())
}
