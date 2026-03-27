use magnus::{function, Ruby, Error};
fn numru_hello() -> String {
    format!("{}", String::from("Hello there!"))
}

#[magnus::init(name="numru")]
fn init(ruby: &Ruby) -> Result<(), Error> {
    ruby.define_global_function("hello", function!(numru_hello, 0));
    Ok(())
}