# frozen_string_literal: true

require "rb_sys/cargo/metadata"

Gem::Specification.new do |spec|
  spec.name = "numru-rext"
  spec.version = RbSys::Cargo::Metadata.new("numru").version
  spec.authors = ["Zara"]
  spec.summary = "Compiled Rust extension for Numru"
  spec.homepage = "https://github.com/desertrat-io/numru"
  spec.license = "MIT"
  spec.required_ruby_version = "~> 4.0.0"
  spec.files = ["LICENSE.txt"]
  spec.require_paths = ["lib"]
  spec.extensions = ["ext/numru/extconf.rb"]
  spec.metadata["cargo_crate_name"] = "numru"
  spec.metadata["allowed_push_host"] = "https://ruby.cloudsmith.io/desertrat-io/numru/"
end
