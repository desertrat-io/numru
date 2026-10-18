# frozen_string_literal: true

require "bundler/setup"

$LOAD_PATH.unshift File.expand_path("../lib", __dir__)

require "numru/numru"

RSpec.configure do |config|
  config.disable_monkey_patching!

  config.expect_with :rspec do |expectations|
    expectations.syntax = :expect
  end
end
