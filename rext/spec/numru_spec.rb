# frozen_string_literal: true

require_relative "spec_helper"

RSpec.describe "numru native extension" do
  it "loads the compiled extension" do
    expect { require "numru/numru" }.not_to raise_error
  end

  it "exposes the integer reduction binding" do
    expect(Numru::Math::Reductive::I32).to respond_to(:sum)
    expect(Numru::Math::Reductive::I32.method(:sum).arity).to eq(2)
  end
end
