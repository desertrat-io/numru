# frozen_string_literal: true

require_relative "../../../spec_helper"

RSpec.describe Numru::Math::Reductive::I32 do
  describe ".sum" do
    i32_min = -(2**31)
    i32_max = (2**31) - 1

    shared_examples "an i32 sum" do |mode|
      {
        "an empty array" => [[], 0],
        "zero" => [[0], 0],
        "a positive integer" => [[42], 42],
        "a negative integer" => [[-42], -42],
        "positive integers" => [[1, 2, 3, 4, 5], 15],
        "negative integers" => [[-1, -2, -3, -4, -5], -15],
        "mixed signs and zeros" => [[-10, 0, 7, -3, 8], 2],
        "cancelling values" => [[17, -17, 42, -42], 0],
        "the minimum i32" => [[i32_min], i32_min],
        "the maximum i32" => [[i32_max], i32_max],
        "both i32 limits" => [[i32_min, i32_max], -1],
        "a sum reaching the upper limit" => [[i32_max - 1, 1], i32_max],
        "a sum reaching the lower limit" => [[i32_min + 1, -1], i32_min]
      }.each do |description, (input, expected)|
        it "sums #{description}" do
          expect(described_class.sum(input, mode)).to eq(expected)
        end
      end

      [3, 4, 5, 7, 8, 9, 4095, 4096, 4097, 8193].each do |length|
        it "sums #{length} elements including vector and chunk remainders" do
          input = Array.new(length) { |index| (index % 7) - 3 }

          expect(described_class.sum(input, mode)).to eq(input.sum)
        end
      end

      [i32_min - 1, i32_max + 1, -(2**100), 2**100].each do |value|
        it "rejects the out-of-range input #{value}" do
          expect { described_class.sum([0, value, 0], mode) }.to raise_error(RangeError)
        end
      end

      [nil, true, false, "1", :one, [1], Object.new].each do |value|
        it "rejects an array containing #{value.inspect}" do
          expect { described_class.sum([value], mode) }.to raise_error(TypeError)
        end
      end

      it "does not mutate the input array" do
        input = [3, -2, 0, 7].freeze

        expect(described_class.sum(input, mode)).to eq(8)
        expect(input).to eq([3, -2, 0, 7])
      end
    end

    %i[normal neon par_neon].each do |mode|
      context "with :#{mode}" do
        include_examples "an i32 sum", mode
      end
    end
  end
end
