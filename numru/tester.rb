require 'bundler/setup'
require 'numru/numru'

puts Numru::Math::Reductive::I32.sum([3, 4, 5, 7, 8, 9, 4095, 4096, 4097, 8193], :par_neon)