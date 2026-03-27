# What's in a name?

NumRu is a NUMeric RUst and RUby computing engine for computational ML related workloads.
Core activities are handled by a 100% Rust core engine and exposed through a compiled ruby extension to maximize speed,
cleverly manage memory,
and allow usage through a more expressive syntax. With the engine separated into its own layer, we free developers to
focus on building whatever libraries
they want without coupling them to a single scripting language.

## But NumPy...

Already does this extremely well, yes. There's no replacing that either. My goal with this project is to address some of
my personal gripes with NumPy as a codebase, not as a library.

- NumPy is old, and various tradeoffs have been made over time to ensure its high performing calculations
- The codebase is a web of Python/C bindings, with C++ thrown in a few places too
- Contributions can be very time consuming to even figure out where to get started
- It re-enforces the idea that you can only do scientific and ML workloads in Python, limiting developer choice
- We can get around some of the inherent issues NumPy will run into in the future in supporting state of the art CPUs,
  GPUs, and TPUs
- Layers of compatibility enhancements have built up over time as open source standards have evolved as well, creating a
  fragile codebase

This project is not to throw NumPy out, or to even minimize its impacts on the scientific computing and AI/ML world.
NumPy is a foundational project
and the inspiration for a generation of data scientists and AI engineers, and if anything this project exists only
because of the impact NumPy has had
in the computing world.

# Structure

NumRu is a layered numerical computing project: a Rust engine for core array and numerical operations, a compiled Ruby
extension that exposes that engine safely and efficiently, and a Ruby API that provides the primary user experience. The
structure is intentional. Rust handles computation, memory layout, and performance-sensitive internals. The extension
layer handles the boundary between languages, and is designed to be as thin as possible. Ruby provides the interface
most
users will actually work with.

# Why use Ruby at all?

Even when Python delegates the heavy lifting to C, C++, or Cython, the surrounding system still inherits Python’s
runtime
model and much of its ecosystem complexity. If the foundation is difficult to change, long-term progress becomes more
expensive than it needs to be.

Ruby was chosen because I think it posses several advantages over Python that negate the original reasoning behind
supporting Python in
the first place, largely by the assumption that Python is the simpler language. That assumption is not as obvious as I
believe people assume.
Ruby can be argued to be just as simple, and in many cases more coherent, especially when you take the full computing
environment into account rather than just a few
lines of syntax. Python often sells simplicity at the language surface while pushing complexity into packaging,
environment management, dependency isolation, version conflicts, and fragmented tooling. Ruby, by contrast, has long had
a more unified and predictable application experience. The language is expressive, the object model is more consistent,
and the ecosystem has historically done a better job of making the development environment feel integrated rather than
improvised.

Ruby also avoids or improves on several frustrations commonly associated with Python:

- A more consistent object model, where everything is more cleanly treated as an object
- A more expressive and readable syntax for building APIs and DSL-like interfaces
- Fewer "there should be one obvious way, except when there are actually several" design inconsistencies
- A more coherent story around dependency management and project setup through Bundler and gems
- Less cultural dependence on constantly escaping into another language just to make core workflows tolerable
- A language design that often feels more intentional and pleasant for developers building abstractions

The point is not that Ruby should replace Python by imitation. The point is that Ruby may make more sense as a
user-facing language for data science and AI/ML if it is paired with a serious native engine. Rust gives the project
safety, speed, and control over numerical internals. Ruby gives it a cleaner, more expressive, more enjoyable top-level
interface. The result aims to be a system that is easier to maintain than NumPy’s current stack, faster than pure Python
can realistically be, and better to use than the status quo suggests is possible.

# tl;dr;

Key points:

- NumRu is a three-layer system: Rust engine, Ruby extension, Ruby API to provide a replacement for NumPy
- This project is motivated by maintainability problems in the NumPy ecosystem, not just performance (which NumPy is
  extremely good at)
- Python’s apparent simplicity often hides complexity in packaging, environments, and tooling
- Ruby is at least as capable of simplicity, and often more coherent at the language and environment level
- Ruby offers a strong candidate for a better user-facing language in scientific computing when backed by Rust.
  Opinionated but not yet proven to be fair

The goal is a cleaner, faster, more maintainable foundation for numerical computing.

### And yes, at least part of the reason is because I was bored on a Friday night. Yes I know this will take a very long time