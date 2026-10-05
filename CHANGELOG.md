# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.0.0] - 2026-10-05

### Breaking Changes

- Operands are now `Value | number` only; strings, booleans and arrays are no longer accepted.
- Replaced `backward(order)` and `getHigherOrderGradient` with `gradients(inputs)`, which returns differentiable `Value`s.
- Replaced `GraphVisualizer` with the `toMermaid(root)` function.
- `MLP` output layers are now linear by default; pass `outputActivation: 'tanh'` for the old behavior.
- `pow` now takes a number exponent. Use `x.log().mul(y).exp()` for a variable exponent.
- Errors are now `RangeError`s that name the failing operation.

### Added

- Added mixed partial derivatives through repeated `gradients` calls
- Added `neg`, `log`, `sigmoid` and `relu` as instance methods
- Added `Module.zeroGrad` and `Module.clipGradNorm` for clipping the global gradient norm
- `Neuron`, `Layer` and `MLP` can now use `tanh`, `relu`, `sigmoid` or `linear` activations
- `forward` now accepts plain numbers as inputs

### Fixed

- Fixed `pow` returning a zero gradient when the base is zero
- Fixed repeated `backward` calls overcounting gradients instead of doubling them
- Fixed a stack overflow in `backward` and `zeroGrad` on deep graphs
- Fixed `div` rejecting valid divisors smaller than `Number.EPSILON`
- Fixed Mermaid output repeating edges for shared nodes and adding extra whitespace to labels
- Fixed `Neuron.forward` accepting the wrong number of inputs without an error

### Changed

- Weights are now initialized scaled by `1 / sqrt(inputs)` and biases start at zero

### Removed

- Removed static operation methods such as `Value.add`; use the instance methods instead
- Removed `Value.from`, `Value.id`, `resetGrad` (use `zeroGrad`) and string conversion via `+value`
- Removed `MLP.train`; write the loop with `forward`, `backward` and `parameters`
- Removed `Value.clipGradients` and `checkGradientHealth`; use `Module.clipGradNorm` instead
