# MicroGrad

[![NPM version](https://img.shields.io/npm/v/@2bad/micrograd)](https://www.npmjs.com/package/@2bad/micrograd)
[![License](https://img.shields.io/npm/l/@2bad/micrograd)](https://opensource.org/license/MIT)
[![GitHub Build Status](https://img.shields.io/github/actions/workflow/status/2BAD/micrograd/build.yml)](https://github.com/2BAD/micrograd/actions/workflows/build.yml)
[![Code coverage](https://img.shields.io/codecov/c/github/2BAD/micrograd)](https://codecov.io/gh/2BAD/micrograd)
[![Written in TypeScript](https://img.shields.io/github/languages/top/2BAD/micrograd)](https://www.typescriptlang.org/)

A tiny autograd engine in TypeScript, ported from Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd). Every operation on a `Value` records its inputs, building a DAG on the fly, and `backward()` walks that graph in reverse topological order to compute gradients. On top of that sits a small neural network library (`Neuron`, `Layer`, `MLP`) that is enough to train a classifier.

It's meant for learning how backpropagation works, not for real workloads. Every scalar is an object, so anything beyond toy networks will be slow.

Compared to the Python original, operations come as both instance and static methods, inputs get coerced and validated, and there are a few extras: higher-order gradients, `log`, `exp`, `sigmoid`, gradient clipping, a gradient health check, and a Mermaid graph renderer.

## Requirements

- Node.js >= 26

## Install

```bash
npm install @2bad/micrograd
```

## Quick start

```typescript
import { Value } from '@2bad/micrograd'

const a = new Value(-4, 'a')
const b = new Value(2, 'b')

const c = a.add(b, 'c')
const d = a.mul(b).add(b.pow(3), 'd')
const e = c.sub(d, 'e')
const f = e.pow(2, 'f')

f.backward()

console.log(f.data) // 4
console.log(a.grad) // 4, df/da
console.log(b.grad) // 28, df/db
```

Gradients accumulate. Call `resetGrad()` on the root before running `backward()` again on the same graph.

## Higher-order gradients

Each op's local derivative is itself built from `Value` operations, so the gradient graph can be differentiated again. Pass an order to `backward` and read the results per node:

```typescript
const x = new Value(3)
x.pow(3).backward(3)

x.getHigherOrderGradient(1) // 27, 3x^2
x.getHigherOrderGradient(2) // 18, 6x
x.getHigherOrderGradient(3) // 6
```

These are pure derivatives of the root with respect to one node (d^n f/dx^n), not mixed partials. Orders that weren't computed return 0.

## Training a network

```typescript
import { MLP, Value } from '@2bad/micrograd'

// 3 inputs, two hidden layers of 4, one output
const mlp = new MLP(3, [4, 4, 1])

const xs = [
  [2, 3, -1],
  [3, -1, 0.5],
  [0.5, 1, 1],
  [1, 1, -1]
]
const ys = [1, -1, -1, 1]

mlp.train(xs, ys, 0.05, 50)

const [out] = mlp.forward([2, 3, -1].map((x) => new Value(x)))
console.log(out?.data) // close to 1
```

`train` does plain gradient descent on squared error against the first output and logs the loss every 10 epochs. Every neuron uses `tanh`. For anything else, write your own loop over `forward()` and `parameters()`.

## Visualizing the graph

```typescript
import { GraphVisualizer, Value } from '@2bad/micrograd'

const p = new Value(2, 'p')
const q = p.mul(3, 'q')
q.backward()

console.log(new GraphVisualizer().generateMermaid(q))
```

The output is a [Mermaid](https://mermaid.js.org/) flowchart with each node's data and grad. Paste it into anything that renders Mermaid, GitHub markdown included.

## API

```typescript
class Value {
  constructor(data: number, label?: string, children?: Value[], operation?: string)
  static from(value: unknown): Value

  data: number
  grad: number
  readonly id: string
  readonly label: string
  readonly children: Value[]
  readonly operation: string

  add(b: unknown, label?: string): Value
  sub(b: unknown, label?: string): Value
  mul(b: unknown, label?: string): Value
  div(b: unknown, label?: string): Value
  pow(b: unknown, label?: string): Value
  exp(): Value
  tanh(): Value

  static add(a: unknown, b: unknown, label?: string): Value
  static sub(a: unknown, b: unknown, label?: string): Value
  static mul(a: unknown, b: unknown, label?: string): Value
  static div(a: unknown, b: unknown, label?: string): Value
  static pow(a: unknown, b: unknown, label?: string): Value
  static exp(a: unknown, label?: string): Value
  static log(a: unknown, label?: string): Value
  static tanh(a: unknown, label?: string): Value
  static relu(a: unknown, label?: string): Value
  static sigmoid(a: unknown, label?: string): Value
  static negate(a: unknown, label?: string): Value

  backward(order?: number): void
  getHigherOrderGradient(order: number): number
  resetGrad(): void
  clipGradients(maxNorm: number): void
  checkGradientHealth(): { hasExploding: boolean; hasVanishing: boolean; maxGrad: number; minGrad: number }
}

class Neuron {
  constructor(inputs: number)
  forward(inputs: Value[]): Value
  parameters(): Value[]
}

class Layer {
  constructor(inputs: number, outputs: number)
  forward(inputs: Value[]): Value[]
  parameters(): Value[]
}

class MLP {
  constructor(inputs: number, outputs: number[])
  forward(inputs: Value[]): Value[]
  parameters(): Value[]
  train(xs: number[][], ys: number[], learningRate?: number, epochs?: number): void
}

class GraphVisualizer {
  generateMermaid(root: Value): string
}
```

Operands typed `unknown` go through `Value.from`, which accepts a `Value`, a number, a numeric string, a boolean (as 1 or 0), or a single-element array. Anything else throws.

`checkGradientHealth` reports exploding gradients above 1e3 and vanishing ones below 1e-3, ignoring zeros.

### Errors

Everything throws a plain `Error` instead of producing `NaN` or `Infinity`: non-finite data or grads, division by a near-zero value, `log` of a non-positive number, `0` raised to a non-positive power, a negative base with a non-integer exponent, and `pow` overflow.

## Development

```bash
git clone https://github.com/2BAD/micrograd.git
cd micrograd
pnpm install
pnpm build          # tsdown
pnpm check          # oxlint + oxfmt + tsc
pnpm test:unit      # vitest with coverage
```

## License

MIT. See [LICENSE](LICENSE).
