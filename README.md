# MicroGrad

[![NPM version](https://img.shields.io/npm/v/@2bad/micrograd)](https://www.npmjs.com/package/@2bad/micrograd)
[![License](https://img.shields.io/npm/l/@2bad/micrograd)](https://opensource.org/license/MIT)
[![GitHub Build Status](https://img.shields.io/github/actions/workflow/status/2BAD/micrograd/build.yml)](https://github.com/2BAD/micrograd/actions/workflows/build.yml)
[![Code coverage](https://img.shields.io/codecov/c/github/2BAD/micrograd)](https://codecov.io/gh/2BAD/micrograd)
[![Written in TypeScript](https://img.shields.io/github/languages/top/2BAD/micrograd)](https://www.typescriptlang.org/)

A tiny autograd engine in TypeScript, ported from Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd). Every operation on a `Value` records its inputs, building a DAG on the fly, and `backward()` walks that graph in reverse topological order to compute gradients. On top of that sits a small neural network library (`Neuron`, `Layer`, `MLP`) that is enough to train a classifier.

It's meant for learning how backpropagation works, not for real workloads. Every scalar is an object, so anything beyond toy networks will be slow.

Compared to the Python original, there are a few extras: higher-order and mixed derivatives, `log` and `sigmoid`, gradient norm clipping, a Mermaid graph renderer, and errors instead of `NaN`.

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

const c = a.add(b)
const d = a.mul(b).add(b.pow(3))
const e = c.sub(d)
const f = e.pow(2)

f.backward()

console.log(f.data) // 4
console.log(a.grad) // 4, df/da
console.log(b.grad) // 28, df/db
```

Operands can be a `Value` or a plain number. `pow` takes a number exponent; for a variable exponent write `x.log().mul(y).exp()`.

`backward()` adds into `grad`, so calling it twice doubles every gradient and several graphs that share a parameter sum their contributions. Reset with `zeroGrad()` on the root, or on a model.

## Higher-order derivatives

`gradients(inputs)` returns the derivative of a value with respect to each input as a new `Value` instead of writing to `grad`. The result is itself a graph, so it can be differentiated again:

```typescript
const x = new Value(3)

const [dx] = x.pow(3).gradients([x]) // 27, 3x^2
const [dx2] = dx.gradients([x]) // 18, 6x
const [dx3] = dx2.gradients([x]) // 6
```

Mixed partials work the same way:

```typescript
const x = new Value(2)
const y = new Value(3)
const f = x.pow(2).mul(y) // x^2 * y

const [dfdx] = f.gradients([x]) // 12, 2xy
const [dfdxdy] = dfdx.gradients([y]) // 4, 2x
```

## Training a network

```typescript
import { MLP } from '@2bad/micrograd'

// 3 inputs, two hidden tanh layers of 4, one linear output
const model = new MLP(3, [4, 4, 1])

const xs = [
  [2, 3, -1],
  [3, -1, 0.5],
  [0.5, 1, 1],
  [1, 1, -1]
]
const ys = [1, -1, -1, 1]

for (let step = 0; step < 100; step++) {
  const loss = xs
    .map((x, i) => model.forward(x)[0].sub(ys[i]).pow(2))
    .reduce((sum, term) => sum.add(term))

  model.zeroGrad()
  loss.backward()
  model.clipGradNorm(1)
  for (const p of model.parameters()) {
    p.data -= 0.05 * p.grad
  }
}

console.log(xs.map((x) => model.forward(x)[0].data)) // close to [1, -1, -1, 1]
```

Hidden layers use `tanh` and the last layer is linear. Change either with `new MLP(3, [4, 4, 1], { activation: 'relu', outputActivation: 'sigmoid' })`. Weights start uniform in `[-1, 1] / sqrt(inputs)` and biases at 0.

## Visualizing the graph

```typescript
import { Value, toMermaid } from '@2bad/micrograd'

const p = new Value(2, 'p')
const q = p.mul(3)
q.label = 'q'
q.backward()

console.log(toMermaid(q))
```

The output is a [Mermaid](https://mermaid.js.org/) flowchart with each node's label, data and grad. Paste it into anything that renders Mermaid, GitHub markdown included.

## API

```typescript
type Operand = Value | number
type Activation = 'linear' | 'relu' | 'sigmoid' | 'tanh'

class Value {
  constructor(data: number, label?: string)

  data: number
  grad: number
  label: string
  readonly op: string
  readonly children: Value[]

  add(other: Operand): Value
  sub(other: Operand): Value
  mul(other: Operand): Value
  div(other: Operand): Value
  pow(exponent: number): Value
  neg(): Value
  exp(): Value
  log(): Value
  tanh(): Value
  sigmoid(): Value
  relu(): Value

  backward(): void
  gradients(inputs: Value[]): Value[]
  zeroGrad(): void
}

abstract class Module {
  parameters(): Value[]
  zeroGrad(): void
  clipGradNorm(maxNorm: number): number
}

class Neuron extends Module {
  constructor(inputs: number, activation?: Activation)
  forward(inputs: Operand[]): Value
}

class Layer extends Module {
  constructor(inputs: number, outputs: number, activation?: Activation)
  forward(inputs: Operand[]): Value[]
}

class MLP extends Module {
  constructor(inputs: number, outputs: number[], options?: { activation?: Activation; outputActivation?: Activation })
  forward(inputs: Operand[]): Value[]
}

function toMermaid(root: Value): string
```

`gradients` returns zero for inputs the value doesn't depend on. `clipGradNorm` scales all parameter gradients so their L2 norm is at most `maxNorm` and returns the norm before clipping.

### Errors

Instead of producing `NaN` or `Infinity`, every operation throws a `RangeError` naming the op, for example `/ produced Infinity` for division by zero or `log produced NaN` for the log of a negative number. The same applies to constructing a `Value` from a non-finite number, to a gradient that overflows during `backward()`, and to a derivative that doesn't exist at the point (such as `x.pow(0.5)` at 0). `Neuron.forward` throws when the input count doesn't match.

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
