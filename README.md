# MicroGrad

A tiny autograd engine in TypeScript, ported from Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd), with a small neural network library (`Neuron`, `Layer`, `MLP`) on top. Built for learning backpropagation, not for real workloads.

[![NPM version](https://img.shields.io/npm/v/@2bad/micrograd)](https://www.npmjs.com/package/@2bad/micrograd)
[![License](https://img.shields.io/npm/l/@2bad/micrograd)](https://opensource.org/license/MIT)
[![GitHub Build Status](https://img.shields.io/github/actions/workflow/status/2BAD/micrograd/build.yml)](https://github.com/2BAD/micrograd/actions/workflows/build.yml)
[![Code coverage](https://img.shields.io/codecov/c/github/2BAD/micrograd)](https://codecov.io/gh/2BAD/micrograd)
[![Written in TypeScript](https://img.shields.io/github/languages/top/2BAD/micrograd)](https://www.typescriptlang.org/)

Extras over the original: higher-order and mixed derivatives, `log`, `sigmoid`, gradient norm clipping, a Mermaid graph renderer, and errors instead of `NaN`.

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
const f = c.sub(d).pow(2)

f.backward()

console.log(f.data) // 4
console.log(a.grad) // 4, df/da
console.log(b.grad) // 28, df/db
```

`backward()` accumulates into `grad`. Call `zeroGrad()` to reset.

## Examples

### Higher-order derivatives

`gradients(inputs)` returns derivatives as `Value`s, which can be differentiated again:

```typescript
const x = new Value(3)

const [dx] = x.pow(3).gradients([x]) // 27, 3x^2
const [dx2] = dx.gradients([x]) // 18, 6x
const [dx3] = dx2.gradients([x]) // 6
```

Mixed partials:

```typescript
const x = new Value(2)
const y = new Value(3)

const [dfdx] = x.pow(2).mul(y).gradients([x]) // 12, 2xy
const [dfdxdy] = dfdx.gradients([y]) // 4, 2x
```

### Training a network

```typescript
import { MLP } from '@2bad/micrograd'

// 3 inputs, two hidden layers of 4, one output
const model = new MLP(3, [4, 4, 1])

const xs = [
  [2, 3, -1],
  [3, -1, 0.5],
  [0.5, 1, 1],
  [1, 1, -1]
]
const ys = [1, -1, -1, 1]

for (let step = 0; step < 100; step++) {
  const loss = xs.map((x, i) => model.forward(x)[0].sub(ys[i]).pow(2)).reduce((sum, term) => sum.add(term))

  model.zeroGrad()
  loss.backward()
  model.clipGradNorm(1)
  for (const p of model.parameters()) {
    p.data -= 0.05 * p.grad
  }
}

console.log(xs.map((x) => model.forward(x)[0].data)) // close to [1, -1, -1, 1]
```

### Visualizing the graph

```typescript
import { Value, toMermaid } from '@2bad/micrograd'

const p = new Value(2, 'p')
const q = p.mul(3)
q.label = 'q'
q.backward()

console.log(toMermaid(q))
```

Prints a [Mermaid](https://mermaid.js.org/) flowchart with each node's data and grad.

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

`MLP` defaults: `activation` (`'tanh'`), `outputActivation` (`'linear'`). `clipGradNorm` returns the norm before clipping.

### Errors

Ops throw a `RangeError` naming the op instead of returning `NaN` or `Infinity`, e.g. `/ produced Infinity`. `backward()` throws on gradient overflow, `Neuron.forward` on a wrong input count.

## License

MIT. See [LICENSE](LICENSE).
