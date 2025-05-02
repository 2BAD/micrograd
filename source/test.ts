import { GraphVisualizer } from './utils/visualizer.ts'
import { Value } from './value.ts'

const visualizer = new GraphVisualizer()

// inputs x1, x2
const x1 = new Value(2.0, 'x1')
const x2 = new Value(0.0, 'x2')

// weights w1, w2
const w1 = new Value(-3.0, 'w1')
const w2 = new Value(1.0, 'w2')

// bias of the neuron
const b = new Value(6.8813735870195432, 'b')

// x1*w1 + x2*w2 + b
const x1w1 = Value.multiply(x1, w1, 'x1*w1')
const x2w2 = Value.multiply(x2, w2, 'x2*w2')
const x1w1x2w2 = Value.sum(x1w1, x2w2, 'x1*w1 + x2*w2')
const n = Value.sum(x1w1x2w2, b, 'n')

// Calculate tanh
// const o = Value.tanh(n, 'o')
const e = Value.exp(Value.multiply(2, n, '2*n'))
const o = Value.divide(Value.subtract(e, 1, 'e-1'), Value.sum(e, 1, 'e+1'), 'o')

o.backward()

// Generate visualization
console.log(visualizer.generateMermaid(o))
