# MicroGrad

A TypeScript implementation of an autograd engine for educational purposes.

## Overview

MicroGrad implements backpropagation (reverse-mode autodiff) over a dynamically built Directed Acyclic Graph (DAG). This project demonstrates how to implement automatic differentiation principles in TypeScript.

## Key Components

- **Value Class**: Core autodiff functionality with gradient computation
- **Neural Network Primitives**: Simple Neuron, Layer, and MLP implementations
- **Graph Visualization**: Tools to visualize computation graphs

## Key improvements over the Python version

- **API Design**: Both instance and static methods for operations compared to instance-only methods
- **Higher Order Gradients**: Support for computing higher-order derivatives
- **Extended Math**: Additional operations including log, exp, tanh, and sigmoid
- **Gradient Tools**: Methods for gradient health checks and gradient clipping
- **Performance**: Iterative stack-based topological sort for better efficiency

## Usage Example

```typescript
import { Value } from '@2bad/micrograd';
  // Create computation graph
  const a = new Value(-4.0, 'a')
  const b = new Value(2.0, 'b')
  let c = Value.add(a, b, 'c') // a + b
  let d = Value.add(Value.mul(a, b), Value.pow(b, 3), 'd') // a * b + b**3

  // c += c + 1
  c = Value.add(c, Value.add(c, new Value(1.0)))

  // c += 1 + c + (-a)
  c = Value.add(c, Value.add(Value.add(new Value(1.0), c), Value.negate(a)))

  // d += d * 2 + (b + a).relu()
  const bPlusA = Value.add(b, a)
  d = Value.add(d, Value.add(Value.mul(d, 2), Value.relu(bPlusA)))

  // d += 3 * d + (b - a).relu()
  const bMinusA = Value.sub(b, a)
  d = Value.add(d, Value.add(Value.mul(3, d), Value.relu(bMinusA)))

  // e = c - d
  const e = Value.sub(c, d, 'e')

  // f = e**2
  const f = Value.pow(e, 2, 'f')

  // g = f / 2.0
  let g = Value.div(f, 2.0, 'g')

  // g += 10.0 / f
  g = Value.add(g, Value.div(10.0, f))

  // Forward pass
  console.log(g.data); // Value of the computation

  // Backward pass (compute gradients)
  g.backward();

  // Access gradients
  console.log(a.grad); // dg/da
  console.log(b.grad); // dg/db
```

## Building and Testing

```bash
# Install dependencies
npm install

# Build the project
npm run build

# Run tests
npm test
```

## Acknowledgements

This project is inspired by [micrograd](https://github.com/karpathy/micrograd) by Andrej Karpathy. The TypeScript implementation extends the core concepts with additional features and type safety.
