/* eslint-disable @typescript-eslint/no-non-null-assertion */
import { Value } from './value.ts'

export class Neuron {
  weights: Value[]
  bias: Value

  constructor(inputs: number) {
    this.weights = Array.from({ length: inputs }).map(() => new Value(Math.random() * 2 - 1))
    this.bias = new Value(Math.random() * 2 - 1)
  }

  forward(inputs: Value[]): Value {
    // w * x + b
    const activation = this.weights.reduce((sum, w, i) => sum.add(w.mul(inputs[i])), this.bias)
    return activation.tanh()
  }

  parameters(): Value[] {
    return [...this.weights, this.bias]
  }
}

export class Layer {
  neurons: Neuron[]

  constructor(inputs: number, outputs: number) {
    this.neurons = Array.from({ length: outputs }).map(() => new Neuron(inputs))
  }

  forward(inputs: Value[]): Value[] {
    return this.neurons.map((neuron) => neuron.forward(inputs))
  }

  parameters(): Value[] {
    return this.neurons.flatMap((neuron) => neuron.parameters())
  }
}

// biome-ignore lint/style/useNamingConvention:
export class MLP {
  layers: Layer[]

  constructor(inputs: number, outputs: number[]) {
    const sizes = [inputs, ...outputs]
    this.layers = sizes.slice(1).map((size, i) => {
      // biome-ignore lint/style/noNonNullAssertion: this is a typescript limitation
      return new Layer(sizes[i]!, size)
    })
  }

  forward(inputs: Value[]): Value[] {
    return this.layers.reduce((prev, layer) => layer.forward(prev), inputs)
  }

  parameters(): Value[] {
    return this.layers.flatMap((layer) => layer.parameters())
  }

  train(xs: number[][], ys: number[], learningRate = 0.1, epochs = 100): void {
    for (let epoch = 0; epoch < epochs; epoch++) {
      let totalLoss = new Value(0)

      for (let i = 0; i < xs.length; i++) {
        // biome-ignore lint/style/noNonNullAssertion:
        const inputs = xs[i]!.map((x) => new Value(x))
        const pred = this.forward(inputs)[0]
        // biome-ignore lint/style/noNonNullAssertion:
        const target = new Value(ys[i]!)
        // biome-ignore lint/style/noNonNullAssertion:
        const loss = pred!.sub(target).pow(2)
        totalLoss = totalLoss.add(loss)
      }

      totalLoss.resetGrad()

      // Backward pass
      totalLoss.backward()

      // Update parameters
      for (const p of this.parameters()) {
        p.data -= learningRate * p.grad
        p.grad = 0
      }

      if (epoch % 10 === 0) {
        console.log(`Epoch ${epoch}, Loss: ${totalLoss.data}`)
      }
    }
  }
}
