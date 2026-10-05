import { type Operand, Value } from './value.ts'

export type Activation = 'linear' | 'relu' | 'sigmoid' | 'tanh'

export abstract class Module {
  abstract parameters(): Value[]

  zeroGrad(): void {
    for (const parameter of this.parameters()) {
      parameter.grad = 0
    }
  }

  clipGradNorm(maxNorm: number): number {
    const parameters = this.parameters()
    const norm = Math.sqrt(parameters.reduce((sum, parameter) => sum + parameter.grad ** 2, 0))
    if (norm > maxNorm) {
      for (const parameter of parameters) {
        parameter.grad *= maxNorm / norm
      }
    }
    return norm
  }
}

export class Neuron extends Module {
  readonly weights: Value[]
  readonly bias = new Value(0)
  readonly activation: Activation

  constructor(inputs: number, activation: Activation = 'tanh') {
    super()
    const scale = 1 / Math.sqrt(inputs)
    this.weights = Array.from({ length: inputs }, () => new Value((Math.random() * 2 - 1) * scale))
    this.activation = activation
  }

  forward(inputs: readonly Operand[]): Value {
    if (inputs.length !== this.weights.length) {
      throw new RangeError(`Expected ${this.weights.length} inputs, got ${inputs.length}`)
    }
    const sum = this.weights.reduce(
      (total, weight, index) => total.add(weight.mul(inputs[index] as Operand)),
      this.bias
    )
    return this.activation === 'linear' ? sum : sum[this.activation]()
  }

  parameters(): Value[] {
    return [...this.weights, this.bias]
  }
}

export class Layer extends Module {
  readonly neurons: Neuron[]

  constructor(inputs: number, outputs: number, activation: Activation = 'tanh') {
    super()
    this.neurons = Array.from({ length: outputs }, () => new Neuron(inputs, activation))
  }

  forward(inputs: readonly Operand[]): Value[] {
    return this.neurons.map((neuron) => neuron.forward(inputs))
  }

  parameters(): Value[] {
    return this.neurons.flatMap((neuron) => neuron.parameters())
  }
}

export class MLP extends Module {
  readonly layers: Layer[]

  constructor(
    inputs: number,
    outputs: readonly number[],
    {
      activation = 'tanh',
      outputActivation = 'linear'
    }: { activation?: Activation; outputActivation?: Activation } = {}
  ) {
    super()
    let fanIn = inputs
    this.layers = outputs.map((size, index) => {
      const layer = new Layer(fanIn, size, index === outputs.length - 1 ? outputActivation : activation)
      fanIn = size
      return layer
    })
  }

  forward(inputs: readonly Operand[]): Value[] {
    let activations = inputs.map((input) => (input instanceof Value ? input : new Value(input)))
    for (const layer of this.layers) {
      activations = layer.forward(activations)
    }
    return activations
  }

  parameters(): Value[] {
    return this.layers.flatMap((layer) => layer.parameters())
  }
}
