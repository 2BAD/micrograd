import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest'
import { Layer, MLP, Neuron } from './nn.ts'
import { Value } from './value.ts'

beforeEach(() => {
  let seed = 42
  vi.spyOn(Math, 'random').mockImplementation(() => {
    seed = (seed * 1_664_525 + 1_013_904_223) % 2 ** 32
    return seed / 2 ** 32
  })
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe('Neuron', () => {
  test('initializes scaled weights and a zero bias', () => {
    const neuron = new Neuron(4)
    expect(neuron.weights).toHaveLength(4)
    expect(neuron.weights.every((weight) => Math.abs(weight.data) <= 0.5)).toBe(true)
    expect(neuron.bias.data).toBe(0)
    expect(neuron.parameters()).toEqual([...neuron.weights, neuron.bias])
  })

  test.each([
    ['tanh', Math.tanh(-0.4)],
    ['relu', 0],
    ['sigmoid', 1 / (1 + Math.exp(0.4))],
    ['linear', -0.4]
  ] as const)('applies %s', (activation, expected) => {
    const neuron = new Neuron(2, activation)
    const [first, second] = neuron.weights
    if (!first || !second) {
      throw new Error('expected two weights')
    }
    first.data = 0.5
    second.data = -0.5
    neuron.bias.data = 0.1
    expect(neuron.forward([1, new Value(2)]).data).toBeCloseTo(expected, 12)
  })

  test('rejects the wrong number of inputs', () => {
    const neuron = new Neuron(2)
    expect(() => neuron.forward([1])).toThrow(new RangeError('Expected 2 inputs, got 1'))
    expect(() => neuron.forward([1, 2, 3])).toThrow(new RangeError('Expected 2 inputs, got 3'))
  })
})

describe('Layer', () => {
  test('feeds the same inputs to every neuron', () => {
    const layer = new Layer(3, 2, 'relu')
    const inputs = [1, -2, 3]
    expect(layer.neurons).toHaveLength(2)
    expect(layer.neurons.every((neuron) => neuron.activation === 'relu')).toBe(true)
    expect(layer.forward(inputs).map((output) => output.data)).toEqual(
      layer.neurons.map((neuron) => neuron.forward(inputs).data)
    )
    expect(layer.parameters()).toHaveLength(8)
  })
})

describe('MLP', () => {
  test('chains layer sizes and keeps the output layer linear by default', () => {
    const mlp = new MLP(3, [4, 4, 1])
    expect(mlp.layers.map((layer) => [layer.neurons.length, layer.neurons[0]?.weights.length])).toEqual([
      [4, 3],
      [4, 4],
      [1, 4]
    ])
    expect(mlp.layers.map((layer) => layer.neurons[0]?.activation)).toEqual(['tanh', 'tanh', 'linear'])
    expect(mlp.parameters()).toHaveLength(4 * 4 + 4 * 5 + 5)
  })

  test('accepts custom activations', () => {
    const mlp = new MLP(2, [3, 1], { activation: 'relu', outputActivation: 'sigmoid' })
    expect(mlp.layers.map((layer) => layer.neurons[0]?.activation)).toEqual(['relu', 'sigmoid'])
  })

  test('forward returns one value per output neuron', () => {
    const outputs = new MLP(2, [3, 2]).forward([0.5, new Value(-1)])
    expect(outputs).toHaveLength(2)
    expect(outputs.every((output) => output instanceof Value)).toBe(true)
  })

  test('zeroGrad clears every parameter', () => {
    const mlp = new MLP(2, [3, 1])
    for (const output of mlp.forward([1, 2])) {
      output.backward()
    }
    mlp.zeroGrad()
    expect(mlp.parameters().every((parameter) => parameter.grad === 0)).toBe(true)
  })

  test('clipGradNorm scales the global norm down to the limit', () => {
    const mlp = new MLP(1, [1], { outputActivation: 'linear' })
    const [weight, bias] = mlp.parameters()
    if (!weight || !bias) {
      throw new Error('expected two parameters')
    }
    weight.grad = 3
    bias.grad = 4
    expect(mlp.clipGradNorm(10)).toBe(5)
    expect([weight.grad, bias.grad]).toEqual([3, 4])
    expect(mlp.clipGradNorm(1)).toBe(5)
    expect(weight.grad).toBeCloseTo(0.6, 12)
    expect(bias.grad).toBeCloseTo(0.8, 12)
  })

  test('learns a small binary classification task', () => {
    const mlp = new MLP(3, [4, 4, 1])
    const xs = [
      [2, 3, -1],
      [3, -1, 0.5],
      [0.5, 1, 1],
      [1, 1, -1]
    ]
    const ys = [1, -1, -1, 1]
    const lossOf = (): Value =>
      xs
        .map((x, index) =>
          mlp.forward(x).reduce((sum, output) => sum.add(output.sub(ys[index] ?? 0).pow(2)), new Value(0))
        )
        .reduce((sum, loss) => sum.add(loss))

    const initialLoss = lossOf().data
    for (let step = 0; step < 100; step++) {
      const loss = lossOf()
      mlp.zeroGrad()
      loss.backward()
      for (const parameter of mlp.parameters()) {
        parameter.data -= 0.05 * parameter.grad
      }
    }

    expect(lossOf().data).toBeLessThan(initialLoss / 100)
  })
})
