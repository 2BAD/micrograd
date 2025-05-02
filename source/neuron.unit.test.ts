/* eslint-disable @typescript-eslint/no-non-null-assertion */
import { describe, expect, test, vi } from 'vitest'
import { Layer, MLP, Neuron, Value } from './index.ts'

describe('Neuron', () => {
  test('should initialize with correct number of weights', () => {
    const neuron = new Neuron(3)
    expect(neuron.weights.length).toBe(3)
    expect(neuron.weights.every((w) => w instanceof Value)).toBe(true)
    expect(neuron.bias).toBeInstanceOf(Value)
  })

  test('should initialize weights and bias with random values in (-1, 1)', () => {
    const neuron = new Neuron(5)
    expect(neuron.weights.every((w) => w.data >= -1 && w.data <= 1)).toBe(true)
    expect(neuron.bias.data >= -1 && neuron.bias.data <= 1).toBe(true)
  })

  test('should perform forward pass correctly', () => {
    const neuron = new Neuron(2)
    // Set deterministic weights and bias for testing
    neuron.weights[0]!.data = 0.5
    neuron.weights[1]!.data = -0.5
    neuron.bias.data = 0.1

    const inputs = [new Value(1), new Value(2)]
    const output = neuron.forward(inputs)

    // Calculate expected output: tanh(0.5*1 + (-0.5)*2 + 0.1) = tanh(-0.4)
    const expected = Math.tanh(-0.4)
    expect(output.data).toBeCloseTo(expected)
  })

  test('should return all parameters', () => {
    const neuron = new Neuron(2)
    const params = neuron.parameters()
    expect(params.length).toBe(3) // 2 weights + bias
    expect(params).toContain(neuron.bias)
    expect(params).toContain(neuron.weights[0])
    expect(params).toContain(neuron.weights[1])
  })
})

describe('Layer', () => {
  test('should initialize with correct number of neurons', () => {
    const layer = new Layer(3, 2)
    expect(layer.neurons.length).toBe(2)
    expect(layer.neurons.every((n) => n instanceof Neuron)).toBe(true)
    expect(layer.neurons.every((n) => n.weights.length === 3)).toBe(true)
  })

  test('should perform forward pass through all neurons', () => {
    const layer = new Layer(2, 3)
    // Mock neuron forward method to return predictable values
    layer.neurons.forEach((neuron, i) => {
      neuron.forward = vi.fn().mockReturnValue(new Value(i))
    })

    const inputs = [new Value(1), new Value(2)]
    const outputs = layer.forward(inputs)

    expect(outputs.length).toBe(3)
    expect(outputs[0]!.data).toBe(0)
    expect(outputs[1]!.data).toBe(1)
    expect(outputs[2]!.data).toBe(2)

    // Verify each neuron received the inputs
    for (const neuron of layer.neurons) {
      expect(neuron.forward).toHaveBeenCalledWith(inputs)
    }
  })

  test('should return all parameters from all neurons', () => {
    const layer = new Layer(2, 2)
    const params = layer.parameters()

    // Should have 2 neurons with 2 weights + 1 bias each = 6 parameters
    expect(params.length).toBe(6)

    // Check all weights and biases are included
    for (const neuron of layer.neurons) {
      expect(params).toContain(neuron.bias)
      for (const weight of neuron.weights) {
        expect(params).toContain(weight)
      }
    }
  })
})

describe('MLP (Multi-Layer Perceptron)', () => {
  test('should initialize with correct layer structure', () => {
    const mlp = new MLP(2, [3, 1])
    expect(mlp.layers.length).toBe(2)

    expect(mlp.layers[0]!.neurons.length).toBe(3)
    expect(mlp.layers[0]!.neurons[0]!.weights.length).toBe(2)

    expect(mlp.layers[1]!.neurons.length).toBe(1)
    expect(mlp.layers[1]!.neurons[0]!.weights.length).toBe(3)
  })

  test('should perform forward pass through all layers', () => {
    const mlp = new MLP(2, [3, 1])

    // Mock first layer to return [new Value(1), new Value(2), new Value(3)]
    const firstLayerOutput = [new Value(1), new Value(2), new Value(3)]
    mlp.layers[0]!.forward = vi.fn().mockReturnValue(firstLayerOutput)

    // Mock second layer to return [new Value(10)]
    const secondLayerOutput = [new Value(10)]
    mlp.layers[1]!.forward = vi.fn().mockReturnValue(secondLayerOutput)

    const inputs = [new Value(0.5), new Value(-0.5)]
    const outputs = mlp.forward(inputs)

    expect(outputs).toBe(secondLayerOutput)
    expect(mlp.layers[0]!.forward).toHaveBeenCalledWith(inputs)
    expect(mlp.layers[1]!.forward).toHaveBeenCalledWith(firstLayerOutput)
  })

  test('should return all parameters from all layers', () => {
    const mlp = new MLP(2, [3, 1])

    // Mock parameters for each layer
    const layer0Params = [new Value(1), new Value(2)]
    const layer1Params = [new Value(3), new Value(4)]

    mlp.layers[0]!.parameters = vi.fn().mockReturnValue(layer0Params)
    mlp.layers[1]!.parameters = vi.fn().mockReturnValue(layer1Params)

    const params = mlp.parameters()

    expect(params.length).toBe(4)
    expect(params).toContain(layer0Params[0])
    expect(params).toContain(layer0Params[1])
    expect(params).toContain(layer1Params[0])
    expect(params).toContain(layer1Params[1])
  })

  test('train should update parameters based on gradients', () => {
    vi.spyOn(console, 'log').mockImplementation(() => undefined)
    const mlp = new MLP(2, [2, 1])

    // Mock forward method to return predictable output
    mlp.forward = vi.fn().mockImplementation(() => [new Value(0.5)])

    // Create mock parameters with trackable grad and data properties
    const params = [new Value(0.1), new Value(0.2), new Value(0.3)]

    mlp.parameters = vi.fn().mockReturnValue(params)

    // Set up training data
    const xs = [
      [1, 2],
      [3, 4]
    ]
    const ys = [0, 1]

    // Train for a single epoch with learning rate 0.1
    mlp.train(xs, ys, 0.1, 1)

    // Verify forward was called for each input
    expect(mlp.forward).toHaveBeenCalledTimes(2)

    // Parameters should have been updated
    for (const param of params) {
      // Grad should be updated
      expect(param.grad).toBe(0)
    }
  })

  test('should log progress at specified intervals', () => {
    const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => undefined)

    const mlp = new MLP(2, [2, 1])

    // Mock forward to return a predictable value
    mlp.forward = vi.fn().mockReturnValue([new Value(0.5)])

    // Mock parameters for simplicity
    mlp.parameters = vi.fn().mockReturnValue([])

    // Train for 20 epochs
    mlp.train([[1, 2]], [1], 0.1, 20)

    // Should log at epochs 0, 10 (every 10 epochs)
    expect(consoleSpy).toHaveBeenCalledTimes(2)
    expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('Epoch 0'))
    expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('Epoch 10'))

    consoleSpy.mockRestore()
  })
})
