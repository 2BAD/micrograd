export type Operand = Value | number

type Edge = readonly [input: Value, localGradient: (output: Value) => Operand]

export class Value {
  data: number
  grad = 0
  label: string
  #op = ''
  #edges: readonly Edge[] = []

  constructor(data: number, label = '') {
    if (!Number.isFinite(data)) {
      throw new RangeError(`Value must be a finite number, got ${data}`)
    }
    this.data = data
    this.label = label
  }

  get op(): string {
    return this.#op
  }

  get children(): Value[] {
    return this.#edges.map(([input]) => input)
  }

  static #from(operand: Operand): Value {
    return operand instanceof Value ? operand : new Value(operand)
  }

  static #result(data: number, op: string, edges: readonly Edge[]): Value {
    if (!Number.isFinite(data)) {
      throw new RangeError(`${op} produced ${data}`)
    }
    const output = new Value(data)
    output.#op = op
    output.#edges = edges
    return output
  }

  add(other: Operand): Value {
    const that = Value.#from(other)
    return Value.#result(this.data + that.data, '+', [
      [this, () => 1],
      [that, () => 1]
    ])
  }

  sub(other: Operand): Value {
    const that = Value.#from(other)
    return Value.#result(this.data - that.data, '-', [
      [this, () => 1],
      [that, () => -1]
    ])
  }

  mul(other: Operand): Value {
    const that = Value.#from(other)
    return Value.#result(this.data * that.data, '*', [
      [this, () => that],
      [that, () => this]
    ])
  }

  div(other: Operand): Value {
    const that = Value.#from(other)
    return Value.#result(this.data / that.data, '/', [
      [this, () => that.pow(-1)],
      [that, (output) => output.div(that).neg()]
    ])
  }

  pow(exponent: number): Value {
    return Value.#result(this.data ** exponent, `^${exponent}`, [
      [this, () => (exponent === 0 ? 0 : this.pow(exponent - 1).mul(exponent))]
    ])
  }

  neg(): Value {
    return Value.#result(-this.data, 'neg', [[this, () => -1]])
  }

  exp(): Value {
    return Value.#result(Math.exp(this.data), 'exp', [[this, (output) => output]])
  }

  log(): Value {
    return Value.#result(Math.log(this.data), 'log', [[this, () => this.pow(-1)]])
  }

  tanh(): Value {
    return Value.#result(Math.tanh(this.data), 'tanh', [[this, (output) => output.mul(output).neg().add(1)]])
  }

  sigmoid(): Value {
    return Value.#result(1 / (1 + Math.exp(-this.data)), 'sigmoid', [
      [this, (output) => output.mul(output.neg().add(1))]
    ])
  }

  relu(): Value {
    return Value.#result(Math.max(0, this.data), 'relu', [[this, () => (this.data > 0 ? 1 : 0)]])
  }

  backward(): void {
    const upstream = new Map<Value, number>([[this, 1]])
    for (const node of this.#topologicalOrder()) {
      const gradient = upstream.get(node) ?? 0
      node.grad += gradient
      if (!Number.isFinite(node.grad)) {
        throw new RangeError(`backward produced ${node.grad}`)
      }
      if (gradient === 0) {
        continue
      }
      for (const [input, localGradient] of node.#edges) {
        const local = localGradient(node)
        upstream.set(input, (upstream.get(input) ?? 0) + gradient * (local instanceof Value ? local.data : local))
      }
    }
  }

  gradients<const T extends readonly Value[]>(inputs: T): { [K in keyof T]: Value } {
    const upstream = new Map<Value, Value>([[this, new Value(1)]])
    for (const node of this.#topologicalOrder()) {
      const gradient = upstream.get(node)
      if (gradient === undefined) {
        continue
      }
      for (const [input, localGradient] of node.#edges) {
        const contribution = gradient.mul(localGradient(node))
        const accumulated = upstream.get(input)
        upstream.set(input, accumulated === undefined ? contribution : accumulated.add(contribution))
      }
    }
    return inputs.map((input) => upstream.get(input) ?? new Value(0)) as { [K in keyof T]: Value }
  }

  zeroGrad(): void {
    for (const node of this.#topologicalOrder()) {
      node.grad = 0
    }
  }

  toString(): string {
    return `Value(data=${this.data}, grad=${this.grad})`
  }

  #topologicalOrder(): Value[] {
    const order: Value[] = []
    const visited = new Set<Value>()
    const stack: [node: Value, expanded: boolean][] = [[this, false]]
    for (let entry = stack.pop(); entry; entry = stack.pop()) {
      const [node, expanded] = entry
      if (expanded) {
        order.push(node)
      } else if (!visited.has(node)) {
        visited.add(node)
        stack.push([node, true])
        for (const [input] of node.#edges) {
          stack.push([input, false])
        }
      }
    }
    return order.reverse()
  }
}
