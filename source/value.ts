/* eslint-disable jsdoc/require-jsdoc */
export class Value {
  public grad: number
  public computeGradient: () => void

  readonly data: number
  readonly label: string
  readonly children: Value[]
  readonly operation: string

  constructor(data: number, label?: string, children?: Value[], operation?: string) {
    this.grad = 0.0
    this.computeGradient = () => undefined

    this.data = data
    this.label = label ?? ''
    this.children = children ?? []
    this.operation = operation ?? '+'
  }

  [Symbol.toPrimitive](hint: string) {
    if (hint === 'number') {
      return this.data
    }
    return `Value(${this.data})`
  }

  prev(): Value[] {
    return Array.from(new Set(this.children))
  }

  backward(): void {
    const sortedNodes: Value[] = []
    const visited = new Set()

    const dfs = (v: Value) => {
      if (!visited.has(v)) {
        visited.add(v)
        for (const child of v.prev()) {
          dfs(child)
        }
        sortedNodes.push(v)
      }
    }

    dfs(this)
    this.grad = 1.0

    for (const v of sortedNodes.reverse()) {
      v.computeGradient()
    }
  }

  static from(value: unknown): Value {
    if (value instanceof Value) {
      return value
    }

    if (Number.isNaN(value)) {
      throw new Error('Unsupported value')
    }

    return new Value(Number(value))
  }

  static add(a: unknown, b: unknown, label?: string): Value {
    const valueA = Value.from(a)
    const valueB = Value.from(b)

    const v = new Value(valueA.data + valueB.data, label, [valueA, valueB], 'add')
    v.computeGradient = () => {
      valueA.grad += 1.0 * v.grad
      valueB.grad += 1.0 * v.grad
    }

    return v
  }

  static multiply(a: unknown, b: unknown, label?: string): Value {
    const valueA = Value.from(a)
    const valueB = Value.from(b)

    const v = new Value(valueA.data * valueB.data, label, [valueA, valueB], 'mul')
    v.computeGradient = () => {
      valueA.grad += valueB.data * v.grad
      valueB.grad += valueA.data * v.grad
    }

    return v
  }

  static exp(a: unknown, label?: string): Value {
    const valueA = Value.from(a)

    const v = new Value(Math.exp(valueA.data), label, [valueA], 'exp')
    v.computeGradient = () => {
      valueA.grad += v.data * v.grad
    }

    return v
  }

  static pow(a: unknown, b: unknown, label?: string): Value {
    const valueA = Value.from(a)
    const valueB = Value.from(b)

    const v = new Value(valueA.data ** valueB.data, label, [valueA, valueB], 'pow')
    v.computeGradient = () => {
      valueA.grad += valueB.data * valueA.data ** (valueB.data - 1) * v.grad
      valueB.grad += valueA.data ** valueB.data * Math.log(valueA.data) * v.grad
    }

    return v
  }

  static tanh(a: unknown, label?: string): Value {
    const valueA = Value.from(a)

    const v = new Value(Math.tanh(valueA.data), label, [valueA], 'tanh')
    v.computeGradient = () => {
      valueA.grad += (1 - v.data ** 2) * v.grad
    }

    return v
  }
}
