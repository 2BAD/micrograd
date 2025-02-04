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
    return new Value(Number(value))
  }

  static add(a: Value, b: Value, label?: string): Value {
    const v = new Value(a.data + b.data, label, [a, b], '+')
    v.computeGradient = () => {
      a.grad += 1.0 * v.grad
      b.grad += 1.0 * v.grad
    }

    return v
  }

  static multiply(a: Value, b: Value, label?: string): Value {
    const v = new Value(a.data * b.data, label, [a, b], '*')
    v.computeGradient = () => {
      a.grad += b.data * v.grad
      b.grad += a.data * v.grad
    }

    return v
  }

  static tanh(a: Value, label?: string): Value {
    const v = new Value(Math.tanh(a.data), label, [a], 'tanh')

    v.computeGradient = () => {
      a.grad += (1 - v.data ** 2) * v.grad
    }

    return v
  }
}
