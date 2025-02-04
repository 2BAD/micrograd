export class Value {
  readonly data: number
  readonly grad: number
  readonly children: Value[]
  readonly operation: string
  readonly label: string

  constructor(data: number, label?: string, children?: Value[], operation?: string) {
    this.data = data
    this.grad = 0.0
    this.children = children ?? []
    this.operation = operation ?? '+'
    this.label = label ?? ''
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

  static from(value: unknown): Value {
    if (value instanceof Value) {
      return value
    }
    return new Value(Number(value))
  }

  static add(a: Value, b: Value, label?: string): Value {
    return new Value(a.data + b.data, label, [a, b], '+')
  }

  static multiply(a: Value, b: Value, label?: string): Value {
    return new Value(a.data * b.data, label, [a, b], '*')
  }

  static tanh(a: Value, label?: string): Value {
    return new Value(Math.tanh(a.data), label, [a], 'tanh')
  }
}
