export class Value {
  readonly data: number
  readonly grad: number
  readonly children: Value[]
  readonly operation: string

  constructor(data: number, children?: Value[], operation?: string) {
    this.data = data
    this.grad = 0.0
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

  static from(value: unknown): Value {
    if (value instanceof Value) {
      return value
    }
    return new Value(Number(value))
  }

  static add(a: Value, b: Value): Value {
    return new Value(a.data + b.data, [a, b], '+')
  }

  static multiply(a: Value, b: Value): Value {
    return new Value(a.data * b.data, [a, b], '*')
  }
}
