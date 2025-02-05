/* eslint-disable jsdoc/require-jsdoc */
export class Value {
  readonly #id: string
  #data: number
  #grad: number
  #computeGradient: () => void
  readonly #higherOrderGrads: Map<number, number>

  readonly label: string
  readonly children: Value[]
  readonly operation: string

  static #instanceCounter = 0
  static readonly EPSILON = 1e-10

  constructor(data: number, label?: string, children?: Value[], operation?: string) {
    this.validateNumber(data)

    this.#id = `value_${Value.#instanceCounter++}`
    this.#data = data
    this.#grad = 0.0
    this.#computeGradient = () => undefined
    this.#higherOrderGrads = new Map()

    this.label = label ?? ''
    this.children = children ?? []
    this.operation = operation ?? '+'
  }

  private validateNumber(value: number): void {
    if (!Number.isFinite(value)) {
      throw new Error('Value must be a finite number')
    }
  }

  get id(): string {
    return this.#id
  }

  get data(): number {
    return this.#data
  }

  set data(value: number) {
    this.validateNumber(value)
    this.#data = value
  }

  get grad(): number {
    return this.#grad
  }

  set grad(value: number) {
    this.validateNumber(value)
    this.#grad = value
  }

  prev(): Value[] {
    return Array.from(new Set(this.children))
  }

  resetGrad(): void {
    this.#grad = 0
    for (const child of this.children) {
      child.resetGrad()
    }
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
    this.#grad = 1.0

    for (const v of sortedNodes.reverse()) {
      v.computeGradient()
    }
  }

  static from(value: unknown): Value {
    // Handle Value instances
    if (value instanceof Value) {
      return value
    }

    // Handle numbers directly
    if (typeof value === 'number') {
      if (!Number.isFinite(value)) {
        throw new Error('Value must be a finite number')
      }
      return new Value(value)
    }

    // Handle string conversion
    if (typeof value === 'string') {
      const trimmed = value.trim()
      const number = Number(trimmed)

      // Check if the string is a valid number representation
      if (!Number.isFinite(number) || !/^[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?$/.test(trimmed)) {
        throw new Error('Invalid number format')
      }

      return new Value(number)
    }

    // Handle boolean values
    if (typeof value === 'boolean') {
      return new Value(value ? 1 : 0)
    }

    // Handle null and undefined
    if (value === null || value === undefined) {
      throw new Error('Cannot create Value from null or undefined')
    }

    // Handle arrays with single numeric value
    if (Array.isArray(value)) {
      if (value.length !== 1) {
        throw new Error('Arrays must contain exactly one numeric value')
      }
      return Value.from(value[0])
    }

    throw new Error(`Cannot convert ${typeof value} to Value`)
  }

  static negate = (a: unknown, label?: string): Value => {
    const value = Value.from(a)

    const v = new Value(value.#data * -1, label, [value], 'neg')
    v.computeGradient = () => {
      value.#grad += -1.0 * v.#grad
    }

    return v
  }

  static add(a: unknown, b: unknown, label?: string): Value {
    const valueA = Value.from(a)
    const valueB = Value.from(b)

    const v = new Value(valueA.#data + valueB.#data, label, [valueA, valueB], 'add')
    v.computeGradient = () => {
      valueA.#grad += 1.0 * v.#grad
      valueB.#grad += 1.0 * v.#grad
    }

    return v
  }

  add(b: unknown, label?: string): Value {
    return Value.add(this, b, label)
  }

  static sub(a: unknown, b: unknown, label?: string): Value {
    const valueA = Value.from(a)
    const valueB = Value.from(b)

    const v = new Value(valueA.#data - valueB.#data, label, [valueA, valueB], 'sub')
    v.computeGradient = () => {
      valueA.#grad += 1.0 * v.#grad
      valueB.#grad += -1.0 * v.#grad
    }

    return v
  }

  sub(b: unknown, label?: string): Value {
    return Value.sub(this, b, label)
  }

  static mul(a: unknown, b: unknown, label?: string): Value {
    const valueA = Value.from(a)
    const valueB = Value.from(b)

    const v = new Value(valueA.#data * valueB.#data, label, [valueA, valueB], 'mul')
    v.computeGradient = () => {
      valueA.#grad += valueB.#data * v.#grad
      valueB.#grad += valueA.#data * v.#grad
    }

    return v
  }

  mul(b: unknown, label?: string): Value {
    return Value.mul(this, b, label)
  }

  static div(a: unknown, b: unknown, label?: string): Value {
    const valueA = Value.from(a)
    const valueB = Value.from(b)
    if (valueB.#data === 0) {
      throw new Error('Division by zero')
    }

    const v = new Value(valueA.#data / valueB.#data, label, [valueA, valueB], 'div')
    v.computeGradient = () => {
      valueA.#grad += (1.0 / valueB.#data) * v.#grad
      valueB.#grad += (-valueA.#data / (valueB.#data * valueB.#data)) * v.#grad
    }

    return v
  }

  div(b: unknown, label?: string): Value {
    return Value.div(this, b, label)
  }

  static exp(a: unknown, label?: string): Value {
    const valueA = Value.from(a)

    const v = new Value(Math.exp(valueA.#data), label, [valueA], 'exp')
    v.computeGradient = () => {
      valueA.#grad += v.#data * v.#grad
    }

    return v
  }

  exp(): Value {
    return Value.exp(this)
  }

  static pow(a: unknown, b: unknown, label?: string): Value {
    const valueA = Value.from(a)
    const valueB = Value.from(b)

    if (valueA.#data < 0 && !Number.isInteger(valueB.#data)) {
      throw new Error('Negative numbers cannot be raised to non-integer powers')
    }

    const v = new Value(valueA.#data ** valueB.#data, label, [valueA, valueB], 'pow')
    v.computeGradient = () => {
      if (valueA.#data === 0) {
        if (valueB.#data > 0) {
          valueA.#grad += 0 // Derivative of 0^x for x > 0 is 0
          valueB.#grad += 0 // Derivative with respect to exponent is also 0
        } else {
          throw new Error('Cannot raise 0 to zero or negative power')
        }
      } else {
        valueA.#grad += valueB.#data * valueA.#data ** (valueB.#data - 1) * v.#grad
        valueB.#grad += valueA.#data ** valueB.#data * Math.log(Math.abs(valueA.#data)) * v.#grad
      }
    }

    return v
  }

  pow(b: unknown, label?: string): Value {
    return Value.pow(this, b, label)
  }

  static tanh(a: unknown, label?: string): Value {
    const valueA = Value.from(a)

    const v = new Value(Math.tanh(valueA.#data), label, [valueA], 'tanh')
    v.computeGradient = () => {
      valueA.#grad += (1.0 - v.#data ** 2) * v.#grad
    }

    return v
  }

  tanh(): Value {
    return Value.tanh(this)
  }
}
