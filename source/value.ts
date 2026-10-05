export class Value {
  readonly #id: string
  #data: number
  #grad: number
  #chainRule: (outputGrad: Value) => [Value, Value][]
  readonly #higherOrderGrads: Map<number, number>

  readonly label: string
  readonly children: Value[]
  readonly operation: string

  static #instanceCounter = 0

  constructor(data: number, label?: string, children?: Value[], operation?: string) {
    Value.validateNumber(data)

    this.#id = `value_${Value.#instanceCounter++}`
    this.#data = data
    this.#grad = 0.0
    this.#chainRule = () => []
    this.#higherOrderGrads = new Map()

    this.label = label ?? ''
    this.children = children ?? []
    this.operation = operation ?? '+'
  }

  [Symbol.toPrimitive](hint: string) {
    if (hint === 'number') {
      return this.#data
    }
    return `Value(${this.#data})`
  }

  static validateNumber(value: number): void {
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
    Value.validateNumber(value)
    this.#data = value
  }

  get grad(): number {
    return this.#grad
  }

  set grad(value: number) {
    Value.validateNumber(value)
    this.#grad = value
  }

  prev(): Value[] {
    return Array.from(new Set(this.children))
  }

  resetGrad(): void {
    const visited = new Set<string>()

    const resetGradHelper = (node: Value) => {
      if (visited.has(node.#id)) {
        return
      }
      visited.add(node.#id)

      node.grad = 0
      node.#higherOrderGrads.clear()
      for (const child of node.children) {
        resetGradHelper(child)
      }
    }

    resetGradHelper(this)
  }

  backward(order = 1): void {
    if (order < 1) {
      throw new Error('Order must be >= 1')
    }

    for (const [node, gradient] of this.#gradientGraphs()) {
      node.grad = node === this ? 1 : node.grad + gradient.data

      if (order > 1) {
        let derivative = gradient
        node.#higherOrderGrads.set(1, derivative.data)
        for (let i = 2; i <= order; i++) {
          derivative = derivative.#gradientGraphs().get(node) ?? new Value(0)
          node.#higherOrderGrads.set(i, derivative.data)
        }
      }
    }
  }

  #gradientGraphs(): Map<Value, Value> {
    const visited = new Set<Value>()
    const order: Value[] = []

    const visit = (node: Value) => {
      if (visited.has(node)) {
        return
      }
      visited.add(node)
      for (const child of node.prev()) {
        visit(child)
      }
      order.push(node)
    }

    visit(this)

    const gradients = new Map<Value, Value>([[this, new Value(1)]])
    for (const node of order.reverse()) {
      const outputGrad = gradients.get(node)
      if (!outputGrad) {
        continue
      }
      for (const [child, contribution] of node.#chainRule(outputGrad)) {
        const existing = gradients.get(child)
        gradients.set(child, existing ? existing.add(contribution) : contribution)
      }
    }

    return gradients
  }

  getHigherOrderGradient(order: number): number {
    if (order < 1) {
      throw new Error('Order must be >= 1')
    }
    return this.#higherOrderGrads.get(order) ?? 0
  }

  // Neural network activation functions
  static sigmoid(a: unknown, label?: string): Value {
    const valueA = Value.from(a)
    const v = new Value(1 / (1 + Math.exp(-valueA.data)), label, [valueA], 'sigmoid')
    v.#chainRule = (outputGrad) => [[valueA, outputGrad.mul(v).mul(Value.sub(1, v))]]
    return v
  }

  static relu(a: unknown, label?: string): Value {
    const valueA = Value.from(a)
    const v = new Value(Math.max(0, valueA.data), label, [valueA], 'relu')
    v.#chainRule = (outputGrad) => [[valueA, outputGrad.mul(valueA.data > 0 ? 1 : 0)]]
    return v
  }

  static log(a: unknown, label?: string): Value {
    const valueA = Value.from(a)
    if (valueA.data <= 0) {
      throw new Error('Log of non-positive number')
    }
    const v = new Value(Math.log(valueA.data), label, [valueA], 'log')
    v.#chainRule = (outputGrad) => [[valueA, outputGrad.div(valueA)]]
    return v
  }

  // Improved static from with better type guards
  static from(value: unknown): Value {
    // Handle Value instances
    if (value instanceof Value) {
      return value
    }

    // Handle numbers directly
    if (typeof value === 'number') {
      Value.validateNumber(value)
      return new Value(value)
    }

    // Handle string conversion
    if (typeof value === 'string') {
      const trimmed = value.trim()
      const number = Number(value.trim())
      if (!Number.isFinite(number) || trimmed.length === 0) {
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

    const v = new Value(value.data * -1, label, [value], 'neg')
    v.#chainRule = (outputGrad) => [[value, Value.negate(outputGrad)]]

    return v
  }

  static add(a: unknown, b: unknown, label?: string): Value {
    const valueA = Value.from(a)
    const valueB = Value.from(b)

    const v = new Value(valueA.data + valueB.data, label, [valueA, valueB], 'add')
    v.#chainRule = (outputGrad) => [
      [valueA, outputGrad],
      [valueB, outputGrad]
    ]

    return v
  }

  add(b: unknown, label?: string): Value {
    return Value.add(this, b, label)
  }

  static sub(a: unknown, b: unknown, label?: string): Value {
    const valueA = Value.from(a)
    const valueB = Value.from(b)

    const v = new Value(valueA.data - valueB.data, label, [valueA, valueB], 'sub')
    v.#chainRule = (outputGrad) => [
      [valueA, outputGrad],
      [valueB, Value.negate(outputGrad)]
    ]

    return v
  }

  sub(b: unknown, label?: string): Value {
    return Value.sub(this, b, label)
  }

  static mul(a: unknown, b: unknown, label?: string): Value {
    const valueA = Value.from(a)
    const valueB = Value.from(b)

    const v = new Value(valueA.data * valueB.data, label, [valueA, valueB], 'mul')
    v.#chainRule = (outputGrad) => [
      [valueA, outputGrad.mul(valueB)],
      [valueB, outputGrad.mul(valueA)]
    ]

    return v
  }

  mul(b: unknown, label?: string): Value {
    return Value.mul(this, b, label)
  }

  static div(a: unknown, b: unknown, label?: string): Value {
    const valueA = Value.from(a)
    const valueB = Value.from(b)

    if (Math.abs(valueB.data) < Number.EPSILON) {
      throw new Error('Division by near-zero value')
    }

    const v = new Value(valueA.data / valueB.data, label, [valueA, valueB], 'div')
    v.#chainRule = (outputGrad) => [
      [valueA, outputGrad.div(valueB)],
      [valueB, Value.negate(outputGrad.mul(valueA).div(valueB).div(valueB))]
    ]

    return v
  }

  div(b: unknown, label?: string): Value {
    return Value.div(this, b, label)
  }

  static exp(a: unknown, label?: string): Value {
    const valueA = Value.from(a)

    const v = new Value(Math.exp(valueA.data), label, [valueA], 'exp')
    v.#chainRule = (outputGrad) => [[valueA, outputGrad.mul(v)]]

    return v
  }

  exp(): Value {
    return Value.exp(this)
  }

  static pow(a: unknown, b: unknown, label?: string): Value {
    const valueA = Value.from(a)
    const valueB = Value.from(b)

    if (Math.abs(valueA.data) <= Number.EPSILON) {
      // If valueA is effectively zero
      if (valueB.data === 0) {
        throw new Error('Cannot raise 0 to zero or negative power')
      }
      if (valueB.data < 0) {
        throw new Error('Division by zero in power operation')
      }
      return new Value(0, label)
    }

    if (valueA.data < 0 && !Number.isInteger(valueB.data)) {
      throw new Error('Negative numbers cannot be raised to non-integer powers')
    }

    const result = valueA.data ** valueB.data
    if (!Number.isFinite(result)) {
      throw new Error('Power operation resulted in overflow')
    }

    const v = new Value(result, label, [valueA, valueB], 'pow')
    v.#chainRule = (outputGrad) => [
      [valueA, outputGrad.mul(valueB).mul(valueA.pow(valueB.sub(1)))],
      [valueB, outputGrad.mul(v).mul(valueA.data > 0 ? Value.log(valueA) : Math.log(Math.abs(valueA.data)))]
    ]

    return v
  }

  pow(b: unknown, label?: string): Value {
    return Value.pow(this, b, label)
  }

  static tanh(a: unknown, label?: string): Value {
    const valueA = Value.from(a)

    const v = new Value(Math.tanh(valueA.#data), label, [valueA], 'tanh')
    v.#chainRule = (outputGrad) => [[valueA, outputGrad.mul(Value.sub(1, v.mul(v)))]]

    return v
  }

  tanh(): Value {
    return Value.tanh(this)
  }

  // Gradient clipping to prevent explosion
  clipGradients(maxNorm: number): void {
    const visited = new Set<string>()

    const clipGradsHelper = (node: Value) => {
      if (visited.has(node.#id)) {
        return
      }
      visited.add(node.#id)

      const gradNorm = Math.abs(node.grad)
      if (gradNorm > maxNorm) {
        node.grad *= maxNorm / gradNorm
      }

      for (const child of node.children) {
        clipGradsHelper(child)
      }
    }

    clipGradsHelper(this)
  }

  // Helper method to detect gradient issues
  checkGradientHealth(): {
    hasExploding: boolean
    hasVanishing: boolean
    maxGrad: number
    minGrad: number
  } {
    const visited = new Set<string>()
    let maxGrad = Number.NEGATIVE_INFINITY
    let minGrad = Number.POSITIVE_INFINITY

    const checkGrads = (node: Value) => {
      if (visited.has(node.#id)) {
        return
      }
      visited.add(node.#id)

      if (node.grad !== 0) {
        maxGrad = Math.max(maxGrad, Math.abs(node.grad))
        minGrad = Math.min(minGrad, Math.abs(node.grad))
      }

      for (const child of node.children) {
        checkGrads(child)
      }
    }

    checkGrads(this)

    return {
      hasExploding: maxGrad > 1e3,
      hasVanishing: minGrad < 1e-3,
      maxGrad,
      minGrad
    }
  }
}
