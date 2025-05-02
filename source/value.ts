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

  [Symbol.toPrimitive](hint: string) {
    if (hint === 'number') {
      return this.#data
    }
    return `Value(${this.#data})`
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

    const visited = new Set<string>()
    const stack: Value[] = []

    // Topological sort using stack-based DFS
    const topoSort = (node: Value) => {
      if (visited.has(node.#id)) {
        return
      }
      visited.add(node.#id)

      for (const child of node.prev()) {
        topoSort(child)
      }
      stack.push(node)
    }

    topoSort(this)
    this.grad = 1.0

    // Compute gradients in reverse order
    while (stack.length > 0) {
      const node = stack.pop()
      if (!node) {
        continue
      }

      node.#computeGradient()

      // Store higher order gradients
      if (order > 1) {
        node.#higherOrderGrads.set(1, node.grad)
        for (let i = 2; i <= order; i++) {
          node.backward(i - 1)
          node.#higherOrderGrads.set(i, node.grad)
        }
      }
    }
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
    v.#computeGradient = () => {
      valueA.grad += v.data * (1 - v.data) * v.grad
    }
    return v
  }

  static relu(a: unknown, label?: string): Value {
    const valueA = Value.from(a)
    const v = new Value(Math.max(0, valueA.data), label, [valueA], 'relu')
    v.#computeGradient = () => {
      valueA.grad += (valueA.data > 0 ? 1 : 0) * v.grad
    }
    return v
  }

  static log(a: unknown, label?: string): Value {
    const valueA = Value.from(a)
    if (valueA.data <= 0) {
      throw new Error('Log of non-positive number')
    }
    const v = new Value(Math.log(valueA.data), label, [valueA], 'log')
    v.#computeGradient = () => {
      valueA.grad += (1 / valueA.data) * v.grad
    }
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
      if (!Number.isFinite(value)) {
        throw new Error('Value must be a finite number')
      }
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
    v.#computeGradient = () => {
      value.grad += -1.0 * v.grad
    }

    return v
  }

  static add(a: unknown, b: unknown, label?: string): Value {
    const valueA = Value.from(a)
    const valueB = Value.from(b)

    const v = new Value(valueA.data + valueB.data, label, [valueA, valueB], 'add')
    v.#computeGradient = () => {
      valueA.grad += 1.0 * v.grad
      valueB.grad += 1.0 * v.grad
    }

    return v
  }

  add(b: unknown, label?: string): Value {
    return Value.add(this, b, label)
  }

  static sub(a: unknown, b: unknown, label?: string): Value {
    const valueA = Value.from(a)
    const valueB = Value.from(b)

    const v = new Value(valueA.data - valueB.data, label, [valueA, valueB], 'sub')
    v.#computeGradient = () => {
      valueA.grad += 1.0 * v.grad
      valueB.grad += -1.0 * v.grad
    }

    return v
  }

  sub(b: unknown, label?: string): Value {
    return Value.sub(this, b, label)
  }

  static mul(a: unknown, b: unknown, label?: string): Value {
    const valueA = Value.from(a)
    const valueB = Value.from(b)

    const v = new Value(valueA.data * valueB.data, label, [valueA, valueB], 'mul')
    v.#computeGradient = () => {
      valueA.grad += valueB.data * v.grad
      valueB.grad += valueA.data * v.grad
    }

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
    v.#computeGradient = () => {
      valueA.grad += (1.0 / valueB.data) * v.grad
      valueB.grad += (-valueA.data / (valueB.data * valueB.data)) * v.grad
    }

    return v
  }

  div(b: unknown, label?: string): Value {
    return Value.div(this, b, label)
  }

  static exp(a: unknown, label?: string): Value {
    const valueA = Value.from(a)

    const v = new Value(Math.exp(valueA.data), label, [valueA], 'exp')
    v.#computeGradient = () => {
      valueA.grad += v.data * v.grad
    }

    return v
  }

  exp(): Value {
    return Value.exp(this)
  }

  static pow(a: unknown, b: unknown, label?: string): Value {
    const valueA = Value.from(a)
    const valueB = Value.from(b)

    if (valueA.data === 0) {
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
    v.#computeGradient = () => {
      if (Math.abs(valueA.data) <= Number.EPSILON) {
        // If valueA is effectively zero
        if (valueB.data <= 0) {
          throw new Error('Cannot raise 0 to zero or negative power')
        }
        valueA.grad += 0 // Derivative of 0^x for x > 0 is 0
        valueB.grad += 0 // Derivative with respect to exponent is also 0
      } else {
        valueA.grad += valueB.data * valueA.data ** (valueB.data - 1) * v.grad
        valueB.grad += valueA.data ** valueB.data * Math.log(Math.abs(valueA.data)) * v.grad
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
    v.#computeGradient = () => {
      valueA.#grad += (1.0 - v.#data ** 2) * v.#grad
    }

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
