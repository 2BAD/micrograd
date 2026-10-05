import { describe, expect, test } from 'vitest'
import { Value } from './value.ts'

const STEP = 1e-5

const finiteDifference = (fn: (x: number) => number, at: number): number => (fn(at + STEP) - fn(at - STEP)) / (2 * STEP)

const UNARY: [name: string, fn: (x: Value) => Value, at: number][] = [
  ['add', (x) => x.add(3), 1.5],
  ['add self', (x) => x.add(x), 1.5],
  ['sub', (x) => x.sub(3), 1.5],
  ['sub reversed', (x) => new Value(3).sub(x), 1.5],
  ['mul', (x) => x.mul(-2.5), 1.5],
  ['mul self', (x) => x.mul(x), 1.5],
  ['div', (x) => x.div(4), 1.5],
  ['div reversed', (x) => new Value(4).div(x), 1.5],
  ['pow', (x) => x.pow(3), -1.5],
  ['pow fractional', (x) => x.pow(0.5), 2],
  ['pow negative', (x) => x.pow(-2), 1.5],
  ['neg', (x) => x.neg(), 1.5],
  ['exp', (x) => x.exp(), 0.7],
  ['log', (x) => x.log(), 0.7],
  ['tanh', (x) => x.tanh(), 0.4],
  ['sigmoid', (x) => x.sigmoid(), 0.4],
  ['relu positive', (x) => x.relu(), 0.4],
  ['relu negative', (x) => x.relu(), -0.4],
  ['composite', (x) => x.mul(x).add(x.exp()).div(x.tanh().add(2)).log(), 0.8]
]

describe('Value', () => {
  test('creates a leaf with no op and no children', () => {
    const value = new Value(5, 'a')
    expect(value.data).toBe(5)
    expect(value.grad).toBe(0)
    expect(value.label).toBe('a')
    expect(value.op).toBe('')
    expect(value.children).toEqual([])
  })

  test.each([Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY])('rejects %s', (data) => {
    expect(() => new Value(data)).toThrow(RangeError)
  })

  test('records op and children', () => {
    const a = new Value(2)
    const b = new Value(3)
    const c = a.mul(b)
    expect(c.op).toBe('*')
    expect(c.children).toEqual([a, b])
    expect(a.pow(2).op).toBe('^2')
  })

  test('wraps number operands as constant leaves', () => {
    const c = new Value(2).add(3)
    expect(c.data).toBe(5)
    expect(c.children[1]?.data).toBe(3)
  })

  test.each([
    ['div by zero', () => new Value(1).div(0), '/ produced Infinity'],
    ['zero div by zero', () => new Value(0).div(0), '/ produced NaN'],
    ['log of zero', () => new Value(0).log(), 'log produced -Infinity'],
    ['log of negative', () => new Value(-1).log(), 'log produced NaN'],
    ['negative base with fractional exponent', () => new Value(-2).pow(0.5), '^0.5 produced NaN'],
    ['zero to a negative power', () => new Value(0).pow(-1), '^-1 produced Infinity'],
    ['exp overflow', () => new Value(1000).exp(), 'exp produced Infinity']
  ])('throws on %s', (_, operation, message) => {
    expect(operation).toThrow(new RangeError(message))
  })

  test('formats as a string', () => {
    expect(String(new Value(2))).toBe('Value(data=2, grad=0)')
  })

  test('computes forward values', () => {
    expect(new Value(2).sub(5).data).toBe(-3)
    expect(new Value(0).pow(0).data).toBe(1)
    expect(new Value(-1000).sigmoid().data).toBe(0)
    expect(new Value(-2).relu().data).toBe(0)
  })
})

describe('backward', () => {
  test.each(UNARY)('%s matches finite differences', (_, fn, at) => {
    const x = new Value(at)
    fn(x).backward()
    expect(x.grad).toBeCloseTo(
      finiteDifference((value) => fn(new Value(value)).data, at),
      6
    )
  })

  test('matches the reference micrograd example', () => {
    const a = new Value(-4)
    const b = new Value(2)
    let c = a.add(b)
    let d = a.mul(b).add(b.pow(3))
    c = c.add(c.add(1))
    c = c.add(new Value(1).add(c).sub(a))
    d = d.add(d.mul(2).add(b.add(a).relu()))
    d = d.add(new Value(3).mul(d).add(b.sub(a).relu()))
    const e = c.sub(d)
    const f = e.pow(2)
    let g = f.div(2)
    g = g.add(new Value(10).div(f))

    g.backward()

    expect(g.data).toBeCloseTo(24.7041, 4)
    expect(a.grad).toBeCloseTo(138.8338, 4)
    expect(b.grad).toBeCloseTo(645.5773, 4)
  })

  test('propagates through a zero base', () => {
    const x = new Value(0)
    x.pow(1).backward()
    expect(x.grad).toBe(1)

    const y = new Value(0)
    y.pow(2).backward()
    expect(y.grad).toBe(0)

    const z = new Value(0)
    z.pow(0).backward()
    expect(z.grad).toBe(0)
  })

  test('accumulates exactly on repeated calls', () => {
    const x = new Value(3)
    const y = x.mul(x).add(x)
    y.backward()
    y.backward()
    expect(x.grad).toBe(14)
  })

  test('accumulates across graphs sharing a leaf', () => {
    const w = new Value(2)
    w.mul(3).backward()
    w.mul(5).backward()
    expect(w.grad).toBe(8)
  })

  test('handles graphs deeper than the call stack', () => {
    const x = new Value(1)
    let sum = x
    for (let index = 0; index < 100_000; index++) {
      sum = sum.add(x)
    }
    sum.backward()
    expect(x.grad).toBe(100_001)
  })

  test('throws when a gradient overflows', () => {
    const y = new Value(1e-100).mul(1e300).mul(1e100)
    expect(() => y.backward()).toThrow(new RangeError('backward produced Infinity'))
  })

  test('zeroGrad resets the whole graph', () => {
    const a = new Value(2)
    const b = new Value(3)
    const c = a.mul(b).add(a)
    c.backward()
    c.zeroGrad()
    expect([a.grad, b.grad, c.grad]).toEqual([0, 0, 0])
  })
})

describe('gradients', () => {
  test('returns gradients without touching grad', () => {
    const x = new Value(3)
    const y = new Value(4)
    const [dx, dy] = x.mul(y).add(y).gradients([x, y])
    expect(dx.data).toBe(4)
    expect(dy.data).toBe(4)
    expect(x.grad).toBe(0)
  })

  test('returns zero for unrelated inputs and one for the output itself', () => {
    const x = new Value(3)
    const y = x.mul(2)
    const [unrelated, self] = y.gradients([new Value(1), y])
    expect(unrelated.data).toBe(0)
    expect(self.data).toBe(1)
  })

  test.each(UNARY)('%s second derivative matches finite differences', (_, fn, at) => {
    const x = new Value(at)
    const [dx] = fn(x).gradients([x])
    const [d2x] = dx.gradients([x])
    const firstDerivative = (value: number): number => {
      const input = new Value(value)
      return fn(input).gradients([input])[0].data
    }
    expect(d2x.data).toBeCloseTo(finiteDifference(firstDerivative, at), 4)
  })

  test('differentiates a polynomial until it vanishes', () => {
    const x = new Value(3)
    const [d1] = x.pow(3).gradients([x])
    const [d2] = d1.gradients([x])
    const [d3] = d2.gradients([x])
    const [d4] = d3.gradients([x])
    expect([d1.data, d2.data, d3.data, d4.data]).toEqual([27, 18, 6, 0])
  })

  test('computes mixed partials', () => {
    const x = new Value(2)
    const y = new Value(3)
    const f = x.pow(2).mul(y).add(y.pow(3))
    const [dfdx, dfdy] = f.gradients([x, y])
    const [d2fdxdy] = dfdx.gradients([y])
    const [d2fdydx] = dfdy.gradients([x])
    expect(dfdx.data).toBe(12)
    expect(dfdy.data).toBe(31)
    expect(d2fdxdy.data).toBe(4)
    expect(d2fdydx.data).toBe(4)
  })

  test('produces a graph that backward can run on', () => {
    const x = new Value(0.5)
    const [dx] = x.tanh().gradients([x])
    dx.backward()
    const t = Math.tanh(0.5)
    expect(x.grad).toBeCloseTo(-2 * t * (1 - t ** 2), 12)
  })
})
