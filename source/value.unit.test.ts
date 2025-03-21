import { beforeEach, describe, expect, test } from 'vitest'
import { Value } from './value.ts'

describe('Value', () => {
  describe('constructor and basic properties', () => {
    test('should initialize with correct default values', () => {
      const v = new Value(5)
      expect(v.data).toBe(5)
      expect(v.grad).toBe(0)
      expect(v.label).toBe('')
      expect(v.children).toEqual([])
      expect(v.operation).toBe('+')
    })

    test('should initialize with custom label and children', () => {
      const child = new Value(3)
      const v = new Value(5, 'test', [child], '*')
      expect(v.data).toBe(5)
      expect(v.label).toBe('test')
      expect(v.children).toEqual([child])
      expect(v.operation).toBe('*')
    })
  })

  describe('Value.from static method', () => {
    describe('number handling', () => {
      test('should handle integer inputs', () => {
        const v = Value.from(5)
        expect(v).toBeInstanceOf(Value)
        expect(v.data).toBe(5)
      })

      test('should handle floating point inputs', () => {
        const v = Value.from(Math.PI)
        expect(v.data).toBeCloseTo(Math.PI)
      })

      test('should handle negative numbers', () => {
        const v = Value.from(-42)
        expect(v.data).toBe(-42)
      })

      test('should handle zero', () => {
        const v = Value.from(0)
        expect(v.data).toBe(0)
      })

      test('should throw for NaN', () => {
        expect(() => Value.from(Number.NaN)).toThrow('Value must be a finite number')
      })

      test('should throw for Infinity', () => {
        expect(() => Value.from(Number.POSITIVE_INFINITY)).toThrow('Value must be a finite number')
        expect(() => Value.from(Number.NEGATIVE_INFINITY)).toThrow('Value must be a finite number')
      })
    })

    describe('Value instance handling', () => {
      test('should return same instance for Value inputs', () => {
        const original = new Value(5)
        const result = Value.from(original)
        expect(result).toBe(original)
      })

      test('should handle nested Value instances', () => {
        const nested = Value.from(Value.from(5))
        expect(nested.data).toBe(5)
      })
    })

    describe('string handling', () => {
      test('should convert valid integer strings', () => {
        expect(Value.from('5').data).toBe(5)
        expect(Value.from('-5').data).toBe(-5)
        expect(Value.from('+5').data).toBe(5)
      })

      test('should convert valid float strings', () => {
        expect(Value.from('3.14').data).toBeCloseTo(3.14)
        expect(Value.from('-3.14').data).toBeCloseTo(-3.14)
        expect(Value.from('0.14').data).toBeCloseTo(0.14)
      })

      test('should convert scientific notation strings', () => {
        expect(Value.from('1e-10').data).toBe(1e-10)
        expect(Value.from('1.23e+4').data).toBe(12300)
      })

      test('should handle whitespace in strings', () => {
        expect(Value.from('  5  ').data).toBe(5)
        expect(Value.from('\n3.14\t').data).toBeCloseTo(3.14)
      })

      test('should throw for invalid number strings', () => {
        expect(() => Value.from('abc')).toThrow('Invalid number format')
        expect(() => Value.from('5.5.5')).toThrow('Invalid number format')
        expect(() => Value.from('3+')).toThrow('Invalid number format')
        expect(() => Value.from('--5')).toThrow('Invalid number format')
      })

      test('should throw for empty strings', () => {
        expect(() => Value.from('')).toThrow('Invalid number format')
        expect(() => Value.from('   ')).toThrow('Invalid number format')
      })
    })

    describe('boolean handling', () => {
      test('should convert true to 1', () => {
        expect(Value.from(true).data).toBe(1)
      })

      test('should convert false to 0', () => {
        expect(Value.from(false).data).toBe(0)
      })
    })

    describe('null and undefined handling', () => {
      test('should throw for null', () => {
        expect(() => Value.from(null)).toThrow('Cannot create Value from null or undefined')
      })

      test('should throw for undefined', () => {
        expect(() => Value.from(undefined)).toThrow('Cannot create Value from null or undefined')
      })
    })

    describe('array handling', () => {
      test('should handle single-element numeric arrays', () => {
        expect(Value.from([5]).data).toBe(5)
        expect(Value.from([-3.14]).data).toBeCloseTo(-3.14)
      })

      test('should throw for empty arrays', () => {
        expect(() => Value.from([])).toThrow('Arrays must contain exactly one numeric value')
      })

      test('should throw for multi-element arrays', () => {
        expect(() => Value.from([1, 2])).toThrow('Arrays must contain exactly one numeric value')
      })

      test('should throw for arrays with non-numeric elements', () => {
        expect(() => Value.from(['abc'])).toThrow('Invalid number format')
      })
    })
  })

  describe('arithmetic operations', () => {
    let a: Value
    let b: Value

    // eslint-disable-next-line vitest/no-hooks
    beforeEach(() => {
      a = new Value(2)
      b = new Value(3)
    })

    describe('addition', () => {
      test('basic addition', () => {
        const c = Value.add(a, b)
        expect(c.data).toBe(5)
        expect(c.children).toContain(a)
        expect(c.children).toContain(b)
      })

      test('method chaining', () => {
        const c = a.add(b)
        expect(c.data).toBe(5)
      })

      test('gradient computation', () => {
        const c = a.add(b)
        c.backward()
        expect(a.grad).toBe(1)
        expect(b.grad).toBe(1)
      })
    })

    describe('subtraction', () => {
      test('basic subtraction', () => {
        const c = Value.sub(a, b)
        expect(c.data).toBe(-1)
      })

      test('gradient computation', () => {
        const c = a.sub(b)
        c.backward()
        expect(a.grad).toBe(1)
        expect(b.grad).toBe(-1)
      })
    })

    describe('multiplication', () => {
      test('basic multiplication', () => {
        const c = Value.mul(a, b)
        expect(c.data).toBe(6)
      })

      test('gradient computation', () => {
        const c = a.mul(b)
        c.backward()
        expect(a.grad).toBe(3) // b.data
        expect(b.grad).toBe(2) // a.data
      })
    })

    describe('division', () => {
      test('basic division', () => {
        const c = Value.div(a, b)
        expect(c.data).toBe(2 / 3)
      })

      test('should throw on division by near-zero value', () => {
        const zero = new Value(0)
        expect(() => a.div(zero)).toThrow('Division by near-zero value')
      })

      test('gradient computation', () => {
        const c = a.div(b)
        c.backward()
        expect(a.grad).toBeCloseTo(1 / 3) // 1/b
        expect(b.grad).toBeCloseTo(-2 / 9) // -a/(b^2)
      })
    })
  })

  describe('exponential operations', () => {
    describe('power edge cases', () => {
      test('should throw for negative base with non-integer exponent', () => {
        const base = new Value(-2)
        const exp = new Value(2.5)
        expect(() => base.pow(exp)).toThrow('Negative numbers cannot be raised to non-integer powers')
      })

      test('should throw for zero base with zero exponent', () => {
        const base = new Value(0)
        const exp = new Value(0)
        const result = base.pow(exp)
        expect(() => result.backward()).toThrow('Cannot raise 0 to zero or negative power')
      })

      test('should throw for zero base with negative exponent', () => {
        const base = new Value(0)
        const exp = new Value(-1)
        const result = base.pow(exp)
        expect(() => result.backward()).toThrow('Cannot raise 0 to zero or negative power')
      })

      test('should handle zero base with positive exponent', () => {
        const base = new Value(0)
        const exp = new Value(2)
        const result = base.pow(exp)
        expect(result.data).toBe(0)
        result.backward()
        expect(base.grad).toBe(0)
        expect(exp.grad).toBe(0)
      })

      test('should handle negative base with even integer exponent', () => {
        const base = new Value(-2)
        const exp = new Value(2)
        const result = base.pow(exp)
        expect(result.data).toBe(4)
      })

      test('should handle negative base with odd integer exponent', () => {
        const base = new Value(-2)
        const exp = new Value(3)
        const result = base.pow(exp)
        expect(result.data).toBe(-8)
      })
    })

    test('exp operation', () => {
      const a = new Value(1)
      const b = Value.exp(a)
      expect(b.data).toBeCloseTo(Math.E)
    })

    test('exp gradient', () => {
      const a = new Value(1)
      const b = a.exp()
      b.backward()
      expect(a.grad).toBeCloseTo(Math.E) // d(e^x)/dx = e^x
    })

    test('power operation', () => {
      const a = new Value(2)
      const b = new Value(3)
      const c = Value.pow(a, b)
      expect(c.data).toBe(8)
    })

    test('power gradient', () => {
      const base = new Value(2)
      const exp = new Value(3)
      const result = base.pow(exp)
      result.backward()
      expect(base.grad).toBeCloseTo(12) // 3 * 2^2
      expect(exp.grad).toBeCloseTo(8 * Math.log(2)) // 2^3 * ln(2)
    })

    test('should handle negative base with integer exponent', () => {
      const base = new Value(-2)
      const exp = new Value(2)
      const result = base.pow(exp)
      expect(result.data).toBe(4)
    })

    test('should throw for negative base with non-integer exponent', () => {
      const base = new Value(-2)
      const exp = new Value(2.5)
      expect(() => base.pow(exp)).toThrow()
    })
  })

  describe('hyperbolic functions', () => {
    test('tanh operation', () => {
      const a = new Value(0)
      const b = Value.tanh(a)
      expect(b.data).toBe(0)
    })

    test('tanh gradient', () => {
      const a = new Value(0)
      const b = a.tanh()
      b.backward()
      expect(a.grad).toBe(1) // d(tanh(x))/dx = 1 - tanh^2(x), at x=0 this is 1
    })

    test('tanh bounds', () => {
      const large = new Value(100)
      const largeTanh = large.tanh()
      expect(largeTanh.data).toBeCloseTo(1)

      const smallNeg = new Value(-100)
      const smallNegTanh = smallNeg.tanh()
      expect(smallNegTanh.data).toBeCloseTo(-1)
    })
  })

  describe('backward propagation', () => {
    test('simple computation graph', () => {
      // Create computation graph: c = a * b + b
      const a = new Value(3)
      const b = new Value(2)
      const prod = a.mul(b)
      const c = prod.add(b)

      c.backward()

      expect(a.grad).toBe(2) // partial derivative with respect to a
      expect(b.grad).toBe(4) // partial derivative with respect to b (2 paths)
    })

    test.skip('complex computation graph', () => {
      // Create more complex graph: d = (a * b + b) * tanh(c)
      const a = new Value(2)
      const b = new Value(3)
      const c = new Value(1)

      const prod = a.mul(b)
      const sum = prod.add(b)
      const t = c.tanh()
      const d = sum.mul(t)

      d.backward()

      // Verify gradients are computed correctly
      expect(a.grad).toBeCloseTo(3 * Math.tanh(1))
      expect(b.grad).toBeCloseTo(4 * Math.tanh(1))
      expect(c.grad).toBeCloseTo(9 * (1 - Math.tanh(1) ** 2))
    })
  })

  describe('edge cases and error handling', () => {
    test('should handle zero values correctly', () => {
      const zero = new Value(0)
      const one = new Value(1)

      expect(zero.add(one).data).toBe(1)
      expect(zero.mul(one).data).toBe(0)
      expect(one.div(one).data).toBe(1)
    })

    test('should handle very large numbers', () => {
      const large = new Value(1e308)
      const small = new Value(1e-308)

      expect(large.mul(small).data).toBeCloseTo(1)
    })

    test('should handle very small numbers', () => {
      const tiny = new Value(1e-308)
      const result = tiny.add(tiny)
      expect(result.data).toBeCloseTo(2e-308)
    })
  })

  describe('toPrimitive conversion', () => {
    test('number hint conversion', () => {
      const v = new Value(5)
      expect(+v).toBe(5)
      expect(Number(v)).toBe(5)
    })

    test('string hint conversion', () => {
      const v = new Value(5)
      expect(String(v)).toBe('Value(5)')
    })
  })
})
