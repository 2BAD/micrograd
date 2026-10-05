import { describe, expect, test } from 'vitest'
import { toMermaid } from './mermaid.ts'
import { Value } from './value.ts'

describe('toMermaid', () => {
  test('renders values, ops and edges', () => {
    const a = new Value(2, 'a')
    const b = new Value(-3, 'b')
    const c = a.mul(b)
    c.label = 'c'
    c.backward()

    expect(toMermaid(c)).toBe(
      [
        'flowchart LR',
        '  v0["c #124; data -6.0000 #124; grad 1.0000"]',
        '  v0_op(("#42;"))',
        '  v0_op --> v0',
        '  v1 --> v0_op',
        '  v2 --> v0_op',
        '  v2["b #124; data -3.0000 #124; grad 2.0000"]',
        '  v1["a #124; data 2.0000 #124; grad -3.0000"]'
      ].join('\n')
    )
  })

  test('emits each node and edge once for shared subgraphs', () => {
    const x = new Value(2)
    const y = x.mul(x)
    const z = y.add(y)
    const lines = toMermaid(z).split('\n')
    expect(lines.filter((line) => line.endsWith('"]'))).toHaveLength(3)
    expect(lines.filter((line) => line.includes('-->'))).toHaveLength(4)
    expect(new Set(lines).size).toBe(lines.length)
  })

  test('escapes characters that break mermaid labels', () => {
    const value = new Value(1, 'say "hi" <b>')
    expect(toMermaid(value)).toContain('say #34;hi#34; #60;b#62;')
  })
})
