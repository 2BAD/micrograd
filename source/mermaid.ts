import type { Value } from './value.ts'

const escape = (text: string): string => text.replaceAll(/[^\w .=-]/g, (char) => `#${char.codePointAt(0)};`)

export const toMermaid = (root: Value): string => {
  const ids = new Map<Value, string>()
  const pending: Value[] = []
  const idOf = (node: Value): string => {
    const known = ids.get(node)
    if (known !== undefined) {
      return known
    }
    const id = `v${ids.size}`
    ids.set(node, id)
    pending.push(node)
    return id
  }

  const lines = ['flowchart LR']
  idOf(root)
  for (let node = pending.pop(); node; node = pending.pop()) {
    const id = idOf(node)
    const label = [node.label, `data ${node.data.toFixed(4)}`, `grad ${node.grad.toFixed(4)}`].filter(Boolean)
    lines.push(`  ${id}["${escape(label.join(' | '))}"]`)
    if (node.op) {
      lines.push(`  ${id}_op(("${escape(node.op)}"))`, `  ${id}_op --> ${id}`)
      for (const child of new Set(node.children)) {
        lines.push(`  ${idOf(child)} --> ${id}_op`)
      }
    }
  }
  return lines.join('\n')
}
