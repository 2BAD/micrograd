import type { Value } from '../value.ts'

export class GraphVisualizer {
  private nodeIds: Map<Value, string>
  private visited: Set<Value>
  private mermaidCode: string[]

  constructor() {
    this.nodeIds = new Map()
    this.visited = new Set()
    this.mermaidCode = []
  }

  private getNodeId(value: Value): string {
    let id = this.nodeIds.get(value)
    if (!id) {
      id = `node${this.nodeIds.size}`
      this.nodeIds.set(value, id)
    }
    return id
  }

  private getOpNodeId(valueId: string): string {
    return `${valueId}_op`
  }

  private trace(value: Value): void {
    const valueId = this.getNodeId(value)

    if (!this.visited.has(value)) {
      this.visited.add(value)

      const opLabel = value.operation

      this.mermaidCode.push(
        `    ${valueId}["${value.label}
        data: ${value.data.toFixed(4)}
        grad: ${value.grad.toFixed(4)}"]:::valueNode;`
      )

      if (value.children.length > 0) {
        const opNodeId = this.getOpNodeId(valueId)
        this.mermaidCode.push(`    ${opNodeId}["${/[*+]/.test(opLabel) ? `\\${opLabel}` : opLabel}"];`)
        this.mermaidCode.push(`    ${opNodeId} --> ${valueId};`)
      }

      for (const child of value.prev()) {
        this.trace(child)
      }
    }

    if (value.children.length > 0) {
      for (const child of value.prev()) {
        const childId = this.getNodeId(child)
        const opNodeId = this.getOpNodeId(valueId)
        this.mermaidCode.push(`    ${childId} --> ${opNodeId};`)
      }
    }
  }

  generateMermaid(root: Value): string {
    this.nodeIds.clear()
    this.visited.clear()
    this.mermaidCode = ['graph LR;']

    this.trace(root)

    this.mermaidCode.push('    classDef valueNode rx,ry:10,10;')

    return this.mermaidCode.join('\n')
  }
}
