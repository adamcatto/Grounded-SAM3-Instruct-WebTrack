/** SSE events are delimited by a blank line. Spec uses CRLF (`\r\n\r\n`); some servers use LF. */
export const SSE_EVENT_SPLIT = /\r?\n\r?\n/

export function parseSseChunk(chunk: string): { event: string; data: string } | null {
  let event = 'message'
  const dataLines: string[] = []
  const normalized = chunk.replace(/\r\n/g, '\n').replace(/\r/g, '\n')
  for (const rawLine of normalized.split('\n')) {
    const line = rawLine.trimEnd()
    if (line.startsWith('event:')) event = line.slice(6).trim()
    else if (line.startsWith('data:')) dataLines.push(line.slice(5).trimStart())
  }
  if (dataLines.length === 0) return null
  return { event, data: dataLines.join('\n').trim() }
}
