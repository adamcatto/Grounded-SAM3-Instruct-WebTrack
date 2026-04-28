export const OBJECT_COLORS = [
  '#5B8DD9',  // blue
  '#E8A445',  // orange
  '#52C41A',  // green
  '#722ED1',  // purple
  '#F5222D',  // red
  '#13C2C2',  // teal
  '#FA8C16',  // dark orange
  '#EB2F96',  // pink
]

export function getObjectColor(index: number): string {
  return OBJECT_COLORS[index % OBJECT_COLORS.length]
}

export function hexToRgba(hex: string, alpha: number): string {
  const r = parseInt(hex.slice(1, 3), 16)
  const g = parseInt(hex.slice(3, 5), 16)
  const b = parseInt(hex.slice(5, 7), 16)
  return `rgba(${r}, ${g}, ${b}, ${alpha})`
}

export function lightenColor(hex: string, amount: number): string {
  const r = Math.min(255, parseInt(hex.slice(1, 3), 16) + amount)
  const g = Math.min(255, parseInt(hex.slice(3, 5), 16) + amount)
  const b = Math.min(255, parseInt(hex.slice(5, 7), 16) + amount)
  return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`
}
