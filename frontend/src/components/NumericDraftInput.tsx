import React, { useEffect, useRef, useState } from 'react'

export type NumericDraftInputProps = {
  value: number
  onChange: (v: number) => void
  /** Called while typing when the draft parses to a number (for live previews). */
  onLiveChange?: (v: number | null) => void
  min: number
  max: number
  disabled?: boolean
  className?: string
  autoFocus?: boolean
  id?: string
}

/** Text input that allows select-all and in-place digit editing; clamps on blur/Enter. */
export default function NumericDraftInput({
  value,
  onChange,
  onLiveChange,
  min,
  max,
  disabled,
  className = '',
  autoFocus,
  id,
}: NumericDraftInputProps) {
  const [draft, setDraft] = useState(() => String(value))
  const focusedRef = useRef(false)

  useEffect(() => {
    if (!focusedRef.current) setDraft(String(value))
  }, [value])

  function clamp(n: number) {
    return Math.max(min, Math.min(max, n))
  }

  function emitLive(raw: string) {
    if (!onLiveChange) return
    const trimmed = raw.trim()
    if (trimmed === '') {
      onLiveChange(null)
      return
    }
    const parsed = parseInt(trimmed, 10)
    onLiveChange(Number.isFinite(parsed) ? clamp(parsed) : null)
  }

  function commitDraft() {
    const trimmed = draft.trim()
    const parsed = trimmed === '' ? null : parseInt(trimmed, 10)
    const committed =
      parsed !== null && Number.isFinite(parsed) ? clamp(parsed) : value
    setDraft(String(committed))
    onChange(committed)
    onLiveChange?.(committed)
  }

  return (
    <input
      id={id}
      type="text"
      inputMode="numeric"
      autoComplete="off"
      spellCheck={false}
      value={draft}
      disabled={disabled}
      autoFocus={autoFocus}
      onFocus={e => {
        focusedRef.current = true
        setDraft(String(value))
        requestAnimationFrame(() => e.currentTarget.select())
      }}
      onChange={e => {
        const next = e.target.value.replace(/\D/g, '')
        setDraft(next)
        emitLive(next)
      }}
      onKeyDown={e => {
        if (disabled) return
        if (e.key === 'Enter') {
          e.preventDefault()
          commitDraft()
          e.currentTarget.blur()
        }
      }}
      onBlur={() => {
        focusedRef.current = false
        if (!disabled) commitDraft()
      }}
      className={className}
    />
  )
}
