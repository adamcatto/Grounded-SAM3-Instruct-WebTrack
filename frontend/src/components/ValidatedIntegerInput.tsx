import React, { useEffect, useRef, useState } from 'react'

type Props = {
  value: number
  min: number
  max: number
  onCommit: (value: number) => void
  disabled?: boolean
  keepSteppers?: boolean
  wrapperClassName?: string
  className?: string
  title?: string
}

/** Integer input with an unconstrained draft and validation before numeric state is updated. */
export default function ValidatedIntegerInput({
  value,
  min,
  max,
  onCommit,
  disabled,
  keepSteppers = false,
  wrapperClassName = '',
  className = '',
  title,
}: Props) {
  const [draft, setDraft] = useState(String(value))
  const focused = useRef(false)
  const skipBlurCommit = useRef(false)
  const valid = /^\d+$/.test(draft)

  useEffect(() => {
    if (!focused.current) setDraft(String(value))
  }, [value])

  function commit() {
    if (!valid) return
    const parsed = Number(draft)
    if (!Number.isSafeInteger(parsed)) return
    const next = Math.max(min, Math.min(max, parsed))
    setDraft(String(next))
    onCommit(next)
  }

  return (
    <span className={`relative block ${wrapperClassName}`}>
      <input
        type={keepSteppers ? 'number' : 'text'}
        inputMode="numeric"
        pattern="[0-9]*"
        min={keepSteppers ? min : undefined}
        max={keepSteppers ? max : undefined}
        step={keepSteppers ? 1 : undefined}
        autoComplete="off"
        spellCheck={false}
        value={draft}
        disabled={disabled}
        title={title}
        onFocus={event => {
          focused.current = true
          requestAnimationFrame(() => event.currentTarget.select())
        }}
        onChange={event => setDraft(event.target.value)}
        onKeyDown={event => {
          if (event.key === 'Enter') {
            event.preventDefault()
            skipBlurCommit.current = true
            commit()
            event.currentTarget.blur()
          } else if (event.key === 'Escape') {
            skipBlurCommit.current = true
            setDraft(String(value))
            event.currentTarget.blur()
          }
        }}
        onBlur={() => {
          focused.current = false
          if (skipBlurCommit.current) {
            skipBlurCommit.current = false
          } else {
            commit()
          }
        }}
        className={className}
      />
      {!valid && (
        <span className="absolute left-0 top-full mt-0.5 whitespace-nowrap text-[10px] leading-none text-red-400">
          Only numbers are allowed
        </span>
      )}
    </span>
  )
}
