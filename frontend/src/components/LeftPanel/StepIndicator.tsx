import React from 'react'

interface Props {
  step: number
  total: number
  title: string
  subtitle: string
}

export default function StepIndicator({ step, total, title, subtitle }: Props) {
  return (
    <div className="px-4 pt-4 pb-3 border-b border-[#2a2a2a]">
      <div className="flex items-start gap-3">
        <span className="flex-shrink-0 mt-0.5 bg-[#2a2a2a] text-[#ccc] text-xs font-semibold px-2 py-0.5 rounded-full">
          {step}/{total}
        </span>
        <div>
          <h3 className="text-sm font-semibold text-white leading-snug">{title}</h3>
          <p className="text-xs text-[#777] mt-1 leading-relaxed">{subtitle}</p>
        </div>
      </div>
    </div>
  )
}
