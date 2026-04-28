import React, { useEffect } from 'react'
import { X, CheckCircle2, AlertCircle, Info } from 'lucide-react'
import { useStore } from '../store/useStore'

const TOAST_DURATION = 5000 // ms before auto-dismiss

export default function ToastContainer() {
  const { toasts, removeToast } = useStore()

  return (
    <div className="fixed bottom-4 right-4 z-50 flex flex-col-reverse gap-2 pointer-events-none">
      {toasts.map(toast => (
        <ToastItem key={toast.id} id={toast.id} message={toast.message} type={toast.type} onRemove={removeToast} />
      ))}
    </div>
  )
}

function ToastItem({ id, message, type, onRemove }: {
  id: string
  message: string
  type: 'info' | 'success' | 'error'
  onRemove: (id: string) => void
}) {
  useEffect(() => {
    const timer = setTimeout(() => onRemove(id), TOAST_DURATION)
    return () => clearTimeout(timer)
  }, [id, onRemove])

  const styles = {
    info:    { bg: 'bg-[#1e1e1e] border-[#333]',        icon: <Info size={13} className="text-blue-400 flex-shrink-0" /> },
    success: { bg: 'bg-[#1a2a1a] border-green-800/50',   icon: <CheckCircle2 size={13} className="text-green-400 flex-shrink-0" /> },
    error:   { bg: 'bg-[#2a1a1a] border-red-800/50',     icon: <AlertCircle size={13} className="text-red-400 flex-shrink-0" /> },
  }

  const { bg, icon } = styles[type]

  return (
    <div
      className={`pointer-events-auto flex items-start gap-2.5 px-3 py-2.5 rounded-lg border shadow-xl text-xs text-[#ccc] max-w-xs ${bg}`}
      style={{ animation: 'fadeSlideIn 0.15s ease-out' }}
    >
      {icon}
      <span className="flex-1 leading-relaxed">{message}</span>
      <button
        onClick={() => onRemove(id)}
        className="flex-shrink-0 text-[#555] hover:text-[#aaa] transition-colors mt-0.5"
      >
        <X size={11} />
      </button>
    </div>
  )
}
