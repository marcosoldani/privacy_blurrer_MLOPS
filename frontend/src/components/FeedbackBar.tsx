import { useState } from 'react'
import { ThumbsUp, ThumbsDown, Check } from 'lucide-react'
import type { ActionType, Rating } from '../api'

type Props = {
  filename: string
  action: ActionType
  onSubmit: (rating: Rating) => Promise<void>
}

export function FeedbackBar({ onSubmit }: Props) {
  const [submitted, setSubmitted] = useState<Rating | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handle = async (rating: Rating) => {
    setError(null)
    try {
      await onSubmit(rating)
      setSubmitted(rating)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Errore feedback')
    }
  }

  if (submitted) {
    return (
      <div className="rounded-lg border border-emerald-500/30 bg-emerald-500/10 px-4 py-2 text-sm text-emerald-200 flex items-center gap-2">
        <Check className="w-4 h-4" />
        Grazie per il feedback ({submitted === 'good' ? '👍' : '👎'}).
      </div>
    )
  }

  return (
    <div className="rounded-lg border border-slate-700 bg-slate-900/60 px-4 py-3 flex items-center justify-between gap-4">
      <span className="text-sm text-slate-200">Sei soddisfatto del risultato?</span>
      <div className="flex items-center gap-2">
        <button
          onClick={() => handle('good')}
          className="flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium bg-emerald-600/80 hover:bg-emerald-500 text-white transition"
        >
          <ThumbsUp className="w-3.5 h-3.5" />
          Sì
        </button>
        <button
          onClick={() => handle('bad')}
          className="flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium bg-rose-600/80 hover:bg-rose-500 text-white transition"
        >
          <ThumbsDown className="w-3.5 h-3.5" />
          No
        </button>
      </div>
      {error && <span className="text-xs text-rose-300 ml-2">{error}</span>}
    </div>
  )
}
