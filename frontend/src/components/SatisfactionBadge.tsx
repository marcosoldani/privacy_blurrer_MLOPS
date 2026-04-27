import type { FeedbackStats } from '../api'

type Props = {
  stats: FeedbackStats | null
}

export function SatisfactionBadge({ stats }: Props) {
  if (!stats || stats.total === 0 || stats.percentage === null) {
    return (
      <div className="flex items-center gap-2 rounded-full border border-slate-800 bg-slate-900/60 px-3 py-1.5 text-xs text-slate-400">
        <span>Soddisfazione: —</span>
      </div>
    )
  }

  const pct = stats.percentage
  const color =
    pct >= 80 ? 'text-emerald-300' : pct >= 50 ? 'text-amber-300' : 'text-rose-300'

  return (
    <div className="flex items-center gap-2 rounded-full border border-slate-800 bg-slate-900/60 px-3 py-1.5 text-xs">
      <span className="text-slate-400">Soddisfazione</span>
      <span className={`font-semibold ${color}`}>{pct}%</span>
      <span className="text-slate-500">({stats.positive}/{stats.total})</span>
    </div>
  )
}
