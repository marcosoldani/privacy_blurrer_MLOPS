import type { HealthStatus } from '../api'

type Props = {
  health: HealthStatus | null
}

export function HealthBadge({ health }: Props) {
  const offline = health === null
  const modelLoaded = health?.model_loaded ?? false

  const { dot, label } = offline
    ? { dot: 'bg-red-500', label: 'Backend offline' }
    : modelLoaded
    ? { dot: 'bg-emerald-500', label: 'Modello pronto' }
    : { dot: 'bg-amber-500', label: 'Modello non caricato' }

  return (
    <div className="flex items-center gap-2 rounded-full border border-slate-800 bg-slate-900/60 px-3 py-1.5 text-xs">
      <span className={`w-2 h-2 rounded-full ${dot} ${!offline ? 'animate-pulse' : ''}`} />
      <span className="text-slate-300">{label}</span>
    </div>
  )
}
