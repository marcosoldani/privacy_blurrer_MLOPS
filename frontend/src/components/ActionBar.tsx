import { Wand2, Sparkles, Grid3x3, Square, Loader2 } from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import type { ActionType } from '../api'

type Props = {
  loading: ActionType | null
  onAction: (action: ActionType) => void
  modelReady: boolean
  disabled?: boolean
}

const ACTIONS: Array<{ id: ActionType; label: string; icon: LucideIcon; hint: string }> = [
  { id: 'predict', label: 'Maschera', icon: Wand2, hint: 'Maschera binaria (persona vs sfondo)' },
  { id: 'gaussian', label: 'Gaussian', icon: Sparkles, hint: 'Blur gaussiano sulle persone' },
  { id: 'pixelate', label: 'Pixelate', icon: Grid3x3, hint: 'Effetto mosaico sulle persone' },
  { id: 'blackout', label: 'Blackout', icon: Square, hint: 'Oscuramento completo delle persone' },
]

export function ActionBar({ loading, onAction, modelReady, disabled }: Props) {
  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-900/40 p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-slate-100">Operazioni</h3>
        {!modelReady && (
          <span className="text-xs text-amber-400">Modello non pronto</span>
        )}
      </div>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
        {ACTIONS.map(({ id, label, icon: Icon, hint }) => {
          const isLoading = loading === id
          const isDisabled = disabled || loading !== null || !modelReady
          return (
            <button
              key={id}
              disabled={isDisabled}
              onClick={() => onAction(id)}
              title={hint}
              className={`
                group relative flex flex-col items-center gap-2 rounded-xl px-3 py-4
                border transition
                ${
                  isDisabled
                    ? 'bg-slate-900/40 border-slate-800 text-slate-500 cursor-not-allowed'
                    : 'bg-slate-900 border-slate-700 hover:border-indigo-500/60 hover:bg-slate-800/80 text-slate-100'
                }
              `}
            >
              {isLoading ? (
                <Loader2 className="w-5 h-5 animate-spin text-indigo-400" />
              ) : (
                <Icon
                  className={`w-5 h-5 ${
                    isDisabled ? 'text-slate-600' : 'text-indigo-400 group-hover:text-indigo-300'
                  }`}
                />
              )}
              <span className="text-sm font-medium">{label}</span>
            </button>
          )
        })}
      </div>
    </div>
  )
}
