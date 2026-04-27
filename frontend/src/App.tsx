import { useEffect, useState } from 'react'
import { Shield, AlertCircle } from 'lucide-react'
import { Dropzone } from './components/Dropzone'
import { ActionBar } from './components/ActionBar'
import { ImagePanel } from './components/ImagePanel'
import { HealthBadge } from './components/HealthBadge'
import { SatisfactionBadge } from './components/SatisfactionBadge'
import { FeedbackBar } from './components/FeedbackBar'
import {
  fetchHealth,
  fetchFeedbackStats,
  runAction,
  submitFeedback,
  API_BASE,
} from './api'
import type { ActionType, FeedbackStats, HealthStatus, Rating } from './api'

const ACTION_LABELS: Record<ActionType, string> = {
  predict: 'Maschera binaria',
  gaussian: 'Gaussian blur',
  pixelate: 'Pixelate',
  blackout: 'Blackout',
}

export function App() {
  const [file, setFile] = useState<File | null>(null)
  const [originalUrl, setOriginalUrl] = useState<string | null>(null)
  const [resultUrl, setResultUrl] = useState<string | null>(null)
  const [resultLabel, setResultLabel] = useState<string | null>(null)
  const [loading, setLoading] = useState<ActionType | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [health, setHealth] = useState<HealthStatus | null>(null)

  // Feedback state
  const [feedbackArmed, setFeedbackArmed] = useState(false) // true = chiediamolo alla prossima azione completata
  const [feedbackContext, setFeedbackContext] = useState<
    { file: string; action: ActionType } | null
  >(null) // se non null, FeedbackBar visibile
  const [feedbackStats, setFeedbackStats] = useState<FeedbackStats | null>(null)

  // Poll health + stats ogni 10s
  useEffect(() => {
    let cancelled = false

    const check = async () => {
      try {
        const h = await fetchHealth()
        if (!cancelled) setHealth(h)
      } catch {
        if (!cancelled) setHealth(null)
      }
      try {
        const s = await fetchFeedbackStats()
        if (!cancelled) setFeedbackStats(s)
      } catch {
        if (!cancelled) setFeedbackStats(null)
      }
    }

    check()
    const id = setInterval(check, 10_000)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [])

  const handleFile = (f: File) => {
    if (originalUrl) URL.revokeObjectURL(originalUrl)
    if (resultUrl) URL.revokeObjectURL(resultUrl)
    setFile(f)
    setOriginalUrl(URL.createObjectURL(f))
    setResultUrl(null)
    setResultLabel(null)
    setError(null)
    // Nuova foto: armiamo il feedback per la prossima azione
    setFeedbackArmed(true)
    setFeedbackContext(null)
  }

  const clearAll = () => {
    if (originalUrl) URL.revokeObjectURL(originalUrl)
    if (resultUrl) URL.revokeObjectURL(resultUrl)
    setFile(null)
    setOriginalUrl(null)
    setResultUrl(null)
    setResultLabel(null)
    setError(null)
    setFeedbackArmed(false)
    setFeedbackContext(null)
  }

  const handleAction = async (action: ActionType) => {
    if (!file) return
    setLoading(action)
    setError(null)
    try {
      const blob = await runAction(file, action)
      if (resultUrl) URL.revokeObjectURL(resultUrl)
      setResultUrl(URL.createObjectURL(blob))
      setResultLabel(ACTION_LABELS[action])
      // Se era la prima azione su questa foto, chiediamo feedback
      if (feedbackArmed) {
        setFeedbackContext({ file: file.name, action })
        setFeedbackArmed(false)
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Errore sconosciuto')
    } finally {
      setLoading(null)
    }
  }

  const handleFeedback = async (rating: Rating) => {
    if (!feedbackContext) return
    await submitFeedback(feedbackContext.file, feedbackContext.action, rating)
    // refresh stats subito (non aspettare il prossimo polling)
    try {
      const s = await fetchFeedbackStats()
      setFeedbackStats(s)
    } catch {
      // ignore stats refresh errors
    }
  }

  const downloadResult = () => {
    if (!resultUrl || !resultLabel) return
    const a = document.createElement('a')
    a.href = resultUrl
    const base = file?.name.replace(/\.[^.]+$/, '') ?? 'output'
    const suffix = resultLabel.toLowerCase().replace(/\s+/g, '_')
    a.download = `${base}_${suffix}.png`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
  }

  const backendDown = health === null
  const modelReady = health?.model_loaded ?? false

  return (
    <div className="min-h-full bg-slate-950 text-slate-100 font-sans">
      <header className="sticky top-0 z-10 border-b border-slate-800/60 bg-slate-950/80 backdrop-blur">
        <div className="mx-auto max-w-[1600px] px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-indigo-600/20 border border-indigo-500/40 flex items-center justify-center">
              <Shield className="w-5 h-5 text-indigo-400" />
            </div>
            <div>
              <h1 className="text-lg font-semibold tracking-tight">Privacy Blurrer</h1>
              <p className="text-xs text-slate-400">
                U-Net person segmentation &amp; anonymization
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <SatisfactionBadge stats={feedbackStats} />
            <HealthBadge health={health} />
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-[1600px] px-6 py-8 space-y-6">
        {backendDown && (
          <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-sm text-amber-200">
            Backend non raggiungibile su <code className="text-amber-100">{API_BASE}</code>.
            Avvia il server con <code className="text-amber-100">uvicorn src.app:app --host 0.0.0.0 --port 8000</code>.
          </div>
        )}

        {error && (
          <div className="flex items-start gap-3 rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm">
            <AlertCircle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
            <div className="flex-1 text-red-200">{error}</div>
            <button
              onClick={() => setError(null)}
              className="text-red-300/70 hover:text-red-200 text-xs"
            >
              chiudi
            </button>
          </div>
        )}

        {!file ? (
          <Dropzone onFile={handleFile} />
        ) : (
          <>
            <div className="grid lg:grid-cols-2 gap-4">
              <ImagePanel
                title="Originale"
                imageUrl={originalUrl}
                filename={file.name}
                onReset={clearAll}
              />
              <ImagePanel
                title={resultLabel ?? 'Risultato'}
                imageUrl={resultUrl}
                loading={loading !== null}
                placeholder="Seleziona un'operazione qui sotto"
                onDownload={resultUrl ? downloadResult : undefined}
              />
            </div>

            <ActionBar
              loading={loading}
              onAction={handleAction}
              modelReady={modelReady}
              disabled={backendDown}
            />

            {feedbackContext && resultUrl && (
              <FeedbackBar
                filename={feedbackContext.file}
                action={feedbackContext.action}
                onSubmit={handleFeedback}
              />
            )}
          </>
        )}
      </main>

    </div>
  )
}
