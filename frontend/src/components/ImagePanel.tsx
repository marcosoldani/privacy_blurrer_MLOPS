import { Download, X, Loader2 } from 'lucide-react'

type Props = {
  title: string
  imageUrl: string | null
  filename?: string
  loading?: boolean
  placeholder?: string
  onReset?: () => void
  onDownload?: () => void
}

export function ImagePanel({
  title,
  imageUrl,
  filename,
  loading,
  placeholder,
  onReset,
  onDownload,
}: Props) {
  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-900/40 overflow-hidden flex flex-col">
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-800">
        <div className="min-w-0">
          <h3 className="text-sm font-medium text-slate-100">{title}</h3>
          {filename && (
            <p className="text-xs text-slate-500 truncate">{filename}</p>
          )}
        </div>
        <div className="flex items-center gap-2 shrink-0">
          {onDownload && (
            <button
              onClick={onDownload}
              className="flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-xs font-medium bg-indigo-600 hover:bg-indigo-500 text-white transition"
            >
              <Download className="w-3.5 h-3.5" />
              Scarica
            </button>
          )}
          {onReset && (
            <button
              onClick={onReset}
              className="flex items-center gap-1.5 rounded-md px-2 py-1.5 text-xs font-medium bg-slate-800 hover:bg-slate-700 border border-slate-700 transition"
              title="Rimuovi immagine"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>

      <div className="relative bg-checker bg-slate-950/60 flex items-center justify-center h-[340px] md:h-[400px] lg:h-[440px]">
        {imageUrl && (
          <img
            src={imageUrl}
            alt={title}
            className="max-w-full max-h-full object-contain"
          />
        )}

        {!imageUrl && !loading && (
          <p className="text-sm text-slate-500 px-6 text-center">
            {placeholder ?? 'Nessuna immagine'}
          </p>
        )}

        {loading && (
          <div className="absolute inset-0 bg-slate-950/70 backdrop-blur-sm flex flex-col items-center justify-center gap-3">
            <Loader2 className="w-7 h-7 animate-spin text-indigo-400" />
            <span className="text-xs text-slate-300">Elaborazione in corso…</span>
          </div>
        )}
      </div>
    </div>
  )
}
