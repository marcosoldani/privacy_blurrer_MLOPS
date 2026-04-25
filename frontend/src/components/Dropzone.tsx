import { useState } from 'react'
import { Upload, Image as ImageIcon } from 'lucide-react'

type Props = {
  onFile: (file: File) => void
}

const ACCEPTED = ['image/jpeg', 'image/png', 'image/bmp', 'image/webp']

export function Dropzone({ onFile }: Props) {
  const [isOver, setIsOver] = useState(false)
  const [localError, setLocalError] = useState<string | null>(null)

  const handleFiles = (files: FileList | null) => {
    const f = files?.[0]
    if (!f) return
    if (!ACCEPTED.includes(f.type)) {
      setLocalError(`Formato non supportato: ${f.type || 'sconosciuto'}. Usa JPG, PNG, BMP o WEBP.`)
      return
    }
    setLocalError(null)
    onFile(f)
  }

  return (
    <div className="space-y-2">
      <label
        onDragOver={(e) => {
          e.preventDefault()
          setIsOver(true)
        }}
        onDragEnter={(e) => e.preventDefault()}
        onDragLeave={(e) => {
          if (e.currentTarget === e.target) setIsOver(false)
        }}
        onDrop={(e) => {
          e.preventDefault()
          setIsOver(false)
          handleFiles(e.dataTransfer.files)
        }}
        className={`
          group block cursor-pointer rounded-2xl border-2 border-dashed transition
          ${
            isOver
              ? 'border-indigo-400 bg-indigo-500/10'
              : 'border-slate-700 hover:border-slate-500 bg-slate-900/40'
          }
          px-6 py-20
        `}
      >
        <input
          type="file"
          accept="image/jpeg,image/png,image/bmp,image/webp"
          className="hidden"
          onChange={(e) => handleFiles(e.target.files)}
        />
        <div className="flex flex-col items-center gap-4 text-center">
          <div className="w-14 h-14 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center group-hover:border-indigo-500/60 transition">
            <Upload className="w-6 h-6 text-slate-400 group-hover:text-indigo-400 transition" />
          </div>
          <div>
            <p className="text-base font-medium text-slate-100">Trascina qui un&apos;immagine</p>
            <p className="text-sm text-slate-400 mt-1">
              oppure{' '}
              <span className="text-indigo-400 group-hover:underline underline-offset-4">
                scegli un file
              </span>
            </p>
          </div>
          <div className="flex items-center gap-2 text-xs text-slate-500">
            <ImageIcon className="w-3.5 h-3.5" />
            JPG · PNG · BMP · WEBP &nbsp;—&nbsp; max 10MB, 32–4096px
          </div>
        </div>
      </label>

      {localError && (
        <p className="text-xs text-red-400 px-1">{localError}</p>
      )}
    </div>
  )
}
