export const API_BASE =
  import.meta.env.VITE_API_BASE ?? 'http://localhost:8000'

export type BlurType = 'gaussian' | 'pixelate' | 'blackout'
export type ActionType = 'predict' | BlurType

export type HealthStatus = {
  status: string
  model_loaded: boolean
}

export async function fetchHealth(signal?: AbortSignal): Promise<HealthStatus> {
  const res = await fetch(`${API_BASE}/health`, { signal })
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`)
  return res.json()
}

export async function runAction(file: File, action: ActionType): Promise<Blob> {
  const form = new FormData()
  form.append('file', file)

  const url =
    action === 'predict'
      ? `${API_BASE}/predict`
      : `${API_BASE}/blur?blur_type=${action}`

  const res = await fetch(url, { method: 'POST', body: form })

  if (!res.ok) {
    let msg = `Errore dal server (${res.status})`
    try {
      const j = await res.json()
      if (j?.detail) msg = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail)
    } catch {
      // body non JSON, usa il messaggio default
    }
    throw new Error(msg)
  }

  return res.blob()
}
