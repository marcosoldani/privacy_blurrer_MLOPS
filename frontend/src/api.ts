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

export type Rating = 'good' | 'bad'

export type FeedbackStats = {
  total: number
  positive: number
  percentage: number | null
}

export async function submitFeedback(
  filename: string,
  action: ActionType,
  rating: Rating,
): Promise<void> {
  const res = await fetch(`${API_BASE}/feedback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ filename, action, rating }),
  })
  if (!res.ok) throw new Error(`Feedback submit failed: ${res.status}`)
}

export async function fetchFeedbackStats(signal?: AbortSignal): Promise<FeedbackStats> {
  const res = await fetch(`${API_BASE}/feedback/stats`, { signal })
  if (!res.ok) throw new Error(`Feedback stats failed: ${res.status}`)
  return res.json()
}
