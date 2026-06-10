import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  headers: { 'Content-Type': 'application/json' },
})

export const checkHealth = () => api.get('/health')

// ── Training ──────────────────────────────────────────────────────────────────
export const listTrainingConfigs = () => api.get('/training/configs')
export const getTrainingConfig = (name) => api.get(`/training/config/${encodeURIComponent(name)}`)
export const listTrainingRuns = () => api.get('/training/runs')
export const startTraining = (data) => api.post('/training/start', data)
export const stopTraining = (jobId) => api.post(`/training/stop/${jobId}`)
export const getTrainingStatus = (jobId) => api.get(`/training/status/${jobId}`)

// ── Inference ─────────────────────────────────────────────────────────────────
export const listInferenceTasks = () => api.get('/inference/tasks')
export const getLatestModel = (taskName) => api.get(`/inference/latest-model/${encodeURIComponent(taskName)}`)
export const startInference = (data) => api.post('/inference/start', data)
export const stopInference = (jobId) => api.post(`/inference/stop/${jobId}`)
export const getInferenceStatus = (jobId) => api.get(`/inference/status/${jobId}`)

// ── Preprocessing ─────────────────────────────────────────────────────────────
export const startPipeline3 = (data) => api.post('/preprocessing/pipeline3/start', data)
export const startRunPipeline = (data) => api.post('/preprocessing/run-pipeline/start', data)
export const stopPreprocessing = (jobId) => api.post(`/preprocessing/stop/${jobId}`)
export const getPreprocessingStatus = (jobId) => api.get(`/preprocessing/status/${jobId}`)

// ── SSE helper (returns EventSource) ─────────────────────────────────────────
export function openLogStream(domain, jobId, onLine, onStatus) {
  const es = new EventSource(`/api/${domain}/stream/${jobId}`)
  es.onmessage = (e) => onLine(e.data)
  es.addEventListener('status', (e) => {
    onStatus(e.data)
    es.close()
  })
  es.onerror = () => {
    onStatus('failed')
    es.close()
  }
  return es
}
