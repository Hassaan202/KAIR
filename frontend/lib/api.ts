import type {
  Job, GpuStatus, LogLine,
  SimplePreprocessConfig, CompletePipelineConfig,
  TrainingJobConfig, InferenceConfig, MetricsResult,
} from "./types";

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...init?.headers },
    ...init,
  });
  if (!res.ok) {
    const err = await res.text().catch(() => res.statusText);
    throw new Error(err);
  }
  return res.json() as Promise<T>;
}

// ---- Status ----
export const getGpuStatus = () => req<GpuStatus>("/status/gpu");
export const getActiveJob = () => req<Job | null>("/status/job");

// ---- Preprocessing ----
export const runSimplePipeline = (config: SimplePreprocessConfig) =>
  req<{ job_id: string }>("/preprocessing/run/simple", { method: "POST", body: JSON.stringify(config) });
export const runCompletePipeline = (config: CompletePipelineConfig) =>
  req<{ job_id: string }>("/preprocessing/run/complete", { method: "POST", body: JSON.stringify(config) });

// ---- Training ----
export const startTraining = (config: TrainingJobConfig) =>
  req<{ job_id: string }>("/training/start", { method: "POST", body: JSON.stringify(config) });
export const stopTraining = (jobId: string) =>
  req<void>(`/training/stop/${jobId}`, { method: "POST" });
export const listCheckpoints = () =>
  req<Array<{ name: string; psnr: number; path: string; size_mb: number; is_best: boolean }>>("/training/checkpoints");
export const deleteCheckpoint = (name: string) =>
  req<void>(`/training/checkpoints/${name}`, { method: "DELETE" });

// ---- Inference ----
export const runInference = (config: InferenceConfig) =>
  req<{ job_id: string }>("/inference/run", { method: "POST", body: JSON.stringify(config) });
export const uploadImage = async (file: File): Promise<{ path: string }> => {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/inference/upload`, { method: "POST", body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
};
export const getInferenceResult = (jobId: string) =>
  req<{ metrics: MetricsResult; sr_path: string } | null>(`/inference/result/${jobId}`);

// ---- Jobs ----
export const getJob = (jobId: string) => req<Job>(`/jobs/${jobId}`);

// ---- SSE log stream ----
export function streamLogs(jobId: string, onLine: (line: LogLine) => void, onDone: () => void): () => void {
  const es = new EventSource(`${BASE}/jobs/${jobId}/logs`);
  es.onmessage = (e) => {
    if (e.data === "[DONE]") { es.close(); onDone(); return; }
    try { onLine(JSON.parse(e.data) as LogLine); } catch { /* ignore malformed */ }
  };
  es.onerror = () => { es.close(); onDone(); };
  return () => es.close();
}
