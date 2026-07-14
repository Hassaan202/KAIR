import { createContext, useContext, useState, useEffect } from 'react'

const JobContext = createContext(null)

const STORAGE_KEY = 'kair_jobs'
const DOMAINS = ['training', 'preprocessing', 'inference-patched', 'inference-raw', 'inference-lr']

function loadJobs() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}')
    return Object.fromEntries(DOMAINS.map((d) => [d, saved[d] ?? null]))
  } catch {
    return Object.fromEntries(DOMAINS.map((d) => [d, null]))
  }
}

export function JobProvider({ children }) {
  const [jobs, setJobs] = useState(loadJobs)

  useEffect(() => {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(jobs)) } catch {}
  }, [jobs])

  const setJobId = (domain, jobId) =>
    setJobs((prev) => ({ ...prev, [domain]: jobId }))

  const clearJobId = (domain) =>
    setJobs((prev) => ({ ...prev, [domain]: null }))

  return (
    <JobContext.Provider value={{ jobs, setJobId, clearJobId }}>
      {children}
    </JobContext.Provider>
  )
}

export function useJobContext() {
  const ctx = useContext(JobContext)
  if (!ctx) throw new Error('useJobContext must be used inside JobProvider')
  return ctx
}
