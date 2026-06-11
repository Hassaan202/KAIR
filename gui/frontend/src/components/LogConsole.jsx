import { useEffect, useRef, useState } from 'react'
import { openLogStream } from '../api/client'

function classifyLine(line) {
  const l = line.toLowerCase()
  if (l.includes('error') || l.includes('traceback') || l.includes('exception')) return 'lv-warn'
  if (l.includes('warn')) return 'lv-warn'
  if (l.includes('complete') || l.includes('saved') || l.includes('done')) return 'lv-ok'
  if (l.startsWith('<') || l.includes('iter:')) return 'lv-step'
  if (line.includes('Delta:') || line.includes('average delta') || line.includes('Δ')) return 'lv-delta'
  if (line.trim().startsWith('LR:') || line.includes('average lr')) return 'lv-lr'
  if (line.trim().startsWith('SR:') || line.includes('average sr')) return 'lv-sr'
  return 'lv-info'
}

export default function LogConsole({ domain, jobId, onStop }) {
  const [lines, setLines] = useState([])
  const [status, setStatus] = useState('pending')
  const bottomRef = useRef(null)
  const esRef = useRef(null)

  useEffect(() => {
    if (!jobId) return
    setLines([])
    setStatus('running')

    esRef.current = openLogStream(
      domain,
      jobId,
      (line) => setLines((prev) => [...prev, line]),
      (s) => setStatus(s)
    )

    return () => esRef.current?.close()
  }, [jobId, domain])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [lines])

  if (!jobId) return null

  const isLive = status === 'running' || status === 'pending'

  return (
    <div style={{ marginTop: 20 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <h3 style={{ fontSize: 15, fontWeight: 600, color: 'var(--ink)' }}>Live Output</h3>
          <div className="run-status">
            {isLive && <span className="pulse-dot" />}
            {status}
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
          <span className="mono" style={{ fontSize: 11, color: 'var(--ink-3)' }}>{lines.length} lines</span>
          {isLive && onStop && (
            <button className="btn" style={{ padding: '5px 12px', fontSize: 12, borderColor: 'var(--bad)', color: 'var(--bad)' }} onClick={onStop}>
              Stop Job
            </button>
          )}
        </div>
      </div>
      <div className="logstream scroll">
        {lines.map((line, i) => (
          <div key={i} className={`ln ${classifyLine(line)}`}>
            {line}
          </div>
        ))}
        <span ref={bottomRef} />
      </div>
    </div>
  )
}
