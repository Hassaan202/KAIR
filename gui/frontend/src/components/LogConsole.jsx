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
  if (line.includes('PREVIEW_READY')) return 'lv-preview'
  return 'lv-info'
}

const PREVIEW_STAGES = [
  { key: 'load_hr',     label: 'HR loaded' },
  { key: 'load_lr',     label: 'LR loaded' },
  { key: 'coreg_a',     label: 'Coreg — ORB' },
  { key: 'coreg_b',     label: 'Coreg — Phase' },
  { key: 'radiometric', label: 'Radiometric' },
  { key: 'patches',     label: 'Sample patches' },
]

const PREVIEW_MARKER_RE = /PREVIEW_READY (\S+) (\S+) (\S+)/

function Lightbox({ preview, label, onClose }) {
  useEffect(() => {
    const handler = (e) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [onClose])

  return (
    <div className="lightbox-overlay" onClick={onClose}>
      <div className="lightbox-box" onClick={(e) => e.stopPropagation()}>
        <div className="lightbox-header">
          <span className="lightbox-title">{label}</span>
          <span className="lightbox-scene">{preview.scene}</span>
          <button className="lightbox-close" onClick={onClose}>✕</button>
        </div>
        <img src={preview.url} alt={label} className="lightbox-img" />
      </div>
    </div>
  )
}

export default function LogConsole({ domain, jobId, onStop, onComplete, onLine, onPreviewsChange }) {
  const [lines, setLines] = useState([])
  const [status, setStatus] = useState('pending')
  const [previews, setPreviews] = useState({})
  const [lightbox, setLightbox] = useState(null)
  const bottomRef = useRef(null)
  const esRef = useRef(null)
  const previewsRef = useRef({})
  const onCompleteRef = useRef(onComplete)
  const onLineRef = useRef(onLine)
  const onPreviewsChangeRef = useRef(onPreviewsChange)
  onCompleteRef.current = onComplete   // keep ref fresh without re-running effect
  onLineRef.current = onLine
  onPreviewsChangeRef.current = onPreviewsChange

  useEffect(() => {
    if (!jobId) return
    setLines([])
    previewsRef.current = {}
    setPreviews({})
    setStatus('running')

    esRef.current = openLogStream(
      domain,
      jobId,
      (line) => {
        setLines((prev) => [...prev, line])
        if (onLineRef.current) onLineRef.current(line)
        if (domain === 'preprocessing') {
          const match = line.match(PREVIEW_MARKER_RE)
          if (match) {
            const [, filename, stage, scene] = match
            const url = `/api/preprocessing/preview/${jobId}/${encodeURIComponent(filename)}`
            const next = { ...previewsRef.current, [stage]: { url, scene } }
            previewsRef.current = next
            setPreviews(next)
            onPreviewsChangeRef.current?.(next)
          }
        }
      },
      (s) => {
        setStatus(s)
        if (s.toLowerCase().includes('completed') && onCompleteRef.current) {
          onCompleteRef.current()
        }
      }
    )

    return () => esRef.current?.close()
  }, [jobId, domain])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [lines])

  if (!jobId) return null

  const isLive = status.toLowerCase().includes('running') || status.toLowerCase().includes('pending')
  const visibleStages = PREVIEW_STAGES.filter((s) => previews[s.key])

  return (
    <div style={{ marginTop: 20 }}>
      {lightbox && (
        <Lightbox
          preview={previews[lightbox.key]}
          label={lightbox.label}
          onClose={() => setLightbox(null)}
        />
      )}

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

      {visibleStages.length > 0 && (
        <div className="preview-strip">
          {visibleStages.map((s) => (
            <button key={s.key} className="preview-thumb" onClick={() => setLightbox(s)}>
              <img src={previews[s.key].url} alt={s.label} />
              <span className="preview-label">{s.label}</span>
              <span className="preview-scene">{previews[s.key].scene}</span>
            </button>
          ))}
        </div>
      )}

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
