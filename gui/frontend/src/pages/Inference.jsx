import { useEffect, useState, useRef } from 'react'
import {
  listInferenceTasks, getLatestModel, startInference, stopInference,
  startRawPairedInference, startLROnlyInference,
  getRawInferenceMetrics, getRawResultImageUrl,
} from '../api/client'
import LogConsole from '../components/LogConsole'
import {
  SelectField, TextField, NumberField, BoolToggle,
  ArrayEditor, CollapsibleSection,
} from '../components/FormFields'
import { openLogStream } from '../api/client'

/* ─── Constants ─────────────────────────────────────────────── */
const UPSAMPLER_OPTIONS = ['pixelshuffle', 'pixelshuffledirect', 'nearest+conv']
const RESI_OPTIONS      = ['1conv', '3conv']
const METRIC_NAMES      = ['psnr', 'ssim', 'it_ssim', 'sam', 'uiqi', 'rmse', 'fsim', 'srer']
const METRIC_LABELS     = ['PSNR', 'SSIM', 'IT-SSIM', 'SAM', 'UIQI', 'RMSE', 'FSIM', 'SRER']
const LOWER_IS_BETTER   = new Set(['sam', 'rmse'])

const DEFAULT_MODEL_CONFIG = {
  upscale: 2, in_chans: 3, img_size: 128, window_size: 8,
  img_range: 1.0, depths: [6,6,6,6,6,6], embed_dim: 180,
  num_heads: [6,6,6,6,6,6], mlp_ratio: 2,
  upsampler: 'pixelshuffle', resi_connection: '1conv',
}

const DEFAULT_COREG = {
  enable_preprocessing: true,
  coreg_a_enabled: true, coreg_a_max_features: 8000,
  coreg_a_match_ratio: 0.75, coreg_a_ransac_thresh: 4.0,
  coreg_b_enabled: true, coreg_b_upsample_factor: 100,
  radiometric_enabled: true, radiometric_block_size: 256,
  radiometric_rmse_threshold: 35.0, radiometric_n_samples: 150000,
  radiometric_post_hist_match: true, histogram_n_sample_windows: 40,
  nodata_value: 0, saturated_value: 32767,
  clip_percentiles: [2.0, 98.0], percentile_n_sample_windows: 20,
  coreg_preview_decim_dim: 2000,
}

function deepSet(obj, path, value) {
  const keys = path.split('.')
  const next = { ...obj }
  let cur = next
  for (let i = 0; i < keys.length - 1; i++) {
    cur[keys[i]] = { ...cur[keys[i]] }
    cur = cur[keys[i]]
  }
  cur[keys[keys.length - 1]] = value
  return next
}

/* ─── Reusable model selection panel ───────────────────────── */
function ModelSelectionCard({ modelSource, setModelSource, tasks, selectedTask, setSelectedTask, latestInfo, customModelPath, setCustomModelPath }) {
  return (
    <div className="card" style={{ marginBottom: 16 }}>
      <div className="card-title">Model Selection</div>
      <div className="mode-tabs" style={{ marginBottom: 14 }}>
        <button type="button" className={`mode-tab ${modelSource === 'auto' ? 'active' : ''}`}
          onClick={() => setModelSource('auto')}>Latest from task</button>
        <button type="button" className={`mode-tab ${modelSource === 'custom' ? 'active' : ''}`}
          onClick={() => setModelSource('custom')}>Custom path</button>
      </div>
      {modelSource === 'auto' ? (
        <>
          <SelectField label="Training task" hint="scans superresolution/ for the highest *_E.pth"
            value={selectedTask} onChange={setSelectedTask}
            options={[{ value: '', label: '— select task —' }, ...tasks.map(t => ({ value: t.task_name, label: t.task_name }))]} />
          {latestInfo && (
            <div style={{ background: 'var(--bg-2)', border: '1px solid var(--line-2)', borderRadius: 'var(--radius-sm)', padding: '10px 14px', marginTop: 12 }}>
              <div style={{ fontSize: 12, fontWeight: 500, color: 'var(--ok)' }}>✓ Latest model: iter {latestInfo.latest_iteration}</div>
              <div className="mono" style={{ fontSize: 11, color: 'var(--ink-3)', marginTop: 3 }}>{latestInfo.model_path}</div>
              {Object.keys(latestInfo.model_config || {}).length > 0 ? (
                <div style={{ fontSize: 11, color: 'var(--cobalt-deep)', marginTop: 4, fontWeight: 500 }}>✓ MODEL_CONFIG autofilled from {latestInfo.options_file || 'train.json'}</div>
              ) : (
                <div style={{ fontSize: 11, color: 'var(--bad)', marginTop: 4, fontWeight: 500 }}>⚠ Config not found. Please fill in manually.</div>
              )}
            </div>
          )}
        </>
      ) : (
        <>
          <TextField label="Model path (.pth)" value={customModelPath} onChange={setCustomModelPath}
            placeholder="superresolution/task/models/175000_E.pth" mono />
          <div style={{ fontSize: 11, color: 'var(--ink-3)', marginTop: 8, padding: '8px 12px', background: 'var(--bg-2)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--line-2)' }}>
            ⚠ Please manually enter the model architecture configuration below.
          </div>
        </>
      )}
    </div>
  )
}

/* ─── Model architecture section ───────────────────────────── */
function ModelArchCard({ modelConfig, setMC }) {
  return (
    <CollapsibleSection title="Model Architecture (MODEL_CONFIG)" defaultOpen={false}>
      <div className="grid-2">
        <NumberField label="Upscale" value={modelConfig.upscale} onChange={v => setMC('upscale', v)} min={1} />
        <NumberField label="In channels" value={modelConfig.in_chans} onChange={v => setMC('in_chans', v)} min={1} />
      </div>
      <div className="grid-2">
        <NumberField label="Image size (LR patch)" value={modelConfig.img_size} onChange={v => setMC('img_size', v)} min={16} step={8} />
        <NumberField label="Window size" value={modelConfig.window_size} onChange={v => setMC('window_size', v)} min={4} step={2} />
      </div>
      <div className="grid-2">
        <NumberField label="Embed dim" value={modelConfig.embed_dim} onChange={v => setMC('embed_dim', v)} min={60} step={12} />
        <NumberField label="MLP ratio" value={modelConfig.mlp_ratio} onChange={v => setMC('mlp_ratio', v)} min={1} />
      </div>
      <div className="grid-2">
        <SelectField label="Upsampler" value={modelConfig.upsampler} onChange={v => setMC('upsampler', v)} options={UPSAMPLER_OPTIONS} />
        <SelectField label="Resi connection" value={modelConfig.resi_connection} onChange={v => setMC('resi_connection', v)} options={RESI_OPTIONS} />
      </div>
      <ArrayEditor label="Depths" value={modelConfig.depths} onChange={v => setMC('depths', v)} />
      <ArrayEditor label="Num heads" value={modelConfig.num_heads} onChange={v => setMC('num_heads', v)} />
    </CollapsibleSection>
  )
}

/* ─── Coregistration options panel ─────────────────────────── */
function CoregSection({ coreg, setCoreg }) {
  const set = (k, v) => setCoreg(prev => ({ ...prev, [k]: v }))
  return (
    <CollapsibleSection title="Satellite Preprocessing (Coregistration + Radiometric)" defaultOpen={true}>
      <div style={{ padding: '10px 14px', background: 'var(--bg-2)', border: '1px solid var(--line-2)', borderRadius: 'var(--radius-sm)', marginBottom: 14, fontSize: 12, color: 'var(--ink-2)', lineHeight: 1.5 }}>
        When enabled, runs the full pipeline3.py alignment stack (CRS reproject → Stage A ORB → Stage B Phase Correlation → Radiometric Regression → Histogram Matching) before patch extraction. Required for raw multi-sensor satellite image pairs with fractional resolution ratios.
      </div>
      <BoolToggle label="Enable preprocessing pipeline" value={coreg.enable_preprocessing}
        onChange={v => set('enable_preprocessing', v)} />
      {coreg.enable_preprocessing && (
        <>
          <div style={{ marginTop: 14, fontWeight: 600, fontSize: 12, color: 'var(--ink-2)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 8 }}>Stage A — ORB Coregistration</div>
          <BoolToggle label="Enable Stage A (ORB)" value={coreg.coreg_a_enabled} onChange={v => set('coreg_a_enabled', v)} />
          {coreg.coreg_a_enabled && (
            <div className="grid-2">
              <NumberField label="Max features" value={coreg.coreg_a_max_features} onChange={v => set('coreg_a_max_features', v)} min={100} step={500} />
              <NumberField label="Match ratio" value={coreg.coreg_a_match_ratio} onChange={v => set('coreg_a_match_ratio', v)} min={0.5} max={1.0} step={0.05} />
            </div>
          )}

          <div style={{ marginTop: 14, fontWeight: 600, fontSize: 12, color: 'var(--ink-2)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 8 }}>Stage B — Phase Correlation</div>
          <BoolToggle label="Enable Stage B (Phase)" value={coreg.coreg_b_enabled} onChange={v => set('coreg_b_enabled', v)} />
          {coreg.coreg_b_enabled && (
            <NumberField label="Upsample factor" value={coreg.coreg_b_upsample_factor} onChange={v => set('coreg_b_upsample_factor', v)} min={1} step={10} />
          )}

          <div style={{ marginTop: 14, fontWeight: 600, fontSize: 12, color: 'var(--ink-2)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 8 }}>Radiometric Normalisation</div>
          <BoolToggle label="Enable radiometric regression" value={coreg.radiometric_enabled} onChange={v => set('radiometric_enabled', v)} />
          {coreg.radiometric_enabled && (
            <>
              <div className="grid-2">
                <NumberField label="Block size (px)" value={coreg.radiometric_block_size} onChange={v => set('radiometric_block_size', v)} min={64} step={64} />
                <NumberField label="RMSE threshold" value={coreg.radiometric_rmse_threshold} onChange={v => set('radiometric_rmse_threshold', v)} min={1} step={1} />
              </div>
              <BoolToggle label="Post histogram matching" value={coreg.radiometric_post_hist_match} onChange={v => set('radiometric_post_hist_match', v)} />
            </>
          )}

          <div style={{ marginTop: 14, fontWeight: 600, fontSize: 12, color: 'var(--ink-2)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 8 }}>Pixel Value Settings</div>
          <div className="grid-2">
            <NumberField label="Nodata value" value={coreg.nodata_value} onChange={v => set('nodata_value', v)} />
            <NumberField label="Saturated value" value={coreg.saturated_value} onChange={v => set('saturated_value', v)} />
          </div>
          <ArrayEditor label="Clip percentiles [lo, hi]" value={coreg.clip_percentiles} onChange={v => set('clip_percentiles', v)} />
        </>
      )}
    </CollapsibleSection>
  )
}

/* ─── Metrics table ─────────────────────────────────────────── */
function MetricsTable({ metrics }) {
  if (!metrics) return null
  const { sr, lr_bicubic, delta } = metrics
  const fmt = (v) => (typeof v === 'number' && isFinite(v)) ? v.toFixed(4) : '—'

  return (
    <div style={{ marginTop: 20 }}>
      <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 10, color: 'var(--ink-1)' }}>Metrics</div>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ borderBottom: '2px solid var(--line-2)' }}>
              <th style={{ textAlign: 'left', padding: '6px 10px', color: 'var(--ink-2)', fontWeight: 600 }}>Row</th>
              {METRIC_LABELS.map(l => (
                <th key={l} style={{ textAlign: 'right', padding: '6px 10px', color: 'var(--ink-2)', fontWeight: 600 }}>{l}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            <tr style={{ borderBottom: '1px solid var(--line-2)' }}>
              <td style={{ padding: '6px 10px', fontWeight: 600, color: 'var(--cobalt-deep)' }}>SR</td>
              {METRIC_NAMES.map(k => (
                <td key={k} style={{ textAlign: 'right', padding: '6px 10px', fontFamily: 'monospace', color: 'var(--ink-1)' }}>{fmt(sr?.[k])}</td>
              ))}
            </tr>
            <tr style={{ borderBottom: '1px solid var(--line-2)' }}>
              <td style={{ padding: '6px 10px', fontWeight: 500, color: 'var(--ink-2)' }}>Bicubic LR</td>
              {METRIC_NAMES.map(k => (
                <td key={k} style={{ textAlign: 'right', padding: '6px 10px', fontFamily: 'monospace', color: 'var(--ink-2)' }}>{fmt(lr_bicubic?.[k])}</td>
              ))}
            </tr>
            <tr>
              <td style={{ padding: '6px 10px', fontWeight: 600, color: 'var(--ink-1)' }}>Δ (SR − LR)</td>
              {METRIC_NAMES.map(k => {
                const d = delta?.[k]
                const lowerBetter = LOWER_IS_BETTER.has(k)
                const improved = typeof d === 'number' && (lowerBetter ? d < 0 : d > 0)
                const worsened = typeof d === 'number' && (lowerBetter ? d > 0 : d < 0)
                return (
                  <td key={k} style={{
                    textAlign: 'right', padding: '6px 10px', fontFamily: 'monospace', fontWeight: 600,
                    color: improved ? 'var(--ok)' : worsened ? 'var(--bad)' : 'var(--ink-2)',
                  }}>
                    {typeof d === 'number' && isFinite(d) ? (d >= 0 ? '+' : '') + d.toFixed(4) : '—'}
                  </td>
                )
              })}
            </tr>
          </tbody>
        </table>
      </div>
      {/* Summary line */}
      <div style={{ marginTop: 10, fontSize: 11, color: 'var(--ink-3)', display: 'flex', gap: 12, flexWrap: 'wrap' }}>
        {METRIC_NAMES.map(k => {
          const d = delta?.[k]
          if (typeof d !== 'number' || !isFinite(d)) return null
          const lowerBetter = LOWER_IS_BETTER.has(k)
          const improved = lowerBetter ? d < 0 : d > 0
          return (
            <span key={k} style={{ color: improved ? 'var(--ok)' : 'var(--bad)' }}>
              Δ{k.toUpperCase().replace('_', '-')} {d >= 0 ? '+' : ''}{d.toFixed(3)}
            </span>
          )
        })}
      </div>
    </div>
  )
}

/* ─── Image viewer ──────────────────────────────────────────── */
function ImageViewer({ images }) {
  const [lightbox, setLightbox] = useState(null)

  return (
    <>
      <div style={{ marginTop: 20 }}>
        <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 10, color: 'var(--ink-1)' }}>Result Images</div>
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
          {images.map(({ label, url }) => (
            <div key={label} style={{ flex: '1 1 140px', minWidth: 120, cursor: 'pointer' }} onClick={() => setLightbox(url)}>
              <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--ink-2)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.05em' }}>{label}</div>
              <img src={url} alt={label}
                style={{ width: '100%', borderRadius: 'var(--radius-sm)', border: '1px solid var(--line-2)', objectFit: 'cover', aspectRatio: '1/1', background: 'var(--bg-2)' }}
                onError={e => { e.target.style.opacity = 0.3 }}
              />
              <div style={{ fontSize: 10, color: 'var(--ink-3)', marginTop: 4, textAlign: 'center' }}>Click to enlarge</div>
            </div>
          ))}
        </div>
      </div>

      {lightbox && (
        <div onClick={() => setLightbox(null)} style={{
          position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.85)', zIndex: 9999,
          display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'zoom-out',
        }}>
          <img src={lightbox} alt="enlarged"
            style={{ maxWidth: '90vw', maxHeight: '90vh', borderRadius: 8, boxShadow: '0 0 60px rgba(0,0,0,0.8)' }} />
          <div style={{ position: 'absolute', top: 20, right: 24, color: '#fff', fontSize: 28, fontWeight: 300, lineHeight: 1 }}>×</div>
        </div>
      )}
    </>
  )
}

/* ─── TAB 1: Patched Images (existing) ─────────────────────── */
function PatchedTab({ tasks }) {
  const [modelSource, setModelSource]         = useState('auto')
  const [selectedTask, setSelectedTask]       = useState('')
  const [latestInfo, setLatestInfo]           = useState(null)
  const [customModelPath, setCustomModelPath] = useState('')
  const [modelConfig, setModelConfig]         = useState(DEFAULT_MODEL_CONFIG)
  const [inferConfig, setInferConfig]         = useState({
    lr_dir: '', hr_dir: '', sr_dir: 'testsets/output/sr',
    tile: '', tile_overlap: 32, overwrite_sr: true, log_dir: 'testsets/output',
  })
  const [jobId, setJobId]   = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]   = useState('')

  const setMC = (path, value) => setModelConfig(prev => deepSet(prev, path, value))
  const setIC = (key, value) => setInferConfig(prev => ({ ...prev, [key]: value }))

  useEffect(() => {
    if (modelSource === 'auto' && selectedTask) {
      getLatestModel(selectedTask)
        .then(r => {
          setLatestInfo(r.data)
          if (r.data.model_config && Object.keys(r.data.model_config).length > 0)
            setModelConfig({ ...DEFAULT_MODEL_CONFIG, ...r.data.model_config })
        })
        .catch(() => setLatestInfo(null))
    }
  }, [selectedTask, modelSource])

  const handleSubmit = async (e) => {
    e.preventDefault(); setError(''); setLoading(true)
    try {
      const modelPath = modelSource === 'auto' ? (latestInfo?.model_path || '') : customModelPath
      const payload = {
        model_path: modelPath, lr_dir: inferConfig.lr_dir, hr_dir: inferConfig.hr_dir,
        sr_dir: inferConfig.sr_dir, tile: inferConfig.tile ? parseInt(inferConfig.tile) : null,
        tile_overlap: inferConfig.tile_overlap, overwrite_sr: inferConfig.overwrite_sr,
        log_dir: inferConfig.log_dir, model_config: modelConfig,
      }
      const r = await startInference(payload)
      setJobId(r.data.job_id)
    } catch (err) {
      setError(err.response?.data?.detail || String(err))
    } finally { setLoading(false) }
  }

  return (
    <div className="module-grid rise" style={{ animationDelay: '100ms' }}>
      <div className="col">
        <form onSubmit={handleSubmit}>
          <ModelSelectionCard
            modelSource={modelSource} setModelSource={setModelSource}
            tasks={tasks} selectedTask={selectedTask} setSelectedTask={setSelectedTask}
            latestInfo={latestInfo} customModelPath={customModelPath} setCustomModelPath={setCustomModelPath}
          />
          <CollapsibleSection title="Input / Output Paths" defaultOpen>
            <TextField label="LR image dir" value={inferConfig.lr_dir}
              onChange={v => setIC('lr_dir', v)} placeholder="testsets/my_test/lr" />
            <TextField label="HR image dir" hint="ground truth for metrics"
              value={inferConfig.hr_dir} onChange={v => setIC('hr_dir', v)} placeholder="testsets/my_test/hr" />
            <TextField label="SR output dir" value={inferConfig.sr_dir}
              onChange={v => setIC('sr_dir', v)} placeholder="testsets/my_test/sr" />
            <TextField label="Log dir" value={inferConfig.log_dir}
              onChange={v => setIC('log_dir', v)} placeholder="testsets/my_test" />
            <div className="grid-2">
              <div className="form-group">
                <label>Tile size <span className="hint">blank = full image</span></label>
                <input type="number" className="num-input" value={inferConfig.tile || ''}
                  onChange={e => setIC('tile', e.target.value)} placeholder="e.g. 256" min={64} step={8} />
              </div>
              <NumberField label="Tile overlap" value={inferConfig.tile_overlap}
                onChange={v => setIC('tile_overlap', v)} min={0} step={8} />
            </div>
            <BoolToggle label="Overwrite existing SR images" value={inferConfig.overwrite_sr}
              onChange={v => setIC('overwrite_sr', v)} />
          </CollapsibleSection>
          <ModelArchCard modelConfig={modelConfig} setMC={setMC} />
          {error && <div style={{ color: 'var(--bad)', fontSize: 13, marginBottom: 12 }}>{error}</div>}
          <button type="submit" className="btn btn-primary full-width"
            disabled={loading || (modelSource === 'auto' && !selectedTask)}>
            {loading ? 'Starting…' : '▶ Run Inference'}
          </button>
        </form>
      </div>
      <div className="col">
        <div className="card">
          <div className="card-title">Metrics Evaluation</div>
          <p className="text-muted text-sm" style={{ lineHeight: 1.5 }}>
            Inference computes 8 metrics — PSNR, SSIM, IT-SSIM, SAM, UIQI, RMSE, FSIM, SRER —
            for both the SR output and the bicubic baseline, then reports deltas to show improvement over simple upsampling.
          </p>
          <div className="mono" style={{ fontSize: 11, color: 'var(--ink-3)', marginTop: 12, padding: 8, background: 'var(--bg)', borderRadius: 'var(--radius-sm)' }}>
            SR rows · LR rows · Delta rows — averaged across all test images.<br />
            Results saved to: log_dir/&lt;sr_dir_name&gt;.log
          </div>
        </div>
        {jobId && <LogConsole domain="inference" jobId={jobId} onStop={() => stopInference(jobId).catch(() => {})} />}
      </div>
    </div>
  )
}

/* ─── TAB 2: Raw HR+LR Inference ───────────────────────────── */
function RawPairedTab({ tasks }) {
  const [modelSource, setModelSource]         = useState('auto')
  const [selectedTask, setSelectedTask]       = useState('')
  const [latestInfo, setLatestInfo]           = useState(null)
  const [customModelPath, setCustomModelPath] = useState('')
  const [modelConfig, setModelConfig]         = useState(DEFAULT_MODEL_CONFIG)
  const [coreg, setCoreg]                     = useState(DEFAULT_COREG)
  const [config, setConfig]                   = useState({
    lr_path: '', hr_path: '', lr_bands: [3,2,1], hr_bands: [1,2,3],
    output_dir: 'testsets/raw_inference_output', patch_size: 128,
    overlap: 32, scale_factor: 2,
  })
  const [jobId, setJobId]     = useState(null)
  const [jobDone, setJobDone] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState('')
  const [metrics, setMetrics] = useState(null)
  const esRef                 = useRef(null)

  const setMC  = (path, value) => setModelConfig(prev => deepSet(prev, path, value))
  const setCfg = (key, value) => setConfig(prev => ({ ...prev, [key]: value }))

  useEffect(() => {
    if (modelSource === 'auto' && selectedTask) {
      getLatestModel(selectedTask)
        .then(r => {
          setLatestInfo(r.data)
          if (r.data.model_config && Object.keys(r.data.model_config).length > 0)
            setModelConfig({ ...DEFAULT_MODEL_CONFIG, ...r.data.model_config })
        })
        .catch(() => setLatestInfo(null))
    }
  }, [selectedTask, modelSource])

  const handleSubmit = async (e) => {
    e.preventDefault(); setError(''); setLoading(true)
    setJobId(null); setJobDone(false); setMetrics(null)
    try {
      const modelPath = modelSource === 'auto' ? (latestInfo?.model_path || '') : customModelPath
      const payload = {
        ...config,
        model_path: modelPath,
        model_network_config: modelConfig,
        coreg,
      }
      const r = await startRawPairedInference(payload)
      const jid = r.data.job_id
      setJobId(jid)
      // Stream logs + poll for metrics on completion
      esRef.current = openLogStream('inference', jid, () => {}, (status) => {
        setJobDone(true)
        if (status === 'completed') {
          // Fetch metrics after job finishes
          setTimeout(() => {
            getRawInferenceMetrics(jid).then(m => setMetrics(m.data)).catch(() => {})
          }, 500)
        }
      })
    } catch (err) {
      setError(err.response?.data?.detail || String(err))
    } finally { setLoading(false) }
  }

  const resultImages = jobDone && jobId ? [
    { label: 'LR', url: getRawResultImageUrl(jobId, 'lr_display.png') },
    { label: 'SR', url: getRawResultImageUrl(jobId, 'sr_display.png') },
    { label: 'HR', url: getRawResultImageUrl(jobId, 'hr_display.png') },
  ] : []

  return (
    <div className="module-grid rise" style={{ animationDelay: '100ms' }}>
      {/* Left: config */}
      <div className="col">
        <form onSubmit={handleSubmit}>
          <ModelSelectionCard
            modelSource={modelSource} setModelSource={setModelSource}
            tasks={tasks} selectedTask={selectedTask} setSelectedTask={setSelectedTask}
            latestInfo={latestInfo} customModelPath={customModelPath} setCustomModelPath={setCustomModelPath}
          />

          <div className="card" style={{ marginBottom: 16 }}>
            <div className="card-title">Input Images</div>
            <TextField label="LR raw image path" value={config.lr_path}
              onChange={v => setCfg('lr_path', v)} placeholder="/path/to/lr.tif" mono />
            <TextField label="HR raw image path" hint="ground truth"
              value={config.hr_path} onChange={v => setCfg('hr_path', v)} placeholder="/path/to/hr.tif" mono />
            <div className="grid-2">
              <ArrayEditor label="LR RGB bands" value={config.lr_bands}
                onChange={v => setCfg('lr_bands', v)} />
              <ArrayEditor label="HR RGB bands" value={config.hr_bands}
                onChange={v => setCfg('hr_bands', v)} />
            </div>
            <TextField label="Output directory" value={config.output_dir}
              onChange={v => setCfg('output_dir', v)} placeholder="testsets/raw_inference_output" mono />
            <div className="grid-2">
              <NumberField label="Patch size (LR px)" value={config.patch_size}
                onChange={v => setCfg('patch_size', v)} min={32} step={8} />
              <NumberField label="Overlap (LR px)" value={config.overlap}
                onChange={v => setCfg('overlap', v)} min={0} step={8} />
            </div>
            <NumberField label="Scale factor" value={config.scale_factor}
              onChange={v => setCfg('scale_factor', v)} min={1} max={8} />
          </div>

          <CoregSection coreg={coreg} setCoreg={setCoreg} />
          <ModelArchCard modelConfig={modelConfig} setMC={setMC} />

          {error && <div style={{ color: 'var(--bad)', fontSize: 13, marginBottom: 12 }}>{error}</div>}
          <button type="submit" className="btn btn-primary full-width"
            disabled={loading || !config.lr_path || !config.hr_path || (modelSource === 'auto' && !selectedTask)}>
            {loading ? 'Starting…' : '▶ Run Paired Inference'}
          </button>
        </form>
      </div>

      {/* Right: log + results */}
      <div className="col">
        {jobId && <LogConsole domain="inference" jobId={jobId} onStop={() => {}} />}
        {jobDone && (
          <div className="card" style={{ marginTop: 16 }}>
            <div className="card-title">Results</div>
            {resultImages.length > 0 && <ImageViewer images={resultImages} />}
            <MetricsTable metrics={metrics} />
            {!metrics && (
              <div style={{ marginTop: 16, fontSize: 12, color: 'var(--ink-3)' }}>
                {jobId ? 'Loading metrics…' : 'Metrics will appear after job completes.'}
              </div>
            )}
          </div>
        )}
        {!jobId && (
          <div className="card">
            <div className="card-title">About</div>
            <p className="text-muted text-sm" style={{ lineHeight: 1.6 }}>
              Runs SwinIR on a raw LR satellite image paired with an HR ground truth.
              Optionally performs full pipeline3.py alignment (Stage A ORB + Stage B Phase Correlation
              + Radiometric Regression + Histogram Matching) to align the LR image to the HR pixel grid,
              enforcing an exact ×{config.scale_factor} scale factor regardless of the real sensor
              resolution ratio (e.g. 1.7×, 2.3×).
            </p>
            <p className="text-muted text-sm" style={{ lineHeight: 1.6, marginTop: 10 }}>
              Outputs: <strong>LR · SR · HR display images</strong> + full <strong>8-metric table</strong>
              (PSNR, SSIM, IT-SSIM, SAM, UIQI, RMSE, FSIM, SRER) vs bicubic baseline.
            </p>
          </div>
        )}
      </div>
    </div>
  )
}

/* ─── TAB 3: LR-Only Inference ──────────────────────────────── */
function LROnlyTab({ tasks }) {
  const [modelSource, setModelSource]         = useState('auto')
  const [selectedTask, setSelectedTask]       = useState('')
  const [latestInfo, setLatestInfo]           = useState(null)
  const [customModelPath, setCustomModelPath] = useState('')
  const [modelConfig, setModelConfig]         = useState(DEFAULT_MODEL_CONFIG)
  const [config, setConfig]                   = useState({
    lr_path: '', lr_bands: [1,2,3],
    output_dir: 'testsets/raw_inference_output/lr_only',
    patch_size: 128, overlap: 32, scale_factor: 2,
  })
  const [jobId, setJobId]     = useState(null)
  const [jobDone, setJobDone] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState('')

  const setMC  = (path, value) => setModelConfig(prev => deepSet(prev, path, value))
  const setCfg = (key, value) => setConfig(prev => ({ ...prev, [key]: value }))

  useEffect(() => {
    if (modelSource === 'auto' && selectedTask) {
      getLatestModel(selectedTask)
        .then(r => {
          setLatestInfo(r.data)
          if (r.data.model_config && Object.keys(r.data.model_config).length > 0)
            setModelConfig({ ...DEFAULT_MODEL_CONFIG, ...r.data.model_config })
        })
        .catch(() => setLatestInfo(null))
    }
  }, [selectedTask, modelSource])

  const handleSubmit = async (e) => {
    e.preventDefault(); setError(''); setLoading(true)
    setJobId(null); setJobDone(false)
    try {
      const modelPath = modelSource === 'auto' ? (latestInfo?.model_path || '') : customModelPath
      const payload = { ...config, model_path: modelPath, model_network_config: modelConfig }
      const r = await startLROnlyInference(payload)
      setJobId(r.data.job_id)
      openLogStream('inference', r.data.job_id, () => {}, () => setJobDone(true))
    } catch (err) {
      setError(err.response?.data?.detail || String(err))
    } finally { setLoading(false) }
  }

  const resultImages = jobDone && jobId ? [
    { label: 'LR Input', url: getRawResultImageUrl(jobId, 'lr_display.png') },
    { label: 'SR Output', url: getRawResultImageUrl(jobId, 'sr_display.png') },
  ] : []

  return (
    <div className="module-grid rise" style={{ animationDelay: '100ms' }}>
      {/* Left: config */}
      <div className="col">
        <form onSubmit={handleSubmit}>
          <ModelSelectionCard
            modelSource={modelSource} setModelSource={setModelSource}
            tasks={tasks} selectedTask={selectedTask} setSelectedTask={setSelectedTask}
            latestInfo={latestInfo} customModelPath={customModelPath} setCustomModelPath={setCustomModelPath}
          />
          <div className="card" style={{ marginBottom: 16 }}>
            <div className="card-title">Input Image</div>
            <TextField label="LR raw image path" value={config.lr_path}
              onChange={v => setCfg('lr_path', v)} placeholder="/path/to/image.tif" mono />
            <ArrayEditor label="LR RGB bands" value={config.lr_bands}
              onChange={v => setCfg('lr_bands', v)} />
            <TextField label="Output directory" value={config.output_dir}
              onChange={v => setCfg('output_dir', v)} placeholder="testsets/raw_inference_output/lr_only" mono />
            <div className="grid-2">
              <NumberField label="Patch size (LR px)" value={config.patch_size}
                onChange={v => setCfg('patch_size', v)} min={32} step={8} />
              <NumberField label="Overlap (LR px)" value={config.overlap}
                onChange={v => setCfg('overlap', v)} min={0} step={8} />
            </div>
            <NumberField label="Scale factor" value={config.scale_factor}
              onChange={v => setCfg('scale_factor', v)} min={1} max={8} />
          </div>
          <ModelArchCard modelConfig={modelConfig} setMC={setMC} />
          {error && <div style={{ color: 'var(--bad)', fontSize: 13, marginBottom: 12 }}>{error}</div>}
          <button type="submit" className="btn btn-primary full-width"
            disabled={loading || !config.lr_path || (modelSource === 'auto' && !selectedTask)}>
            {loading ? 'Starting…' : '▶ Run LR-Only Inference'}
          </button>
        </form>
      </div>

      {/* Right: log + results */}
      <div className="col">
        {jobId && <LogConsole domain="inference" jobId={jobId} onStop={() => {}} />}
        {jobDone && resultImages.length > 0 && (
          <div className="card" style={{ marginTop: 16 }}>
            <div className="card-title">Results</div>
            <ImageViewer images={resultImages} />
          </div>
        )}
        {!jobId && (
          <div className="card">
            <div className="card-title">About</div>
            <p className="text-muted text-sm" style={{ lineHeight: 1.6 }}>
              Runs SwinIR on a single LR image with no HR ground truth. No metrics are
              computed — only the LR input and SR output are saved and displayed.
              Useful for real-world inference on unlabelled satellite imagery.
            </p>
            <p className="text-muted text-sm" style={{ lineHeight: 1.6, marginTop: 10 }}>
              Supports GeoTIFF, JP2, PNG, and other standard image formats.
              Band selection applies only to multi-band geospatial files.
            </p>
          </div>
        )}
      </div>
    </div>
  )
}

/* ─── Main page ─────────────────────────────────────────────── */
export default function Inference() {
  const [activeTab, setActiveTab] = useState(0)
  const [tasks, setTasks]         = useState([])

  useEffect(() => {
    listInferenceTasks().then(r => setTasks(r.data)).catch(() => {})
  }, [])

  const tabs = [
    { label: 'Patched Images',     icon: '▦' },
    { label: 'Raw HR+LR Inference', icon: '⟳' },
    { label: 'LR-Only Inference',  icon: '↑' },
  ]

  return (
    <div>
      <div className="topbar">
        <div className="topbar-title"><h2>Inference</h2></div>
      </div>

      <div className="content">
        <h1 className="editorial rise" style={{ fontSize: 32, marginBottom: 10 }}>Run Inference and Metrics</h1>
        <p className="rise" style={{ color: 'var(--ink-2)', marginBottom: 24, maxWidth: 680 }}>
          Run a trained SwinIR model on LR images and compute super-resolution quality metrics.
          Choose a mode below — patched directory inference, raw paired satellite image inference,
          or LR-only inference without ground truth.
        </p>

        {/* Top-level tab bar */}
        <div className="mode-tabs rise" style={{ marginBottom: 28, animationDelay: '80ms' }}>
          {tabs.map((t, i) => (
            <button key={i} type="button"
              className={`mode-tab ${activeTab === i ? 'active' : ''}`}
              onClick={() => setActiveTab(i)}>
              <span style={{ marginRight: 6 }}>{t.icon}</span>{t.label}
            </button>
          ))}
        </div>

        {activeTab === 0 && <PatchedTab tasks={tasks} />}
        {activeTab === 1 && <RawPairedTab tasks={tasks} />}
        {activeTab === 2 && <LROnlyTab tasks={tasks} />}
      </div>
    </div>
  )
}
