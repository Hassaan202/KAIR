import { useEffect, useState } from 'react'
import { listInferenceTasks, getLatestModel, startInference, stopInference } from '../api/client'
import LogConsole from '../components/LogConsole'
import {
  SelectField, TextField, NumberField, BoolToggle,
  ArrayEditor, CollapsibleSection
} from '../components/FormFields'

const UPSAMPLER_OPTIONS = ['pixelshuffle', 'pixelshuffledirect', 'nearest+conv']
const RESI_OPTIONS = ['1conv', '3conv']

const DEFAULT_MODEL_CONFIG = {
  upscale: 2, in_chans: 3, img_size: 128, window_size: 8,
  img_range: 1.0, depths: [6,6,6,6,6,6], embed_dim: 180,
  num_heads: [6,6,6,6,6,6], mlp_ratio: 2,
  upsampler: 'pixelshuffle', resi_connection: '1conv',
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

export default function Inference() {
  const [modelSource, setModelSource] = useState('auto')
  const [tasks, setTasks] = useState([])
  const [selectedTask, setSelectedTask] = useState('')
  const [latestInfo, setLatestInfo] = useState(null)
  const [customModelPath, setCustomModelPath] = useState('')
  const [modelConfig, setModelConfig] = useState(DEFAULT_MODEL_CONFIG)
  const [inferConfig, setInferConfig] = useState({
    lr_dir: '', hr_dir: '', sr_dir: 'testsets/output/sr',
    tile: '', tile_overlap: 32, overwrite_sr: true, log_dir: 'testsets/output',
  })
  const [jobId, setJobId] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const setMC = (path, value) => setModelConfig((prev) => deepSet(prev, path, value))
  const setIC = (key, value) => setInferConfig((prev) => ({ ...prev, [key]: value }))

  useEffect(() => {
    listInferenceTasks().then((r) => setTasks(r.data)).catch(() => {})
  }, [])

  useEffect(() => {
    if (modelSource === 'auto' && selectedTask) {
      getLatestModel(selectedTask)
        .then((r) => {
          setLatestInfo(r.data)
          if (r.data.model_config && Object.keys(r.data.model_config).length > 0) {
            setModelConfig({ ...DEFAULT_MODEL_CONFIG, ...r.data.model_config })
          }
        })
        .catch(() => setLatestInfo(null))
    }
  }, [selectedTask, modelSource])

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      const modelPath = modelSource === 'auto'
        ? (latestInfo?.model_path || '')
        : customModelPath
      const payload = {
        model_path: modelPath,
        lr_dir: inferConfig.lr_dir,
        hr_dir: inferConfig.hr_dir,
        sr_dir: inferConfig.sr_dir,
        tile: inferConfig.tile ? parseInt(inferConfig.tile) : null,
        tile_overlap: inferConfig.tile_overlap,
        overwrite_sr: inferConfig.overwrite_sr,
        log_dir: inferConfig.log_dir,
        model_config: modelConfig,
      }
      const r = await startInference(payload)
      setJobId(r.data.job_id)
    } catch (err) {
      setError(err.response?.data?.detail || String(err))
    } finally {
      setLoading(false)
    }
  }

  const handleStop = async () => {
    if (jobId) await stopInference(jobId).catch(() => {})
  }

  return (
    <div>
      <div className="topbar">
        <div className="topbar-title">
          <h2>Inference</h2>
          <span className="crumb">/ swinir / evaluate</span>
        </div>
      </div>

      <div className="content">
        <div className="eyebrow rise">Evaluation</div>
        <h1 className="editorial rise" style={{ fontSize: 32, marginBottom: 10 }}>Run Inference and Metrics</h1>
        <p className="rise" style={{ color: 'var(--ink-2)', marginBottom: 30, maxWidth: 600 }}>
          Run a trained SwinIR model on LR patches and compute super-resolution metrics against HR ground truth.
        </p>

        <div className="module-grid rise" style={{ animationDelay: '100ms' }}>
          {/* ─── Left: Config ─────────────────────────────── */}
          <div className="col">
            <form onSubmit={handleSubmit}>
              {/* Model source */}
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
                    <SelectField
                      label="Training task"
                      hint="scans superresolution/ for the highest *_E.pth"
                      value={selectedTask}
                      onChange={setSelectedTask}
                      options={[
                        { value: '', label: '— select task —' },
                        ...tasks.map((t) => ({ value: t.task_name, label: t.task_name }))
                      ]}
                    />
                    {latestInfo && (
                      <div style={{ background: 'var(--bg-2)', border: '1px solid var(--line-2)', borderRadius: 'var(--radius-sm)', padding: '10px 14px', marginTop: 12 }}>
                        <div style={{ fontSize: 12, fontWeight: 500, color: 'var(--ok)' }}>
                          ✓ Latest model: iter {latestInfo.latest_iteration}
                        </div>
                        <div className="mono" style={{ fontSize: 11, color: 'var(--ink-3)', marginTop: 3 }}>
                          {latestInfo.model_path}
                        </div>
                        {Object.keys(latestInfo.model_config || {}).length > 0 && (
                          <div style={{ fontSize: 11, color: 'var(--cobalt-deep)', marginTop: 4, fontWeight: 500 }}>
                            ✓ MODEL_CONFIG autofilled from train.json
                          </div>
                        )}
                      </div>
                    )}
                  </>
                ) : (
                  <TextField
                    label="Model path (.pth)"
                    value={customModelPath}
                    onChange={setCustomModelPath}
                    placeholder="superresolution/task/models/175000_E.pth"
                    mono
                  />
                )}
              </div>

              {/* Inference paths */}
              <CollapsibleSection title="Input / Output Paths" defaultOpen>
                <TextField label="LR image dir" value={inferConfig.lr_dir}
                  onChange={(v) => setIC('lr_dir', v)} placeholder="testsets/my_test/lr" />
                <TextField label="HR image dir" hint="ground truth for metrics"
                  value={inferConfig.hr_dir} onChange={(v) => setIC('hr_dir', v)}
                  placeholder="testsets/my_test/hr" />
                <TextField label="SR output dir" value={inferConfig.sr_dir}
                  onChange={(v) => setIC('sr_dir', v)} placeholder="testsets/my_test/sr" />
                <TextField label="Log dir" value={inferConfig.log_dir}
                  onChange={(v) => setIC('log_dir', v)} placeholder="testsets/my_test" />
                <div className="grid-2">
                  <div className="form-group">
                    <label>Tile size <span className="hint">blank = full image</span></label>
                    <input type="number" className="num-input" value={inferConfig.tile || ''}
                      onChange={(e) => setIC('tile', e.target.value)}
                      placeholder="e.g. 256" min={64} step={8} />
                  </div>
                  <NumberField label="Tile overlap" value={inferConfig.tile_overlap}
                    onChange={(v) => setIC('tile_overlap', v)} min={0} step={8} />
                </div>
                <BoolToggle label="Overwrite existing SR images"
                  value={inferConfig.overwrite_sr}
                  onChange={(v) => setIC('overwrite_sr', v)} />
              </CollapsibleSection>

              {/* MODEL_CONFIG */}
              <CollapsibleSection title="Model Architecture (MODEL_CONFIG)" defaultOpen={false}>
                <div className="grid-2">
                  <NumberField label="Upscale" value={modelConfig.upscale}
                    onChange={(v) => setMC('upscale', v)} min={1} />
                  <NumberField label="In channels" value={modelConfig.in_chans}
                    onChange={(v) => setMC('in_chans', v)} min={1} />
                </div>
                <div className="grid-2">
                  <NumberField label="Image size (LR patch)" value={modelConfig.img_size}
                    onChange={(v) => setMC('img_size', v)} min={16} step={8} />
                  <NumberField label="Window size" value={modelConfig.window_size}
                    onChange={(v) => setMC('window_size', v)} min={4} step={2} />
                </div>
                <div className="grid-2">
                  <NumberField label="Embed dim" value={modelConfig.embed_dim}
                    onChange={(v) => setMC('embed_dim', v)} min={60} step={12} />
                  <NumberField label="MLP ratio" value={modelConfig.mlp_ratio}
                    onChange={(v) => setMC('mlp_ratio', v)} min={1} />
                </div>
                <div className="grid-2">
                  <SelectField label="Upsampler" value={modelConfig.upsampler}
                    onChange={(v) => setMC('upsampler', v)} options={UPSAMPLER_OPTIONS} />
                  <SelectField label="Resi connection" value={modelConfig.resi_connection}
                    onChange={(v) => setMC('resi_connection', v)} options={RESI_OPTIONS} />
                </div>
                <ArrayEditor label="Depths" value={modelConfig.depths}
                  onChange={(v) => setMC('depths', v)} />
                <ArrayEditor label="Num heads" value={modelConfig.num_heads}
                  onChange={(v) => setMC('num_heads', v)} />
              </CollapsibleSection>

              {error && (
                <div style={{ color: 'var(--bad)', fontSize: 13, marginBottom: 12 }}>{error}</div>
              )}

              <button type="submit" className="btn btn-primary full-width"
                disabled={loading || (modelSource === 'auto' && !selectedTask)}>
                {loading ? 'Starting…' : '▶ Run Inference'}
              </button>
            </form>
          </div>

          {/* ─── Right: Log ───────────────────────────────── */}
          <div className="col">
            <div className="card">
              <div className="card-title">Metrics Evaluation</div>
              <p className="text-muted text-sm" style={{ lineHeight: 1.5 }}>
                Inference computes 8 metrics — PSNR, SSIM, IT-SSIM, SAM, UIQI, RMSE, FSIM, SRER —
                for both the SR output and the bicubic baseline, then reports ΔPSNR and all other
                deltas (SR − LR bicubic) to show the model's improvement over simple upsampling.
              </p>
              <div className="mono" style={{ fontSize: 11, color: 'var(--ink-3)', marginTop: 12, padding: 8, background: 'var(--bg)', borderRadius: 'var(--radius-sm)' }}>
                SR rows · LR rows · Delta rows — averaged across all test images.<br/>
                Results saved to: log_dir/&lt;sr_dir_name&gt;.log
              </div>
            </div>
            {jobId && <LogConsole domain="inference" jobId={jobId} onStop={handleStop} />}
          </div>
        </div>
      </div>
    </div>
  )
}
