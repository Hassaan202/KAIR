import { useState } from 'react'
import { startPipeline3, startRunPipeline, stopPreprocessing } from '../api/client'
import LogConsole from '../components/LogConsole'
import {
  SelectField, TextField, NumberField, BoolToggle,
  ArrayEditor, CollapsibleSection
} from '../components/FormFields'

// ── Pipeline A defaults ────────────────────────────────────────────────────────
const DEFAULT_P3 = {
  hr_image_path: '', lr_image_path: '', output_dir: 'output_patches',
  hr_rgb_bands: [1, 2, 3], lr_rgb_bands: [3, 2, 1],
  scale_factor: 2, hr_patch_size: 256, stride: 256,
  nodata_value: 0, saturated_value: 32767, clip_percentiles: [2.0, 98.0],
  max_nodata_fraction: 0.05, min_variance: 120.0, min_ecc_score: 0.78, min_ssim: 0.60,
  radiometric_block_size: 256, radiometric_rmse_threshold: 35.0,
  radiometric_n_samples: 150000, radiometric_post_hist_match: true,
  coreg_a: { enabled: true, max_features: 8000, match_ratio: 0.75, ransac_thresh: 4.0, downsample: 0.25 },
  coreg_b: { enabled: true, downsample: 0.25, upsample_factor: 100 },
  coreg_c: { enabled: true, max_iter: 100, eps: 1e-5, warp_mode: 'translation', discard_on_fail: true },
  train_test_split: false, train_ratio: 0.8, test_output_dir: '',
}

// ── Pipeline B defaults ────────────────────────────────────────────────────────
const DEFAULT_RP = {
  task: 'preprocess_sr_x2', pipeline_mode: 'hr_only',
  degradation_type: 'satellite', scale: 2, n_channels: 3, seed: 42, num_workers: 1,
  input_hr_dir: '', input_lr_dir: '', output_hr_dir: 'trainsets/hr', output_lr_dir: 'trainsets/lr',
  supported_extensions: ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'],
  save_format: 'png', save_hr_copy: true,
  normalize_enabled: false, normalize_low_percentile: 2.0, normalize_high_percentile: 98.0,
  cloud_mask_enabled: false, cloud_mask_threshold: 0.4, cloud_mask_average_over: 4,
  cloud_mask_dilation_size: 2, cloud_mask_nodata: 0.0, cloud_mask_auto_scale: true,
  bsrgan: { jpeg_prob: 0.9, scale2_prob: 0.25, isp_prob: 0.25, noise_level1: 2, noise_level2: 25 },
  real_esrgan: {
    blur_prob_1: 1.0, resize_prob_1: 1.0, gaussian_noise_prob_1: 0.5,
    poisson_noise_prob_1: 0.1, speckle_noise_prob_1: 0.1, jpeg_prob_1: 0.9,
    noise_level1_s1: 2, noise_level2_s1: 25, blur_prob_2: 0.8, resize_prob_2: 1.0,
    gaussian_noise_prob_2: 0.5, poisson_noise_prob_2: 0.1, speckle_noise_prob_2: 0.1,
    jpeg_prob_2: 0.8, noise_level1_s2: 2, noise_level2_s2: 15,
    final_jpeg_prob: 0.5, resize_back_prob: 0.5, isp_prob: 0.1,
  },
  bsrgan_plus: {
    shuffle_prob: 0.5, use_sharp: false, sharpening_weight: 0.5,
    sharpening_radius: 50, sharpening_threshold: 10, poisson_prob: 0.1,
    speckle_prob: 0.1, isp_prob: 0.1, noise_level1: 2, noise_level2: 25,
  },
  satellite: {
    blur_prob_1: 1.0, blur_type_1: 'mtf', resize_prob_1: 0.75,
    poisson_prob_1: 0.75, read_noise_prob_1: 0.55, haze_prob_1: 0.45, jpeg_prob_1: 0.12,
    blur_prob_2: 0.92, blur_type_2: 'mtf', resize_prob_2: 0.70,
    poisson_prob_2: 0.60, read_noise_prob_2: 0.45, haze_prob_2: 0.35, jpeg_prob_2: 0.08,
    final_jpeg_prob: 0.10, resize_back_prob: 0.35, isp_prob: 0.08,
    noise_level1: 0.8, noise_level2: 5.0,
    mtf_sigma_optics_range: [0.8, 2.8], mtf_detector_width_range: [0.7, 1.8],
    mtf_atm_sigma_range: [0.4, 1.8],
  },
  train_test_split: false, train_ratio: 0.8,
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

// ── Pipeline A Component ─────────────────────────────────────────────────────

function Pipeline3Form({ onJobStart }) {
  const [form, setForm] = useState(DEFAULT_P3)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const set = (path, value) => setForm((prev) => deepSet(prev, path, value))

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      const r = await startPipeline3(form)
      onJobStart(r.data.job_id)
    } catch (err) {
      setError(err.response?.data?.detail || String(err))
    } finally {
      setLoading(false)
    }
  }

  return (
    <form onSubmit={handleSubmit}>
      {/* ── Paths ── */}
      <CollapsibleSection title="Input / Output Paths" defaultOpen>
        <TextField label="HR image path" hint=".JP2 or GeoTIFF"
          value={form.hr_image_path} onChange={(v) => set('hr_image_path', v)}
          placeholder="path/to/HR.JP2" />
        <TextField label="LR image path" hint=".JP2 or GeoTIFF"
          value={form.lr_image_path} onChange={(v) => set('lr_image_path', v)}
          placeholder="path/to/LR.JP2" />
        <TextField label="Output directory"
          value={form.output_dir} onChange={(v) => set('output_dir', v)}
          placeholder="output_patches" />
        <div className="grid-2">
          <ArrayEditor label="HR RGB bands (1-indexed)" value={form.hr_rgb_bands}
            onChange={(v) => set('hr_rgb_bands', v)} />
          <ArrayEditor label="LR RGB bands (1-indexed)" value={form.lr_rgb_bands}
            onChange={(v) => set('lr_rgb_bands', v)} />
        </div>
      </CollapsibleSection>

      {/* ── Patch geometry ── */}
      <CollapsibleSection title="Patch Geometry" defaultOpen>
        <div className="grid-3">
          <SelectField label="Scale factor" value={form.scale_factor}
            onChange={(v) => set('scale_factor', parseInt(v))}
            options={[{ value: 2, label: '×2' }, { value: 4, label: '×4' }]} />
          <NumberField label="HR patch size" value={form.hr_patch_size}
            onChange={(v) => set('hr_patch_size', v)} min={64} step={32}
            hint={`LR = ${form.hr_patch_size / form.scale_factor}`} />
          <NumberField label="Stride" value={form.stride}
            onChange={(v) => set('stride', v)} min={16} step={16} />
        </div>
      </CollapsibleSection>

      {/* ── Quality filters ── */}
      <CollapsibleSection title="Quality Filters" defaultOpen={false}>
        <div className="grid-2">
          <NumberField label="Max nodata fraction" value={form.max_nodata_fraction}
            onChange={(v) => set('max_nodata_fraction', v)} min={0} max={1} step={0.01} />
          <NumberField label="Min variance" value={form.min_variance}
            onChange={(v) => set('min_variance', v)} min={0} step={10} />
        </div>
        <div className="grid-2">
          <NumberField label="Min ECC score" value={form.min_ecc_score}
            onChange={(v) => set('min_ecc_score', v)} min={0} max={1} step={0.01} />
          <NumberField label="Min SSIM" value={form.min_ssim}
            onChange={(v) => set('min_ssim', v)} min={0} max={1} step={0.01} />
        </div>
        <div className="grid-2">
          <NumberField label="Nodata value" value={form.nodata_value}
            onChange={(v) => set('nodata_value', v)} min={0} />
          <NumberField label="Saturated value" value={form.saturated_value}
            onChange={(v) => set('saturated_value', v)} min={1} />
        </div>
      </CollapsibleSection>

      {/* ── Coregistration ── */}
      <CollapsibleSection title="Coregistration" defaultOpen={false}>
        <div className="section-title" style={{ marginTop: 10 }}><h3>Stage A — ORB Keypoint Matching</h3></div>
        <BoolToggle label="Enable Stage A" value={form.coreg_a.enabled}
          onChange={(v) => set('coreg_a.enabled', v)} />
        {form.coreg_a.enabled && (
          <div className="grid-2">
            <NumberField label="Max features" value={form.coreg_a.max_features}
              onChange={(v) => set('coreg_a.max_features', v)} min={100} step={500} />
            <NumberField label="RANSAC thresh" value={form.coreg_a.ransac_thresh}
              onChange={(v) => set('coreg_a.ransac_thresh', v)} min={0.5} step={0.5} />
          </div>
        )}
        <div className="section-title" style={{ marginTop: 16 }}><h3>Stage B — Phase Correlation</h3></div>
        <BoolToggle label="Enable Stage B" value={form.coreg_b.enabled}
          onChange={(v) => set('coreg_b.enabled', v)} />
        <div className="section-title" style={{ marginTop: 16 }}><h3>Stage C — Per-patch ECC Refinement</h3></div>
        <BoolToggle label="Enable Stage C" value={form.coreg_c.enabled}
          onChange={(v) => set('coreg_c.enabled', v)} />
        {form.coreg_c.enabled && (
          <div className="grid-2">
            <NumberField label="Max iterations" value={form.coreg_c.max_iter}
              onChange={(v) => set('coreg_c.max_iter', v)} min={10} step={10} />
            <SelectField label="Warp mode" value={form.coreg_c.warp_mode}
              onChange={(v) => set('coreg_c.warp_mode', v)}
              options={['translation', 'euclidean']} />
          </div>
        )}
      </CollapsibleSection>

      {/* ── Radiometric ── */}
      <CollapsibleSection title="Radiometric Normalisation" defaultOpen={false}>
        <div className="grid-2">
          <NumberField label="RMSE threshold" value={form.radiometric_rmse_threshold}
            onChange={(v) => set('radiometric_rmse_threshold', v)} step={1} />
          <NumberField label="Block size" value={form.radiometric_block_size}
            onChange={(v) => set('radiometric_block_size', v)} min={64} step={64} />
        </div>
        <BoolToggle label="Post histogram matching (correct NIR leakage)"
          value={form.radiometric_post_hist_match}
          onChange={(v) => set('radiometric_post_hist_match', v)} />
      </CollapsibleSection>

      {/* ── Train/test split ── */}
      <CollapsibleSection title="Train / Test Split" defaultOpen={false}>
        <BoolToggle label="Apply train/test split after pipeline completes"
          value={form.train_test_split} onChange={(v) => set('train_test_split', v)} />
        {form.train_test_split && (
          <>
            <NumberField label="Train ratio" value={form.train_ratio}
              onChange={(v) => set('train_ratio', v)} min={0.1} max={0.99} step={0.05} />
            <TextField label="Test output directory (optional)"
              hint="defaults to OUTPUT_DIR_test"
              value={form.test_output_dir} onChange={(v) => set('test_output_dir', v)} />
          </>
        )}
      </CollapsibleSection>

      {error && <div style={{ color: 'var(--bad)', fontSize: 13, marginBottom: 12 }}>{error}</div>}

      <button type="submit" className="btn btn-primary full-width" disabled={loading} style={{ marginTop: 8 }}>
        {loading ? 'Starting…' : '▶ Run Pipeline'}
      </button>
    </form>
  )
}

// ── Pipeline B Component ─────────────────────────────────────────────────────

function RunPipelineForm({ onJobStart }) {
  const [form, setForm] = useState(DEFAULT_RP)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const set = (path, value) => setForm((prev) => deepSet(prev, path, value))

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      const r = await startRunPipeline(form)
      onJobStart(r.data.job_id)
    } catch (err) {
      setError(err.response?.data?.detail || String(err))
    } finally {
      setLoading(false)
    }
  }

  const deg = form.degradation_type

  return (
    <form onSubmit={handleSubmit}>
      <CollapsibleSection title="Pipeline Settings" defaultOpen>
        <TextField label="Task name" value={form.task} onChange={(v) => set('task', v)} />
        <div className="grid-2">
          <SelectField label="Pipeline mode" value={form.pipeline_mode}
            onChange={(v) => set('pipeline_mode', v)}
            options={[
              { value: 'hr_only', label: 'HR only → degrade to LR' },
              { value: 'hr_lr_pair', label: 'HR+LR pair (no degradation)' },
            ]} />
          <SelectField label="Scale" value={form.scale}
            onChange={(v) => set('scale', parseInt(v))}
            options={[{ value: 2, label: '×2' }, { value: 3, label: '×3' }, { value: 4, label: '×4' }]} />
        </div>
        <div className="grid-2">
          <SelectField label="Channels" value={form.n_channels}
            onChange={(v) => set('n_channels', parseInt(v))}
            options={[{ value: 1, label: 'Grayscale (1)' }, { value: 3, label: 'RGB (3)' }]} />
          <NumberField label="Workers" value={form.num_workers}
            onChange={(v) => set('num_workers', v)} min={1} />
        </div>
      </CollapsibleSection>

      <CollapsibleSection title="Paths" defaultOpen>
        <TextField label="Input HR dir" value={form.input_hr_dir}
          onChange={(v) => set('input_hr_dir', v)} />
        {form.pipeline_mode === 'hr_lr_pair' && (
          <TextField label="Input LR dir" value={form.input_lr_dir}
            onChange={(v) => set('input_lr_dir', v)} />
        )}
        <div className="grid-2">
          <TextField label="Output HR dir" value={form.output_hr_dir}
            onChange={(v) => set('output_hr_dir', v)} />
          <TextField label="Output LR dir" value={form.output_lr_dir}
            onChange={(v) => set('output_lr_dir', v)} />
        </div>
        <div className="grid-2">
          <SelectField label="Save format" value={form.save_format}
            onChange={(v) => set('save_format', v)}
            options={['png', 'tif', 'jpg']} />
          <BoolToggle label="Save HR copy" value={form.save_hr_copy}
            onChange={(v) => set('save_hr_copy', v)} />
        </div>
      </CollapsibleSection>

      {form.pipeline_mode === 'hr_only' && (
        <CollapsibleSection title="Degradation" defaultOpen>
          <SelectField label="Degradation type" value={form.degradation_type}
            onChange={(v) => set('degradation_type', v)}
            options={['bsrgan', 'real_esrgan', 'bsrgan_plus', 'satellite']} />

          {deg === 'bsrgan' && (
            <div className="grid-2">
              <NumberField label="JPEG prob" value={form.bsrgan.jpeg_prob}
                onChange={(v) => set('bsrgan.jpeg_prob', v)} min={0} max={1} step={0.05} />
              <NumberField label="ISP prob" value={form.bsrgan.isp_prob}
                onChange={(v) => set('bsrgan.isp_prob', v)} min={0} max={1} step={0.05} />
              <NumberField label="Noise level 1" value={form.bsrgan.noise_level1}
                onChange={(v) => set('bsrgan.noise_level1', v)} min={0} step={1} />
              <NumberField label="Noise level 2" value={form.bsrgan.noise_level2}
                onChange={(v) => set('bsrgan.noise_level2', v)} min={0} step={5} />
            </div>
          )}

          {deg === 'satellite' && (
            <>
              <p style={{ color: 'var(--ink-3)', fontSize: 12, marginBottom: 16 }}>
                Satellite-optimized: MTF/PSF blur, shot noise, read noise, atmospheric haze.
              </p>
              <div className="grid-2">
                <SelectField label="Stage 1 blur type" value={form.satellite.blur_type_1}
                  onChange={(v) => set('satellite.blur_type_1', v)}
                  options={['mtf', 'anisotropic']} />
                <NumberField label="Stage 1 blur prob" value={form.satellite.blur_prob_1}
                  onChange={(v) => set('satellite.blur_prob_1', v)} min={0} max={1} step={0.05} />
              </div>
              <div className="grid-2">
                <NumberField label="Poisson prob (stage 1)" value={form.satellite.poisson_prob_1}
                  onChange={(v) => set('satellite.poisson_prob_1', v)} min={0} max={1} step={0.05} />
                <NumberField label="Haze prob (stage 1)" value={form.satellite.haze_prob_1}
                  onChange={(v) => set('satellite.haze_prob_1', v)} min={0} max={1} step={0.05} />
              </div>
              <div className="grid-2">
                <NumberField label="Noise level min" value={form.satellite.noise_level1}
                  onChange={(v) => set('satellite.noise_level1', v)} min={0} step={0.1} />
                <NumberField label="Noise level max" value={form.satellite.noise_level2}
                  onChange={(v) => set('satellite.noise_level2', v)} min={0} step={0.5} />
              </div>
              <ArrayEditor label="MTF optics sigma range [min, max]"
                value={form.satellite.mtf_sigma_optics_range}
                onChange={(v) => set('satellite.mtf_sigma_optics_range', v)}
                integer={false} />
            </>
          )}

          {deg === 'bsrgan_plus' && (
            <div className="grid-2">
              <NumberField label="Shuffle prob" value={form.bsrgan_plus.shuffle_prob}
                onChange={(v) => set('bsrgan_plus.shuffle_prob', v)} min={0} max={1} step={0.05} />
              <BoolToggle label="Use sharp" value={form.bsrgan_plus.use_sharp}
                onChange={(v) => set('bsrgan_plus.use_sharp', v)} />
            </div>
          )}

          {deg === 'real_esrgan' && (
            <div className="grid-2">
              <NumberField label="Blur prob (stage 1)" value={form.real_esrgan.blur_prob_1}
                onChange={(v) => set('real_esrgan.blur_prob_1', v)} min={0} max={1} step={0.1} />
              <NumberField label="JPEG prob (stage 1)" value={form.real_esrgan.jpeg_prob_1}
                onChange={(v) => set('real_esrgan.jpeg_prob_1', v)} min={0} max={1} step={0.05} />
            </div>
          )}
        </CollapsibleSection>
      )}

      <CollapsibleSection title="Normalisation" defaultOpen={false}>
        <BoolToggle label="Enable percentile normalisation"
          value={form.normalize_enabled} onChange={(v) => set('normalize_enabled', v)} />
        {form.normalize_enabled && (
          <div className="grid-2">
            <NumberField label="Low percentile" value={form.normalize_low_percentile}
              onChange={(v) => set('normalize_low_percentile', v)} min={0} max={49} step={0.5} />
            <NumberField label="High percentile" value={form.normalize_high_percentile}
              onChange={(v) => set('normalize_high_percentile', v)} min={51} max={100} step={0.5} />
          </div>
        )}
      </CollapsibleSection>

      <CollapsibleSection title="Train / Test Split" defaultOpen={false}>
        <BoolToggle label="Split outputs into train/test sets after completion"
          value={form.train_test_split} onChange={(v) => set('train_test_split', v)} />
        {form.train_test_split && (
          <NumberField label="Train ratio" value={form.train_ratio}
            onChange={(v) => set('train_ratio', v)} min={0.1} max={0.99} step={0.05} />
        )}
      </CollapsibleSection>

      {error && <div style={{ color: 'var(--bad)', fontSize: 13, marginBottom: 12 }}>{error}</div>}

      <button type="submit" className="btn btn-primary full-width" disabled={loading} style={{ marginTop: 8 }}>
        {loading ? 'Starting…' : '▶ Run Pipeline'}
      </button>
    </form>
  )
}

// ── Main Preprocessing page ─────────────────────────────────────────────────

export default function Preprocessing() {
  const [tab, setTab] = useState('pipeline3')
  const [jobId, setJobId] = useState(null)

  const handleStop = async () => {
    if (jobId) await stopPreprocessing(jobId).catch(() => { })
  }

  return (
    <div>
      <div className="topbar">
        <div className="topbar-title">
          <h2>Preprocessing</h2>
        </div>
      </div>

      <div className="content">
        <h1 className="editorial rise" style={{ fontSize: 32, marginBottom: 10 }}>Dataset Preparation</h1>
        <p className="rise" style={{ color: 'var(--ink-2)', marginBottom: 30, maxWidth: 600 }}>
          Prepare HR/LR training pairs from raw satellite imagery or degrade existing HR patches.
        </p>

        <div className="module-grid rise" style={{ animationDelay: '100ms' }}>
          <div className="col">
            <div className="mode-tabs">
              <button className={`mode-tab ${tab === 'pipeline3' ? 'active' : ''}`}
                onClick={() => setTab('pipeline3')}>
                Pipeline A — Pleiades
              </button>
              <button className={`mode-tab ${tab === 'run_pipeline' ? 'active' : ''}`}
                onClick={() => setTab('run_pipeline')}>
                Pipeline B — HR Degradation
              </button>
            </div>

            {tab === 'pipeline3' && (
              <div className="card animate-in" style={{ padding: 0, overflow: 'hidden', border: 'none', boxShadow: 'none', background: 'transparent' }}>
                <div style={{ marginBottom: 16 }}>
                  <span style={{ fontSize: 15, fontWeight: 600 }}>Pleiades / Multi-sensor Patch Extraction</span>
                </div>
                <p style={{ color: 'var(--ink-3)', fontSize: 12.5, marginBottom: 20, lineHeight: 1.5 }}>
                  Takes paired HR + LR GeoTIFF or JP2 satellite images, performs 3-stage
                  coregistration (ORB → Phase Correlation → ECC), radiometric regression,
                  and extracts quality-filtered patch pairs.
                </p>
                <Pipeline3Form onJobStart={setJobId} />
              </div>
            )}

            {tab === 'run_pipeline' && (
              <div className="card animate-in" style={{ padding: 0, overflow: 'hidden', border: 'none', boxShadow: 'none', background: 'transparent' }}>
                <div style={{ marginBottom: 16 }}>
                  <span style={{ fontSize: 15, fontWeight: 600 }}>HR-only / HR+LR Pair Preprocessing</span>
                </div>
                <p style={{ color: 'var(--ink-3)', fontSize: 12.5, marginBottom: 20, lineHeight: 1.5 }}>
                  Processes HR images with optional normalisation and cloud masking.
                  In HR-only mode, generates synthetic LR via BSRGAN, Real-ESRGAN, or
                  satellite-optimized degradation. In HR+LR pair mode, preprocesses both
                  HR and LR without degradation.
                </p>
                <RunPipelineForm onJobStart={setJobId} />
              </div>
            )}
          </div>

          <div className="col">
            {jobId && <LogConsole domain="preprocessing" jobId={jobId} onStop={handleStop} />}
          </div>
        </div>
      </div>
    </div>
  )
}
