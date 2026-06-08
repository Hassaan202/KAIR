"use client";
import { useState, useCallback, useRef, useEffect } from "react";
import Accordion from "@/components/design/Accordion";
import LogStream from "@/components/design/LogStream";
import {
  Slider, NumberField, TextInput, Select, Segmented, Toggle, CardToggle
} from "@/components/design/Primitives";
import { Icons } from "@/components/design/Primitives";
import FieldScene from "@/components/design/FieldScene";
import { DEFAULT_SIMPLE_PREPROCESS, DEFAULT_COMPLETE_PIPELINE } from "@/lib/defaults";
import type {
  SimplePreprocessConfig, CompletePipelineConfig,
  DegradationType, PipelineVariant, LogLine, SaveFormat,
} from "@/lib/types";
import { runSimplePipeline, runCompletePipeline, streamLogs } from "@/lib/api";
import { useAppStore } from "@/lib/store";

// ---- Dataset options from the repo ----
const DATASETS = [
  { id: "uc_airbus", name: "UC Airbus (synth)", sub: "paired · 256px · mixed", tag: "paired", rec: true },
  { id: "sen2venus", name: "SEN2VENµS",         sub: "29,241 tiles · 10→2m", tag: "paired" },
  { id: "nwpu",      name: "NWPU-RESISC45",     sub: "31,500 scenes · 256px", tag: "scene" },
  { id: "ucmerced",  name: "UC Merced",         sub: "2,100 tiles · 256px · 21 classes", tag: "scene" },
  { id: "custom",    name: "Custom path",       sub: "specify your own dirs", tag: "custom" },
];

const REGIONS: Record<string, string> = {
  uc_airbus: "urban", sen2venus: "delta", nwpu: "farmland", ucmerced: "farmland", custom: "mountain",
};

const DATASET_PATHS: Record<string, { hr: string; lr: string }> = {
  uc_airbus: { hr: "trainsets/uc_airbus_both_synth/hr", lr: "trainsets/uc_airbus_both_synth/lr" },
  sen2venus:  { hr: "trainsets/Sen2Venus/HR",            lr: "trainsets/Sen2Venus/LR_x4" },
  nwpu:       { hr: "trainsets/NWPU_RESIC45",            lr: "" },
  ucmerced:   { hr: "trainsets/UCMerced_LandUse/HR",     lr: "trainsets/UCMerced_LandUse/LR_flat_x2" },
  custom:     { hr: "", lr: "" },
};

// Degradation descriptions
const DEGRADATION_CARDS: { id: DegradationType; name: string; sub: string }[] = [
  { id: "bsrgan",      name: "BSRGAN",        sub: "realistic blur + JPEG + noise" },
  { id: "real_esrgan", name: "Real-ESRGAN",    sub: "2-stage degradation pipeline" },
  { id: "bsrgan_plus", name: "BSRGAN-Plus",    sub: "shuffle + sharpening + extended" },
  { id: "satellite",   name: "Satellite",      sub: "MTF + haze + shot noise (recommended)" },
];

// Optuna metric names (exactly from best_degradation.json)
const OPT_METRICS = ["PSNR", "SSIM", "SAM", "FSIM", "RMSE", "UIQI", "IT-SSIM", "SRER"] as const;
const DEFAULT_WEIGHTS: Record<string, number> = {
  PSNR: 20, SSIM: 20, SAM: 15, UIQI: 10, FSIM: 15, RMSE: 10, "IT-SSIM": 5, SRER: 5,
};

export default function PreprocessModule() {
  const [variant, setVariant] = useState<PipelineVariant>("simple");
  const [source, setSource] = useState<"preloaded" | "custom">("preloaded");
  const [dataset, setDataset] = useState("uc_airbus");

  // Simple pipeline config (from preprocessing_pipeline/config.json)
  const [cfg, setCfg] = useState<SimplePreprocessConfig>(DEFAULT_SIMPLE_PREPROCESS);
  // Complete pipeline config (from preprocessing_pipeline/config_l2.json)
  const [completeCfg, setCompleteCfg] = useState<CompletePipelineConfig>(DEFAULT_COMPLETE_PIPELINE);

  // Optuna state
  const [optuna, setOptuna] = useState(true);
  const [trials, setTrials] = useState(40);
  const [weights, setWeights] = useState<Record<string, number>>(DEFAULT_WEIGHTS);

  // Job / log state
  const [logs, setLogs] = useState<LogLine[]>([]);
  const [running, setRunning] = useState(false);
  const [done, setDone] = useState(false);
  const [progress, setProgress] = useState(0);
  const stopStream = useRef<(() => void) | null>(null);

  const { setActiveJob, setRunStatus } = useAppStore();

  const set = <K extends keyof SimplePreprocessConfig>(k: K, v: SimplePreprocessConfig[K]) =>
    setCfg((c: SimplePreprocessConfig) => ({ ...c, [k]: v }));

  // Sync dataset selection to paths
  useEffect(() => {
    if (source === "preloaded" && dataset !== "custom") {
      const paths = DATASET_PATHS[dataset] || { hr: "", lr: "" };
      setCfg((c: SimplePreprocessConfig) => ({ ...c, input_hr_dir: paths.hr, input_lr_dir: paths.lr || null }));
    }
  }, [dataset, source]);

  const weightTotal = (Object.values(weights) as number[]).reduce((a, b) => a + b, 0);

  const normalizeWeights = () => {
    const sum = weightTotal || 1;
    const next: Record<string, number> = {};
    OPT_METRICS.forEach((m) => next[m] = Math.round((weights[m] / sum) * 100));
    setWeights(next);
  };

  const handleRun = async () => {
    setRunning(true); setDone(false); setProgress(0); setLogs([]);
    setRunStatus("running");
    const dsName = DATASETS.find((d) => d.id === dataset)?.name ?? dataset;
    setActiveJob("pending", `preprocess · ${dsName}`, `${variant} pipeline`);

    try {
      const res = variant === "simple"
        ? await runSimplePipeline(cfg)
        : await runCompletePipeline(completeCfg);

      stopStream.current = streamLogs(
        res.job_id,
        (line: LogLine) => {
          setLogs((l: LogLine[]) => [...l, line]);
          const m = line.text.match(/(\d+)\/(\d+)/);
          if (m) setProgress(Math.round((parseInt(m[1]) / parseInt(m[2])) * 100));
        },
        () => {
          setRunning(false); setDone(true); setProgress(100);
          setRunStatus("done"); setActiveJob(null);
        },
      );
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setLogs((l: LogLine[]) => [...l, { ts: new Date().toTimeString().slice(0, 8), text: `Error: ${msg}`, lv: "warn" as const }]);
      setRunning(false); setRunStatus("failed"); setActiveJob(null);
    }
  };

  const handleStop = () => {
    stopStream.current?.();
    setRunning(false); setRunStatus("idle"); setActiveJob(null);
  };

  // Complete pipeline step toggle helper
  const setCC = <K extends keyof CompletePipelineConfig>(k: K, v: CompletePipelineConfig[K]) =>
    setCompleteCfg((c: CompletePipelineConfig) => ({ ...c, [k]: v }));

  return (
    <div className="content">
      {/* Pipeline variant selector */}
      <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 22 }}>
        <div className="section-title" style={{ margin: 0 }}><h3>Pipeline type</h3></div>
        <Segmented
          options={["simple", "complete"]}
          value={variant}
          onChange={(v) => setVariant(v as PipelineVariant)}
        />
        <span className="mono" style={{ fontSize: 11, color: "var(--ink-3)" }}>
          {variant === "simple" ? "run_pipeline.py · HR-only or paired" : "complete_pipeline.py · 6-stage satellite"}
        </span>
      </div>

      <div className="module-grid">
        {/* ===== LEFT: config ===== */}
        <div className="col">

          {/* 01 — Dataset source */}
          <Accordion title="Dataset source" sub="01"
            right={<span className="chip">{DATASETS.find((d) => d.id === dataset)?.name}</span>}>
            <div style={{ display: "flex", gap: 8, marginBottom: 14 }}>
              {(["preloaded", "custom"] as const).map((k) => (
                <button key={k} className={"src-tab" + (source === k ? " on" : "")} onClick={() => setSource(k)}>
                  {k === "preloaded" ? "Preloaded" : "Custom path"}
                </button>
              ))}
            </div>

            {source === "preloaded" && (
              <div style={{ display: "grid", gap: 8 }}>
                {DATASETS.filter((d) => d.id !== "custom").map((d) => (
                  <button
                    key={d.id}
                    className={"ds-row" + (dataset === d.id ? " on" : "")}
                    onClick={() => setDataset(d.id)}
                  >
                    <div className="ds-mini">
                      <FieldScene seed={d.id.charCodeAt(0) * 3} region={REGIONS[d.id] || "farmland"} detail="hi" />
                    </div>
                    <div style={{ flex: 1, textAlign: "left" }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                        <strong style={{ fontSize: 13.5 }}>{d.name}</strong>
                        {d.rec && <span className="chip" style={{ fontSize: 10, padding: "1px 7px", color: "var(--cobalt-deep)", borderColor: "var(--cobalt-soft)" }}>recommended</span>}
                      </div>
                      <span className="mono" style={{ fontSize: 11, color: "var(--ink-3)" }}>{d.sub}</span>
                    </div>
                    <span className="chip" style={{ fontSize: 10 }}>{d.tag}</span>
                  </button>
                ))}
              </div>
            )}

            {source === "custom" && (
              <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                <TextInput label="Input HR directory" value={cfg.input_hr_dir}
                  onChange={(v) => set("input_hr_dir", v)} mono />
                {cfg.pipeline_mode === "hr_lr_pair" && (
                  <TextInput label="Input LR directory" value={cfg.input_lr_dir ?? ""}
                    onChange={(v) => set("input_lr_dir", v || null)} mono />
                )}
                <TextInput label="Output HR directory" value={cfg.output_hr_dir}
                  onChange={(v) => set("output_hr_dir", v)} mono />
                <TextInput label="Output LR directory" value={cfg.output_lr_dir}
                  onChange={(v) => set("output_lr_dir", v)} mono />
              </div>
            )}
          </Accordion>

          {/* 02 — Pipeline settings */}
          <Accordion title="Pipeline settings" sub="02" right={<span className="chip">×{cfg.scale}</span>}>
            <div className="grid-2" style={{ marginBottom: 16 }}>
              <div>
                <div className="field-label"><span>Pipeline mode</span></div>
                <Segmented
                  options={["hr_only", "hr_lr_pair"]}
                  value={cfg.pipeline_mode}
                  onChange={(v) => set("pipeline_mode", v as SimplePreprocessConfig["pipeline_mode"])}
                />
              </div>
              <div>
                <div className="field-label"><span>Scale factor</span></div>
                <Segmented options={["×2", "×3", "×4", "×8"]}
                  value={"×" + cfg.scale}
                  onChange={(v) => set("scale", parseInt(v.slice(1)) as SimplePreprocessConfig["scale"])} />
              </div>
              <div>
                <div className="field-label"><span>Channels</span></div>
                <Segmented options={["1 (gray)", "3 (RGB)"]}
                  value={cfg.n_channels === 1 ? "1 (gray)" : "3 (RGB)"}
                  onChange={(v) => set("n_channels", v.startsWith("1") ? 1 : 3)} />
              </div>
              <NumberField label="Workers" value={cfg.num_workers}
                onChange={(v) => set("num_workers", v)} />
            </div>

            <div className="grid-2">
              <div>
                <div className="field-label"><span>Save format</span></div>
                <Segmented options={["png", "tif", "jpg"]}
                  value={cfg.save_format}
                  onChange={(v) => set("save_format", v as SaveFormat)} />
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 10, paddingTop: 24 }}>
                <Toggle checked={cfg.save_hr_copy} onChange={(v) => set("save_hr_copy", v)} />
                <span style={{ fontSize: 13, color: "var(--ink-2)" }}>Save HR copy</span>
              </div>
            </div>
          </Accordion>

          {/* 03 — Normalization & cloud masking */}
          <Accordion title="Preprocessing" sub="03">
            <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
              {/* Normalization */}
              <div>
                <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
                  <Toggle checked={cfg.normalize_enabled} onChange={(v) => set("normalize_enabled", v)} />
                  <span style={{ fontSize: 13, fontWeight: 500 }}>Percentile normalization</span>
                </div>
                {cfg.normalize_enabled && (
                  <div className="grid-2">
                    <Slider label="Low percentile" value={cfg.normalize_low_percentile}
                      min={0} max={20} step={0.5} unit="%" onChange={(v) => set("normalize_low_percentile", v)} />
                    <Slider label="High percentile" value={cfg.normalize_high_percentile}
                      min={80} max={100} step={0.5} unit="%" onChange={(v) => set("normalize_high_percentile", v)} />
                  </div>
                )}
              </div>

              {/* Cloud masking (Sentinel-2 only) */}
              <div>
                <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
                  <Toggle checked={cfg.cloud_mask_enabled} onChange={(v) => set("cloud_mask_enabled", v)} />
                  <span style={{ fontSize: 13, fontWeight: 500 }}>Sentinel-2 cloud masking</span>
                  <span className="chip" style={{ fontSize: 10 }}>10-band S2 only</span>
                </div>
                {cfg.cloud_mask_enabled && (
                  <div className="grid-2">
                    <Slider label="Threshold" value={cfg.cloud_mask_threshold}
                      min={0} max={1} step={0.05} onChange={(v) => set("cloud_mask_threshold", v)}
                      fmt={(v) => v.toFixed(2)} />
                    <NumberField label="Average over" value={cfg.cloud_mask_average_over}
                      onChange={(v) => set("cloud_mask_average_over", v)} />
                    <NumberField label="Dilation size" value={cfg.cloud_mask_dilation_size}
                      onChange={(v) => set("cloud_mask_dilation_size", v)} />
                    <div style={{ display: "flex", alignItems: "center", gap: 10, paddingTop: 24 }}>
                      <Toggle checked={cfg.cloud_mask_auto_scale} onChange={(v) => set("cloud_mask_auto_scale", v)} />
                      <span style={{ fontSize: 12, color: "var(--ink-2)" }}>Auto-scale ÷10000</span>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </Accordion>

          {/* 04 — Degradation model */}
          <Accordion title="Degradation model" sub="04"
            right={<span className="chip">{cfg.degradation_type}</span>}>
            <div className="grid-4">
              {DEGRADATION_CARDS.map((d) => (
                <CardToggle
                  key={d.id}
                  on={cfg.degradation_type === d.id}
                  onClick={() => set("degradation_type", d.id)}
                  title={d.name}
                  sub={d.sub}
                />
              ))}
            </div>

            {/* Degradation params — show only for active type */}
            <div style={{ marginTop: 18 }}>
              {cfg.degradation_type === "bsrgan" && (
                <>
                  <div className="section-title" style={{ marginBottom: 12 }}><h3 style={{ fontSize: 13 }}>BSRGAN parameters</h3></div>
                  <div className="grid-2">
                    <Slider label="JPEG probability" value={cfg.bsrgan.jpeg_prob} min={0} max={1} step={0.05}
                      onChange={(v) => setCfg((c) => ({ ...c, bsrgan: { ...c.bsrgan, jpeg_prob: v } }))} fmt={(v) => v.toFixed(2)} />
                    <Slider label="Scale2 probability" value={cfg.bsrgan.scale2_prob} min={0} max={1} step={0.05}
                      onChange={(v) => setCfg((c) => ({ ...c, bsrgan: { ...c.bsrgan, scale2_prob: v } }))} fmt={(v) => v.toFixed(2)} />
                    <Slider label="ISP probability" value={cfg.bsrgan.isp_prob} min={0} max={1} step={0.05}
                      onChange={(v) => setCfg((c) => ({ ...c, bsrgan: { ...c.bsrgan, isp_prob: v } }))} fmt={(v) => v.toFixed(2)} />
                    <Slider label="Noise level min" value={cfg.bsrgan.noise_level1} min={0} max={50} step={1}
                      onChange={(v) => setCfg((c) => ({ ...c, bsrgan: { ...c.bsrgan, noise_level1: v } }))} />
                    <Slider label="Noise level max" value={cfg.bsrgan.noise_level2} min={0} max={100} step={1}
                      onChange={(v) => setCfg((c) => ({ ...c, bsrgan: { ...c.bsrgan, noise_level2: v } }))} />
                  </div>
                </>
              )}

              {cfg.degradation_type === "satellite" && (
                <>
                  <div className="section-title" style={{ marginBottom: 12 }}><h3 style={{ fontSize: 13 }}>Satellite parameters — Stage 1 (sensor)</h3></div>
                  <div className="grid-2">
                    <Slider label="Blur probability" value={cfg.satellite.blur_prob_1} min={0} max={1} step={0.05}
                      onChange={(v) => setCfg((c) => ({ ...c, satellite: { ...c.satellite, blur_prob_1: v } }))} fmt={(v) => v.toFixed(2)} />
                    <div>
                      <div className="field-label"><span>Blur type</span></div>
                      <Segmented options={["mtf", "anisotropic"]} value={cfg.satellite.blur_type_1}
                        onChange={(v) => setCfg((c) => ({ ...c, satellite: { ...c.satellite, blur_type_1: v as "mtf" | "anisotropic" } }))} />
                    </div>
                    <Slider label="Poisson (shot) prob" value={cfg.satellite.poisson_prob_1} min={0} max={1} step={0.05}
                      onChange={(v) => setCfg((c) => ({ ...c, satellite: { ...c.satellite, poisson_prob_1: v } }))} fmt={(v) => v.toFixed(2)} />
                    <Slider label="Read noise prob" value={cfg.satellite.read_noise_prob_1} min={0} max={1} step={0.05}
                      onChange={(v) => setCfg((c) => ({ ...c, satellite: { ...c.satellite, read_noise_prob_1: v } }))} fmt={(v) => v.toFixed(2)} />
                    <Slider label="Haze probability" value={cfg.satellite.haze_prob_1} min={0} max={1} step={0.05}
                      onChange={(v) => setCfg((c) => ({ ...c, satellite: { ...c.satellite, haze_prob_1: v } }))} fmt={(v) => v.toFixed(2)} />
                  </div>
                  <div className="section-title" style={{ marginBottom: 12, marginTop: 16 }}><h3 style={{ fontSize: 13 }}>MTF ranges</h3></div>
                  <div className="grid-3">
                    <div>
                      <div className="field-label"><span>Optics σ range</span></div>
                      <div style={{ display: "flex", gap: 8 }}>
                        <input className="num-input" type="number" value={cfg.satellite.mtf_sigma_optics_range[0]} step={0.1}
                          onChange={(e) => setCfg((c) => ({ ...c, satellite: { ...c.satellite, mtf_sigma_optics_range: [parseFloat(e.target.value), c.satellite.mtf_sigma_optics_range[1]] } }))} />
                        <input className="num-input" type="number" value={cfg.satellite.mtf_sigma_optics_range[1]} step={0.1}
                          onChange={(e) => setCfg((c) => ({ ...c, satellite: { ...c.satellite, mtf_sigma_optics_range: [c.satellite.mtf_sigma_optics_range[0], parseFloat(e.target.value)] } }))} />
                      </div>
                    </div>
                    <div>
                      <div className="field-label"><span>Detector width range</span></div>
                      <div style={{ display: "flex", gap: 8 }}>
                        <input className="num-input" type="number" value={cfg.satellite.mtf_detector_width_range[0]} step={0.1}
                          onChange={(e) => setCfg((c) => ({ ...c, satellite: { ...c.satellite, mtf_detector_width_range: [parseFloat(e.target.value), c.satellite.mtf_detector_width_range[1]] } }))} />
                        <input className="num-input" type="number" value={cfg.satellite.mtf_detector_width_range[1]} step={0.1}
                          onChange={(e) => setCfg((c) => ({ ...c, satellite: { ...c.satellite, mtf_detector_width_range: [c.satellite.mtf_detector_width_range[0], parseFloat(e.target.value)] } }))} />
                      </div>
                    </div>
                    <div>
                      <div className="field-label"><span>Atm. turbulence range</span></div>
                      <div style={{ display: "flex", gap: 8 }}>
                        <input className="num-input" type="number" value={cfg.satellite.mtf_atm_sigma_range[0]} step={0.1}
                          onChange={(e) => setCfg((c) => ({ ...c, satellite: { ...c.satellite, mtf_atm_sigma_range: [parseFloat(e.target.value), c.satellite.mtf_atm_sigma_range[1]] } }))} />
                        <input className="num-input" type="number" value={cfg.satellite.mtf_atm_sigma_range[1]} step={0.1}
                          onChange={(e) => setCfg((c) => ({ ...c, satellite: { ...c.satellite, mtf_atm_sigma_range: [c.satellite.mtf_atm_sigma_range[0], parseFloat(e.target.value)] } }))} />
                      </div>
                    </div>
                  </div>
                </>
              )}

              {cfg.degradation_type === "real_esrgan" && (
                <>
                  <div className="section-title" style={{ marginBottom: 12 }}><h3 style={{ fontSize: 13 }}>Real-ESRGAN — Stage 1</h3></div>
                  <div className="grid-2">
                    {(["blur_prob_1","resize_prob_1","gaussian_noise_prob_1","poisson_noise_prob_1","speckle_noise_prob_1","jpeg_prob_1"] as const).map((k) => (
                      <Slider key={k} label={k.replace(/_1$/, "").replace(/_/g, " ")}
                        value={(cfg.real_esrgan as unknown as Record<string, number>)[k] as number}
                        min={0} max={1} step={0.05}
                        onChange={(v) => setCfg((c) => ({ ...c, real_esrgan: { ...c.real_esrgan, [k]: v } }))}
                        fmt={(v) => v.toFixed(2)} />
                    ))}
                  </div>
                  <div className="section-title" style={{ marginBottom: 12, marginTop: 16 }}><h3 style={{ fontSize: 13 }}>Stage 2 & final</h3></div>
                  <div className="grid-2">
                    {(["blur_prob_2","resize_prob_2","gaussian_noise_prob_2","jpeg_prob_2","final_jpeg_prob","resize_back_prob","isp_prob"] as const).map((k) => (
                      <Slider key={k} label={k.replace(/_2$/, "").replace(/_/g, " ")}
                        value={(cfg.real_esrgan as unknown as Record<string, number>)[k] as number}
                        min={0} max={1} step={0.05}
                        onChange={(v) => setCfg((c) => ({ ...c, real_esrgan: { ...c.real_esrgan, [k]: v } }))}
                        fmt={(v) => v.toFixed(2)} />
                    ))}
                  </div>
                </>
              )}

              {cfg.degradation_type === "bsrgan_plus" && (
                <>
                  <div className="section-title" style={{ marginBottom: 12 }}><h3 style={{ fontSize: 13 }}>BSRGAN-Plus parameters</h3></div>
                  <div className="grid-2">
                    <Slider label="Shuffle probability" value={cfg.bsrgan_plus.shuffle_prob} min={0} max={1} step={0.05}
                      onChange={(v) => setCfg((c) => ({ ...c, bsrgan_plus: { ...c.bsrgan_plus, shuffle_prob: v } }))} fmt={(v) => v.toFixed(2)} />
                    <div style={{ display: "flex", alignItems: "center", gap: 10, paddingTop: 24 }}>
                      <Toggle checked={cfg.bsrgan_plus.use_sharp}
                        onChange={(v) => setCfg((c) => ({ ...c, bsrgan_plus: { ...c.bsrgan_plus, use_sharp: v } }))} />
                      <span style={{ fontSize: 13 }}>USM sharpening</span>
                    </div>
                    {cfg.bsrgan_plus.use_sharp && <>
                      <Slider label="Sharpening weight" value={cfg.bsrgan_plus.sharpening_weight} min={0} max={1} step={0.05}
                        onChange={(v) => setCfg((c) => ({ ...c, bsrgan_plus: { ...c.bsrgan_plus, sharpening_weight: v } }))} fmt={(v) => v.toFixed(2)} />
                      <NumberField label="Sharpening radius" value={cfg.bsrgan_plus.sharpening_radius}
                        onChange={(v) => setCfg((c) => ({ ...c, bsrgan_plus: { ...c.bsrgan_plus, sharpening_radius: v } }))} />
                    </>}
                    <Slider label="Poisson probability" value={cfg.bsrgan_plus.poisson_prob} min={0} max={1} step={0.05}
                      onChange={(v) => setCfg((c) => ({ ...c, bsrgan_plus: { ...c.bsrgan_plus, poisson_prob: v } }))} fmt={(v) => v.toFixed(2)} />
                    <Slider label="Speckle probability" value={cfg.bsrgan_plus.speckle_prob} min={0} max={1} step={0.05}
                      onChange={(v) => setCfg((c) => ({ ...c, bsrgan_plus: { ...c.bsrgan_plus, speckle_prob: v } }))} fmt={(v) => v.toFixed(2)} />
                  </div>
                </>
              )}
            </div>
          </Accordion>

          {/* 05 — Optuna degradation optimizer */}
          <Accordion title="Degradation optimizer (Optuna)" sub="05" open
            right={<Toggle checked={optuna} onChange={setOptuna} />}>
            {optuna ? (
              <>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
                  <span className="mono" style={{ fontSize: 12, color: "var(--ink-2)" }}>
                    TPE search across {OPT_METRICS.length} metrics
                  </span>
                  <div style={{ width: 160 }}>
                    <Slider label="Trials" value={trials} min={10} max={200} step={5} onChange={setTrials} />
                  </div>
                </div>
                <div className="section-title">
                  <h3 style={{ fontSize: 13 }}>Objective weights</h3>
                  <span className="mono" style={{ color: weightTotal === 100 ? "var(--ok)" : "var(--warn)" }}>Σ {weightTotal}%</span>
                  <button className="btn btn-ghost" style={{ marginLeft: "auto", padding: "3px 9px", fontSize: 11 }}
                    onClick={normalizeWeights}>normalise →100</button>
                </div>
                <div className="grid-2" style={{ marginTop: 10 }}>
                  {OPT_METRICS.map((m) => (
                    <Slider key={m} label={m} value={weights[m]} min={0} max={40} unit="%"
                      onChange={(v) => setWeights((w) => ({ ...w, [m]: v }))} />
                  ))}
                </div>
              </>
            ) : (
              <div className="mono" style={{ fontSize: 12, color: "var(--ink-3)", padding: "6px 0" }}>
                Optimizer off — using fixed degradation parameters from above.
              </div>
            )}
          </Accordion>

          {/* 06 — Complete pipeline stages (shown only when complete variant selected) */}
          {variant === "complete" && (
            <>
              <Accordion title="Stage 1 — Cloud / Shadow masking" sub="06"
                right={<Toggle checked={completeCfg.masking.enabled}
                  onChange={(v) => setCC("masking", { ...completeCfg.masking, enabled: v })} />}>
                {completeCfg.masking.enabled && (
                  <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                    <div>
                      <div className="field-label"><span>Method</span></div>
                      <Segmented options={["qa_band", "s2cloudless"]} value={completeCfg.masking.method}
                        onChange={(v) => setCC("masking", { ...completeCfg.masking, method: v as "qa_band" | "s2cloudless" })} />
                    </div>
                    {completeCfg.masking.method === "qa_band" && (
                      <div>
                        <div className="field-label">
                          <span>Invalid SCL classes</span>
                          <span className="val" style={{ fontSize: 10 }}>0=NoData 1=Sat 3=Shadow 8-10=Cloud 11=Snow</span>
                        </div>
                        <input className="text-input" value={completeCfg.masking.invalid_classes.join(", ")}
                          onChange={(e) => setCC("masking", { ...completeCfg.masking, invalid_classes: e.target.value.split(",").map((x) => parseInt(x.trim())).filter((x) => !isNaN(x)) })} />
                      </div>
                    )}
                    {completeCfg.masking.method === "s2cloudless" && (
                      <div className="grid-2">
                        <Slider label="Cloud threshold" value={completeCfg.masking.s2_threshold} min={0} max={1} step={0.05}
                          onChange={(v) => setCC("masking", { ...completeCfg.masking, s2_threshold: v })} fmt={(v) => v.toFixed(2)} />
                        <NumberField label="Average over" value={completeCfg.masking.s2_average_over}
                          onChange={(v) => setCC("masking", { ...completeCfg.masking, s2_average_over: v })} />
                        <NumberField label="Dilation size" value={completeCfg.masking.s2_dilation_size}
                          onChange={(v) => setCC("masking", { ...completeCfg.masking, s2_dilation_size: v })} />
                      </div>
                    )}
                  </div>
                )}
              </Accordion>

              <Accordion title="Stage 2 — Relative normalization" sub="07"
                right={<Toggle checked={completeCfg.relative_normalization.enabled}
                  onChange={(v) => setCC("relative_normalization", { ...completeCfg.relative_normalization, enabled: v })} />}>
                {completeCfg.relative_normalization.enabled && (
                  <div className="grid-2">
                    <div>
                      <div className="field-label"><span>Method</span></div>
                      <Segmented options={["histogram_match", "mean_std_transfer"]}
                        value={completeCfg.relative_normalization.method}
                        onChange={(v) => setCC("relative_normalization", { ...completeCfg.relative_normalization, method: v as "histogram_match" | "mean_std_transfer" })} />
                    </div>
                    <div>
                      <div className="field-label"><span>Direction</span></div>
                      <Segmented options={["lr_to_hr", "hr_to_lr"]}
                        value={completeCfg.relative_normalization.direction}
                        onChange={(v) => setCC("relative_normalization", { ...completeCfg.relative_normalization, direction: v as "lr_to_hr" | "hr_to_lr" })} />
                    </div>
                  </div>
                )}
              </Accordion>

              <Accordion title="Stage 4 — Co-registration (ECC)" sub="08"
                right={<Toggle checked={completeCfg.registration.enabled}
                  onChange={(v) => setCC("registration", { ...completeCfg.registration, enabled: v })} />}>
                {completeCfg.registration.enabled && (
                  <div className="grid-2">
                    <div>
                      <div className="field-label"><span>Warp mode</span></div>
                      <Segmented options={["translation", "euclidean", "affine", "homography"]}
                        value={completeCfg.registration.warp_mode}
                        onChange={(v) => setCC("registration", { ...completeCfg.registration, warp_mode: v as "translation" | "euclidean" | "affine" | "homography" })} />
                    </div>
                    <NumberField label="Max iterations" value={completeCfg.registration.num_iters}
                      onChange={(v) => setCC("registration", { ...completeCfg.registration, num_iters: v })} />
                    <NumberField label="Gaussian filter size (odd)" value={completeCfg.registration.gauss_filt_size}
                      onChange={(v) => setCC("registration", { ...completeCfg.registration, gauss_filt_size: v })} />
                    <div style={{ display: "flex", alignItems: "center", gap: 10, paddingTop: 24 }}>
                      <Toggle checked={completeCfg.registration.skip_on_failure}
                        onChange={(v) => setCC("registration", { ...completeCfg.registration, skip_on_failure: v })} />
                      <span style={{ fontSize: 13 }}>Skip on ECC failure</span>
                    </div>
                  </div>
                )}
              </Accordion>

              <Accordion title="Stage 6 — Mask-aware tiling" sub="09"
                right={<Toggle checked={completeCfg.tiling.enabled}
                  onChange={(v) => setCC("tiling", { ...completeCfg.tiling, enabled: v })} />}>
                {completeCfg.tiling.enabled && (
                  <div className="grid-2">
                    <Slider label="Crop size (HR px)" value={completeCfg.tiling.crop_size}
                      min={64} max={512} step={32}
                      onChange={(v) => setCC("tiling", { ...completeCfg.tiling, crop_size: v })} />
                    <Slider label="Stride" value={completeCfg.tiling.step}
                      min={16} max={512} step={16}
                      onChange={(v) => setCC("tiling", { ...completeCfg.tiling, step: v })} />
                    <Slider label="Max invalid ratio" value={completeCfg.tiling.max_invalid_ratio}
                      min={0} max={1} step={0.05}
                      onChange={(v) => setCC("tiling", { ...completeCfg.tiling, max_invalid_ratio: v })}
                      fmt={(v) => v.toFixed(2)} />
                    <div>
                      <div className="field-label"><span>Tile save format</span></div>
                      <Segmented options={["png", "tif", "jpg"]} value={completeCfg.tiling.save_format}
                        onChange={(v) => setCC("tiling", { ...completeCfg.tiling, save_format: v as "png" | "tif" | "jpg" })} />
                    </div>
                  </div>
                )}
              </Accordion>
            </>
          )}
        </div>

        {/* ===== RIGHT: run panel + output ===== */}
        <div className="col" style={{ position: "sticky", top: 88 }}>
          <div className="card" style={{ padding: 18 }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 14 }}>
              <div className="section-title" style={{ margin: 0 }}><h3>Run pipeline</h3></div>
              <span className="mono" style={{ fontSize: 11, color: running ? "var(--cobalt)" : "var(--ink-3)" }}>
                {running ? progress + "%" : "ready"}
              </span>
            </div>

            {running ? (
              <button className="btn" style={{ width: "100%", justifyContent: "center", borderColor: "var(--warn)", color: "var(--warn)" }}
                onClick={handleStop}>
                <Icons.stop size={13} /> Stop
              </button>
            ) : (
              <button className="btn btn-primary" style={{ width: "100%", justifyContent: "center" }}
                onClick={handleRun}>
                <Icons.play size={13} /> Run {variant} pipeline
              </button>
            )}

            <div style={{ marginTop: 14 }}>
              <LogStream
                lines={logs.length ? logs : [{ ts: "--:--:--", text: "log stream idle — press run to start", lv: "info" }]}
                running={running}
                height={220}
              />
            </div>
          </div>

          <div className="card" style={{ padding: 18 }}>
            <div className="section-title"><h3>Output summary</h3><span className="mono">{done ? "verified" : "—"}</span></div>
            <div className="kv" style={{ marginTop: 12 }}>
              <span className="k">HR output dir</span><span className="v">{cfg.output_hr_dir || "—"}</span>
              <span className="k">LR output dir</span><span className="v">{cfg.output_lr_dir || "—"}</span>
              <span className="k">Scale</span><span className="v">×{cfg.scale}</span>
              <span className="k">Format</span><span className="v">{cfg.save_format}</span>
            </div>

            <div className="section-title" style={{ marginTop: 18 }}><h3 style={{ fontSize: 13 }}>Verification thumbnails</h3></div>
            <div className="grid-3" style={{ marginTop: 10 }}>
              <div className="thumb" style={{ opacity: done ? 1 : 0.3 }}>
                <div style={{ position: "absolute", inset: 0 }}>
                  <FieldScene seed={11} region="farmland" detail="hi" />
                </div>
                <div className="cap">HR tile</div>
              </div>
              <div className="thumb" style={{ opacity: done ? 1 : 0.3 }}>
                <div style={{ position: "absolute", inset: 0, filter: "blur(1.5px) saturate(.7)" }}>
                  <FieldScene seed={11} region="farmland" detail="lo" />
                </div>
                <div className="cap">LR tile</div>
              </div>
              <div className="thumb checker" style={{ opacity: done ? 1 : 0.3 }}>
                <div className="cap">pair check</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

