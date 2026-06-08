"use client";
import { useState, useRef, useCallback } from "react";
import dynamic from "next/dynamic";
import LogStream from "@/components/design/LogStream";
import { Segmented, MetricCard } from "@/components/design/Primitives";
import { Icons } from "@/components/design/Primitives";
import { NumberField, TextInput, Toggle } from "@/components/design/Primitives";
import FieldScene, { REGIONS } from "@/components/design/FieldScene";
import { DEFAULT_INFERENCE } from "@/lib/defaults";
import type { InferenceConfig, LogLine, MetricsResult, Upsampler } from "@/lib/types";
import { runInference, uploadImage, streamLogs } from "@/lib/api";
import { useAppStore } from "@/lib/store";
import { METRIC_WEIGHTS } from "@/lib/types";

const SplitCompare = dynamic(() => import("@/components/design/SplitCompare"), { ssr: false });

const METRIC_DISPLAY: { key: keyof MetricsResult; label: string; unit: string; higherBetter: boolean }[] = [
  { key: "psnr",    label: "PSNR",    unit: "dB",  higherBetter: true  },
  { key: "ssim",    label: "SSIM",    unit: "",    higherBetter: true  },
  { key: "sam",     label: "SAM",     unit: "°",   higherBetter: false },
  { key: "uiqi",    label: "UIQI",    unit: "",    higherBetter: true  },
  { key: "fsim",    label: "FSIM",    unit: "",    higherBetter: true  },
  { key: "rmse",    label: "RMSE",    unit: "",    higherBetter: false },
  { key: "it_ssim", label: "IT-SSIM", unit: "",    higherBetter: true  },
  { key: "srer",    label: "SRER",    unit: "",    higherBetter: true  },
];

type ViewMode = "split" | "triptych";
type Region = "delta" | "farmland" | "urban" | "mountain";
const REGION_OPTIONS: Region[] = ["delta", "farmland", "urban", "mountain"];

export default function InferenceModule() {
  const [cfg, setCfg] = useState<InferenceConfig>(DEFAULT_INFERENCE);
  const [logs, setLogs] = useState<LogLine[]>([]);
  const [running, setRunning] = useState(false);
  const [hasResult, setHasResult] = useState(false);
  const [metrics, setMetrics] = useState<MetricsResult | null>(null);
  const [view, setView] = useState<ViewMode>("split");
  const [region, setRegion] = useState<Region>("delta");
  const [seed, setSeed] = useState(9);
  const [uploadedFile, setUploadedFile] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const stopStream = useRef<(() => void) | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const { setActiveJob, setRunStatus } = useAppStore();

  const setCfgField = <K extends keyof InferenceConfig>(k: K, v: InferenceConfig[K]) =>
    setCfg((c) => ({ ...c, [k]: v }));
  const setMC = <K extends keyof InferenceConfig["model_config"]>(k: K, v: InferenceConfig["model_config"][K]) =>
    setCfg((c) => ({ ...c, model_config: { ...c.model_config, [k]: v } }));

  const handleFileUpload = async (file: File) => {
    try {
      const res = await uploadImage(file);
      setUploadedFile(file.name);
      setCfgField("lr_dir", res.path.replace(/\/[^/]+$/, ""));
    } catch (e) {
      console.error(e);
    }
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault(); setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFileUpload(file);
  }, []);

  const handleRun = async () => {
    setRunning(true); setHasResult(false); setLogs([]); setMetrics(null);
    setRunStatus("running");
    setActiveJob("pending", "inference · SwinIR", "reconstructing");

    try {
      const res = await runInference(cfg);
      stopStream.current = streamLogs(
        res.job_id,
        (line: LogLine) => {
          setLogs((l: LogLine[]) => [...l, line]);
          // Parse metrics from log lines like "Average PSNR: 46.58"
          const psnrM = line.text.match(/PSNR[:\s]+([\d.]+)/i);
          const ssimM = line.text.match(/SSIM[:\s]+([\d.]+)/i);
          const samM  = line.text.match(/SAM[:\s]+([\d.]+)/i);
          const srerM = line.text.match(/SRER[:\s]+([\d.]+)/i);
          if (psnrM || ssimM) {
            setMetrics((m) => ({
              psnr:    psnrM ? parseFloat(psnrM[1]) : (m?.psnr ?? 0),
              ssim:    ssimM ? parseFloat(ssimM[1]) : (m?.ssim ?? 0),
              it_ssim: ssimM ? parseFloat(ssimM[1]) : (m?.it_ssim ?? 0),
              sam:     samM  ? parseFloat(samM[1])  : (m?.sam  ?? 0),
              uiqi:    m?.uiqi  ?? 0,
              fsim:    m?.fsim  ?? 0,
              rmse:    m?.rmse  ?? 0,
              srer:    srerM ? parseFloat(srerM[1]) : (m?.srer ?? 0),
            }));
          }
        },
        () => {
          setRunning(false); setHasResult(true);
          setRunStatus("done"); setActiveJob(null);
          setSeed((s) => s + 1);
        },
      );
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setLogs((l) => [...l, { ts: new Date().toTimeString().slice(0, 8), text: `Error: ${msg}`, lv: "warn" }]);
      setRunning(false); setRunStatus("failed"); setActiveJob(null);
    }
  };

  return (
    <div className="content-wide">
      {/* ---- Control strip ---- */}
      <div className="infer-bar card">
        {/* File upload */}
        <div
          className="dropzone-mini"
          style={{ cursor: "pointer", borderColor: dragOver ? "var(--accent)" : undefined }}
          onClick={() => fileInputRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
        >
          <Icons.upload size={18} />
          <div>
            <div style={{ fontWeight: 500, fontSize: 13 }}>
              {uploadedFile ?? "Drop image or click to browse"}
            </div>
            <span className="mono" style={{ fontSize: 10.5, color: "var(--ink-3)" }}>
              GeoTIFF · PNG · JPEG · TIF — LR input image
            </span>
          </div>
          {uploadedFile && (
            <button className="btn btn-ghost" style={{ marginLeft: "auto", padding: "5px 10px", fontSize: 12 }}
              onClick={(e) => { e.stopPropagation(); setUploadedFile(null); }}>
              remove
            </button>
          )}
        </div>
        <input ref={fileInputRef} type="file" accept=".png,.jpg,.jpeg,.tif,.tiff"
          style={{ display: "none" }}
          onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFileUpload(f); }} />

        {/* Controls row */}
        <div className="infer-controls">
          <div>
            <div className="micro-label">Scale</div>
            <Segmented options={["×2", "×3", "×4"]}
              value={"×" + cfg.model_config.upscale}
              onChange={(v) => setMC("upscale", parseInt(v.slice(1)) as 2 | 3 | 4)} />
          </div>
          <div>
            <div className="micro-label">Upsampler</div>
            <Segmented options={["pixelshuffle", "nearest+conv"]}
              value={cfg.model_config.upsampler}
              onChange={(v) => setMC("upsampler", v as Upsampler)} />
          </div>
          <div>
            <div className="micro-label">Embed dim</div>
            <Segmented options={["60", "180"]}
              value={String(cfg.model_config.embed_dim)}
              onChange={(v) => setMC("embed_dim", parseInt(v) as 60 | 180)} />
          </div>
          <div>
            <div className="micro-label">Scene (preview)</div>
            <Segmented options={REGION_OPTIONS}
              value={region}
              onChange={(v) => { setRegion(v as Region); setSeed((s) => s + 1); }} />
          </div>
          <button
            className="btn btn-primary"
            style={{ marginLeft: "auto", alignSelf: "end" }}
            onClick={handleRun}
            disabled={running}
          >
            {running ? "Reconstructing…" : <><Icons.spark size={14} /> Reconstruct</>}
          </button>
        </div>

        {/* Paths sub-row */}
        <div className="grid-2" style={{ gap: 10 }}>
          <TextInput label="Model checkpoint path" value={cfg.model_path}
            onChange={(v) => setCfgField("model_path", v)} mono
            placeholder="superresolution/task/models/100000_G.pth" />
          <TextInput label="HR reference directory (for metrics)" value={cfg.hr_dir}
            onChange={(v) => setCfgField("hr_dir", v)} mono />
          <TextInput label="SR output directory" value={cfg.sr_dir}
            onChange={(v) => setCfgField("sr_dir", v)} mono />
          <div className="grid-2">
            <div>
              <div className="field-label"><span>Tile size</span><span className="val">{cfg.tile ?? "none"}</span></div>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <Toggle checked={cfg.tile !== null}
                  onChange={(v) => setCfgField("tile", v ? 256 : null)} />
                {cfg.tile !== null && (
                  <input className="num-input" type="number" value={cfg.tile ?? 256} step={64}
                    onChange={(e) => setCfgField("tile", parseInt(e.target.value))}
                    style={{ width: 80 }} />
                )}
              </div>
            </div>
            <NumberField label="Tile overlap" value={cfg.tile_overlap}
              onChange={(v) => setCfgField("tile_overlap", v)} />
          </div>
        </div>
      </div>

      {/* ---- Viewer ---- */}
      <div className="infer-viewer">
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12 }}>
          <div className="section-title" style={{ margin: 0 }}>
            <h3 style={{ fontSize: 16 }}>Reconstruction viewer</h3>
            <span className="mono" style={{ color: "var(--ink-3)" }}>
              {REGIONS[region]?.name} · ×{cfg.model_config.upscale}
            </span>
          </div>
          <Segmented options={["split", "triptych"]} value={view} onChange={(v) => setView(v as ViewMode)} />
        </div>

        {view === "split" ? (
          <div style={{ position: "relative", opacity: hasResult || running ? 1 : 0.55, transition: "opacity .3s" }}>
            <SplitCompare
              key={seed}
              seed={seed}
              region={region}
              height={460}
              autoIntro={hasResult}
              labelL={`LR · ${10 * cfg.model_config.upscale}m input`}
              labelR={`SR · ×${cfg.model_config.upscale} · SwinIR`}
            />
            {running && (
              <div className="viewer-overlay">
                <span className="cursor" /> reconstructing…
              </div>
            )}
          </div>
        ) : (
          <div className="triptych">
            {[
              { label: "LR · input",       detail: "lo" as const, opacity: 1 },
              { label: "SR · SwinIR",       detail: "hi" as const, opacity: hasResult ? 1 : 0.5 },
              { label: "HR · ground truth", detail: "hi" as const, opacity: hasResult ? 1 : 0.5 },
            ].map((panel, i) => (
              <div key={i} className="tri-panel" style={{ opacity: panel.opacity }}>
                <div className={"tri-scene" + (panel.detail === "lo" ? " lr-blur" : "")}>
                  <FieldScene seed={seed + i} region={region} detail={panel.detail} />
                </div>
                <span className="split-tag left">{panel.label}</span>
              </div>
            ))}
          </div>
        )}

        {/* ---- All 8 metrics ---- */}
        <div className="grid-4" style={{ marginTop: 18 }}>
          {METRIC_DISPLAY.map((m) => {
            const val = metrics?.[m.key];
            const weight = METRIC_WEIGHTS[m.key];
            return (
              <MetricCard
                key={m.key}
                label={m.label}
                value={hasResult && val != null ? val.toFixed(m.key === "psnr" ? 2 : 4) : "—"}
                unit={m.unit}
                sub={hasResult ? `weight: ${(weight * 100).toFixed(0)}% · ${m.higherBetter ? "higher=better" : "lower=better"}` : "awaiting run"}
                subColor={hasResult ? (m.higherBetter && val != null && val > 0 ? "var(--ok)" : "var(--ink-3)") : "var(--ink-3)"}
              />
            );
          })}
        </div>

        {/* ---- Log stream + download ---- */}
        <div style={{ marginTop: 18 }}>
          <LogStream
            lines={logs.length ? logs : [{ ts: "--:--:--", text: "inference log idle — press Reconstruct to start", lv: "info" }]}
            running={running}
            height={140}
          />
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 12, marginTop: 18 }}>
          <button className="btn btn-primary btn-lg" disabled={!hasResult}>
            <Icons.download size={15} /> Download SR output
          </button>
          <span className="mono" style={{ fontSize: 11.5, color: "var(--ink-3)" }}>
            {cfg.sr_dir} · ×{cfg.model_config.upscale} · preserves CRS / geotransform
          </span>
          {logs.length > 0 && (
            <span className="chip" style={{ marginLeft: "auto" }}>
              {running ? "running" : hasResult ? "done" : "error"}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
