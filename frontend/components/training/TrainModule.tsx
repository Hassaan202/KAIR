"use client";
import { useState, useRef } from "react";
import { useQuery } from "@tanstack/react-query";
import Accordion from "@/components/design/Accordion";
import LogStream from "@/components/design/LogStream";
import LineChart from "@/components/design/LineChart";
import {
  Slider, NumberField, TextInput, Segmented, Toggle, CardToggle, BenchRow,
} from "@/components/design/Primitives";
import { Icons } from "@/components/design/Primitives";
import { DEFAULT_TRAINING } from "@/lib/defaults";
import type { TrainingJobConfig, LogLine, LossFn, GanType, Upsampler, ResiConnection, InitType } from "@/lib/types";
import { startTraining, stopTraining, listCheckpoints, deleteCheckpoint, streamLogs } from "@/lib/api";
import { useAppStore } from "@/lib/store";

// Loss function options from the repo
const LOSS_OPTIONS: { id: LossFn; name: string; sub: string }[] = [
  { id: "l1",          name: "L1",          sub: "recommended · smooth" },
  { id: "l2",          name: "L2",          sub: "MSE · penalises outliers" },
  { id: "l2sum",       name: "L2 sum",      sub: "sum over pixels" },
  { id: "ssim",        name: "SSIM",        sub: "structural similarity" },
  { id: "charbonnier", name: "Charbonnier", sub: "robust L1 variant" },
];

// Dataset types from data/select_dataset.py
const DATASET_TYPES = [
  { value: "sr",        label: "sr — paired HR/LR" },
  { value: "blindsr",   label: "blindsr — on-the-fly BSRGAN" },
  { value: "dncnn",     label: "dncnn — Gaussian denoising" },
  { value: "fdncnn",    label: "fdncnn — flexible denoising" },
  { value: "ffdnet",    label: "ffdnet — variable noise level" },
  { value: "srmd",      label: "srmd — multi-degradation SR" },
  { value: "dpsr",      label: "dpsr — degradation prior" },
  { value: "jpeg",      label: "jpeg — artifact reduction" },
  { value: "plain",     label: "plain — single image" },
  { value: "plainpatch",label: "plainpatch — patch-based" },
];


// GAN types
const GAN_TYPES: GanType[] = ["gan", "ragan", "lsgan", "wgan", "softplusgan"];

function makeCurve(n: number, start: number, end: number, noise: number, up = true): [number, number][] {
  const out: [number, number][] = [];
  for (let i = 0; i < n; i++) {
    const t = i / (n - 1);
    const base = up
      ? start + (end - start) * (1 - Math.pow(1 - t, 1.7))
      : start + (end - start) * Math.pow(t, 0.6);
    out.push([i, base + (Math.random() - 0.5) * noise]);
  }
  return out;
}

export default function TrainModule() {
  const [cfg, setCfg] = useState<TrainingJobConfig>(DEFAULT_TRAINING);
  const [mode, setMode] = useState<"plain" | "gan">("plain");
  const [logs, setLogs] = useState<LogLine[]>([]);
  const [running, setRunning] = useState(false);
  const [done, setDone] = useState(false);
  const [progress, setProgress] = useState(0);
  const [jobId, setJobId] = useState<string | null>(null);
  const [liveMetrics, setLiveMetrics] = useState({ psnr: 0, ssim: 0, sam: 4.8, srer: 0 });
  const stopStream = useRef<(() => void) | null>(null);
  const psnrCurve = useRef(makeCurve(40, 28, 46.58, 0.5));
  const lossCurve = useRef(makeCurve(40, 0.42, 0.031, 0.01, false));

  const { setActiveJob, setRunStatus } = useAppStore();

  const { data: checkpoints, refetch: refetchCkpts } = useQuery({
    queryKey: ["checkpoints"],
    queryFn: listCheckpoints,
    refetchInterval: running ? 30000 : false,
  });

  const set = <K extends keyof TrainingJobConfig>(k: K, v: TrainingJobConfig[K]) =>
    setCfg((c) => ({ ...c, [k]: v }));
  const setTrain = <K extends keyof TrainingJobConfig["train"]>(k: K, v: TrainingJobConfig["train"][K]) =>
    setCfg((c) => ({ ...c, train: { ...c.train, [k]: v } }));
  const setNetG = <K extends keyof TrainingJobConfig["netG"]>(k: K, v: TrainingJobConfig["netG"][K]) =>
    setCfg((c) => ({ ...c, netG: { ...c.netG, [k]: v } }));

  const visN = Math.max(2, Math.round((progress / 100) * 40));

  const handleStart = async () => {
    const fullCfg = { ...cfg, model: mode };
    setRunning(true); setDone(false); setProgress(0); setLogs([]);
    setRunStatus("running");
    setActiveJob("pending", `train · SwinIR ×${cfg.netG.upscale}`, `${mode} · scale ×${cfg.scale}`);

    try {
      const res = await startTraining(fullCfg);
      setJobId(res.job_id);
      stopStream.current = streamLogs(
        res.job_id,
        (line) => {
          setLogs((l) => [...l, line]);
          const m = line.text.match(/psnr\s+([\d.]+)/i);
          if (m) {
            const psnr = parseFloat(m[1]);
            setLiveMetrics((prev) => ({ ...prev, psnr }));
            const ssimM = line.text.match(/ssim\s+([\d.]+)/i);
            if (ssimM) setLiveMetrics((prev) => ({ ...prev, ssim: parseFloat(ssimM[1]) }));
          }
          const pM = line.text.match(/epoch\s+(\d+)\/(\d+)/i);
          if (pM) setProgress(Math.round((parseInt(pM[1]) / parseInt(pM[2])) * 100));
        },
        () => {
          setRunning(false); setDone(true); setProgress(100);
          setRunStatus("done"); setActiveJob(null);
          refetchCkpts();
        },
      );
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setLogs((l) => [...l, { ts: new Date().toTimeString().slice(0, 8), text: `Error: ${msg}`, lv: "warn" }]);
      setRunning(false); setRunStatus("failed"); setActiveJob(null);
    }
  };

  const handleStop = async () => {
    if (jobId) await stopTraining(jobId).catch(() => {});
    stopStream.current?.();
    setRunning(false); setRunStatus("idle"); setActiveJob(null);
  };

  const handleDeleteCkpt = async (name: string) => {
    await deleteCheckpoint(name).catch(() => {});
    refetchCkpts();
  };

  return (
    <div className="content">
      <div className="module-grid">
        {/* ===== LEFT: config ===== */}
        <div className="col">

          {/* 01 — Model */}
          <Accordion title="Model" sub="01" right={<span className="chip">SwinIR</span>}>
            <div className="grid-2">
              <div>
                <div className="field-label"><span>Training mode</span></div>
                <Segmented options={["plain", "gan"]} value={mode} onChange={(v) => setMode(v as "plain" | "gan")} />
                <div className="mono" style={{ fontSize: 11, color: "var(--ink-3)", marginTop: 6 }}>
                  {mode === "plain" ? "main_train_swinir.py" : "main_train_swinir_gan.py"}
                </div>
              </div>
              <div>
                <div className="field-label"><span>Scale factor</span></div>
                <Segmented options={["×2", "×3", "×4"]}
                  value={"×" + cfg.scale}
                  onChange={(v) => {
                    const s = parseInt(v.slice(1)) as 2 | 3 | 4;
                    set("scale", s); setNetG("upscale", s);
                  }} />
              </div>
              <div>
                <div className="field-label"><span>Channels</span></div>
                <Segmented options={["1 (gray)", "3 (RGB)"]}
                  value={cfg.n_channels === 1 ? "1 (gray)" : "3 (RGB)"}
                  onChange={(v) => set("n_channels", v.startsWith("1") ? 1 : 3)} />
              </div>
              <div>
                <div className="field-label"><span>Root directory</span></div>
                <Segmented options={["superresolution", "denoising", "dejpeg"]}
                  value={cfg.path.root}
                  onChange={(v) => set("path", { ...cfg.path, root: v as "superresolution" | "denoising" | "dejpeg" })} />
              </div>
            </div>
          </Accordion>

          {/* 02 — SwinIR architecture */}
          <Accordion title="SwinIR architecture (netG)" sub="02">
            <div className="grid-2">
              <div>
                <div className="field-label"><span>Embed dim</span></div>
                <Segmented options={["60 (lightweight)", "180 (medium)"]}
                  value={cfg.netG.embed_dim === 60 ? "60 (lightweight)" : "180 (medium)"}
                  onChange={(v) => setNetG("embed_dim", v.startsWith("60") ? 60 : 180)} />
              </div>
              <div>
                <div className="field-label"><span>Upsampler</span></div>
                <Segmented options={["pixelshuffle", "nearest+conv"]}
                  value={cfg.netG.upsampler}
                  onChange={(v) => setNetG("upsampler", v as Upsampler)} />
              </div>
              <div>
                <div className="field-label"><span>Residual connection</span></div>
                <Segmented options={["1conv", "3conv"]}
                  value={cfg.netG.resi_connection}
                  onChange={(v) => setNetG("resi_connection", v as ResiConnection)} />
              </div>
              <div>
                <div className="field-label"><span>Window size</span></div>
                <Segmented options={["7", "8"]}
                  value={String(cfg.netG.window_size)}
                  onChange={(v) => setNetG("window_size", parseInt(v) as 7 | 8)} />
              </div>
              <div>
                <div className="field-label"><span>Image size (patch)</span></div>
                <Segmented options={["64", "128"]}
                  value={String(cfg.netG.img_size)}
                  onChange={(v) => setNetG("img_size", parseInt(v) as 64 | 128)} />
              </div>
              <div>
                <div className="field-label"><span>Image range</span></div>
                <Segmented options={["1.0", "255.0"]}
                  value={String(cfg.netG.img_range)}
                  onChange={(v) => setNetG("img_range", parseFloat(v) as 1.0 | 255.0)} />
              </div>
              <TextInput label="Depths (comma-separated)" value={cfg.netG.depths.join(", ")}
                onChange={(v) => setNetG("depths", v.split(",").map((x) => parseInt(x.trim())).filter((x) => !isNaN(x)))} />
              <TextInput label="Num heads (comma-separated)" value={cfg.netG.num_heads.join(", ")}
                onChange={(v) => setNetG("num_heads", v.split(",").map((x) => parseInt(x.trim())).filter((x) => !isNaN(x)))} />
              <div>
                <div className="field-label"><span>Init type</span></div>
                <select className="text-input" value={cfg.netG.init_type}
                  onChange={(e) => setNetG("init_type", e.target.value as InitType)}>
                  {["default","orthogonal","normal","uniform","xavier_normal","xavier_uniform","kaiming_normal","kaiming_uniform"].map((o) => (
                    <option key={o} value={o}>{o}</option>
                  ))}
                </select>
              </div>
            </div>
          </Accordion>

          {/* 03 — Dataset */}
          <Accordion title="Dataset" sub="03">
            <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
              <div>
                <div className="field-label"><span>Train dataset type</span></div>
                <select className="text-input" value={cfg.datasets.train.dataset_type}
                  onChange={(e) => setCfg((c) => ({ ...c, datasets: { ...c.datasets, train: { ...c.datasets.train, dataset_type: e.target.value as TrainingJobConfig["datasets"]["train"]["dataset_type"] } } }))}>
                  {DATASET_TYPES.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
                </select>
              </div>
              <TextInput label="Train HR directory" value={cfg.datasets.train.dataroot_H}
                onChange={(v) => setCfg((c) => ({ ...c, datasets: { ...c.datasets, train: { ...c.datasets.train, dataroot_H: v } } }))} mono />
              <TextInput label="Train LR directory" value={cfg.datasets.train.dataroot_L ?? ""}
                onChange={(v) => setCfg((c) => ({ ...c, datasets: { ...c.datasets, train: { ...c.datasets.train, dataroot_L: v || null } } }))} mono />
              <TextInput label="Test HR directory" value={cfg.datasets.test.dataroot_H}
                onChange={(v) => setCfg((c) => ({ ...c, datasets: { ...c.datasets, test: { ...c.datasets.test, dataroot_H: v } } }))} mono />
              <TextInput label="Test LR directory" value={cfg.datasets.test.dataroot_L ?? ""}
                onChange={(v) => setCfg((c) => ({ ...c, datasets: { ...c.datasets, test: { ...c.datasets.test, dataroot_L: v || null } } }))} mono />
              <div className="grid-2">
                <Slider label="Patch size (H_size)" value={cfg.datasets.train.H_size}
                  min={64} max={512} step={32}
                  onChange={(v) => setCfg((c) => ({ ...c, datasets: { ...c.datasets, train: { ...c.datasets.train, H_size: v } } }))} />
                <Slider label="Batch size" value={cfg.datasets.train.dataloader_batch_size}
                  min={1} max={64} step={1}
                  onChange={(v) => setCfg((c) => ({ ...c, datasets: { ...c.datasets, train: { ...c.datasets.train, dataloader_batch_size: v } } }))} />
                <NumberField label="Num workers" value={cfg.datasets.train.dataloader_num_workers}
                  onChange={(v) => setCfg((c) => ({ ...c, datasets: { ...c.datasets, train: { ...c.datasets.train, dataloader_num_workers: v } } }))} />
                <div style={{ display: "flex", alignItems: "center", gap: 10, paddingTop: 24 }}>
                  <Toggle checked={cfg.datasets.train.use_photometric_aug}
                    onChange={(v) => setCfg((c) => ({ ...c, datasets: { ...c.datasets, train: { ...c.datasets.train, use_photometric_aug: v } } }))} />
                  <span style={{ fontSize: 13 }}>Photometric augmentation</span>
                </div>
              </div>
            </div>
          </Accordion>

          {/* 04 — Loss functions */}
          <Accordion title="Loss functions" sub="04">
            <div className="grid-3" style={{ marginBottom: 14 }}>
              {LOSS_OPTIONS.map((l) => (
                <CardToggle key={l.id} on={cfg.train.G_lossfn_type === l.id}
                  onClick={() => setTrain("G_lossfn_type", l.id)}
                  title={l.name} sub={l.sub} />
              ))}
            </div>
            {mode === "gan" && (
              <>
                <div className="section-title" style={{ marginBottom: 10 }}><h3 style={{ fontSize: 13 }}>GAN losses</h3></div>
                <div className="grid-2">
                  <div>
                    <div className="field-label"><span>GAN type</span></div>
                    <select className="text-input" value={cfg.train.gan_type}
                      onChange={(e) => setTrain("gan_type", e.target.value as GanType)}>
                      {GAN_TYPES.map((g) => <option key={g} value={g}>{g}</option>)}
                    </select>
                  </div>
                  <Slider label="D loss weight" value={cfg.train.D_lossfn_weight} min={0} max={2} step={0.05}
                    onChange={(v) => setTrain("D_lossfn_weight", v)} fmt={(v) => v.toFixed(2)} />
                  <Slider label="Perceptual weight" value={cfg.train.F_lossfn_weight} min={0} max={2} step={0.05}
                    onChange={(v) => setTrain("F_lossfn_weight", v)} fmt={(v) => v.toFixed(2)} />
                  <div>
                    <div className="field-label"><span>Perceptual loss</span></div>
                    <Segmented options={["l1", "l2"]} value={cfg.train.F_lossfn_type}
                      onChange={(v) => setTrain("F_lossfn_type", v as "l1" | "l2")} />
                  </div>
                </div>
              </>
            )}
          </Accordion>

          {/* 05 — Optimizer & scheduler */}
          <Accordion title="Optimizer & scheduler" sub="05">
            <div className="grid-2">
              <Slider label="G learning rate (×1e-4)" value={cfg.train.G_optimizer_lr * 1e4}
                min={0.1} max={10} step={0.1}
                onChange={(v) => setTrain("G_optimizer_lr", v * 1e-4)}
                fmt={(v) => (v).toFixed(1) + "e-4"} />
              {mode === "gan" && (
                <Slider label="D learning rate (×1e-4)" value={cfg.train.D_optimizer_lr * 1e4}
                  min={0.1} max={10} step={0.1}
                  onChange={(v) => setTrain("D_optimizer_lr", v * 1e-4)}
                  fmt={(v) => (v).toFixed(1) + "e-4"} />
              )}
              <Slider label="EMA decay (E_decay)" value={cfg.train.E_decay}
                min={0} max={1} step={0.001}
                onChange={(v) => setTrain("E_decay", v)} fmt={(v) => v.toFixed(3)} />
              <NumberField label="Optimizer weight decay" value={cfg.train.G_optimizer_wd}
                onChange={(v) => setTrain("G_optimizer_wd", v)} step={0.0001} />
            </div>
            <div className="section-title" style={{ marginTop: 16, marginBottom: 10 }}><h3 style={{ fontSize: 13 }}>Scheduler (MultiStepLR)</h3></div>
            <div className="grid-2">
              <TextInput label="Milestones (comma-separated)" value={cfg.train.G_scheduler_milestones.join(", ")}
                onChange={(v) => setTrain("G_scheduler_milestones", v.split(",").map((x) => parseInt(x.trim())).filter((x) => !isNaN(x)))} />
              <Slider label="Gamma" value={cfg.train.G_scheduler_gamma}
                min={0.1} max={1} step={0.05}
                onChange={(v) => setTrain("G_scheduler_gamma", v)} fmt={(v) => v.toFixed(2)} />
            </div>
          </Accordion>

          {/* 06 — Checkpointing */}
          <Accordion title="Checkpointing" sub="06">
            <div className="grid-3">
              <NumberField label="Test every N iters" value={cfg.train.checkpoint_test}
                onChange={(v) => setTrain("checkpoint_test", v)} />
              <NumberField label="Save every N iters" value={cfg.train.checkpoint_save}
                onChange={(v) => setTrain("checkpoint_save", v)} />
              <NumberField label="Print every N iters" value={cfg.train.checkpoint_print}
                onChange={(v) => setTrain("checkpoint_print", v)} />
            </div>
            <TextInput label="Pretrained netG (optional)" value={cfg.path.pretrained_netG ?? ""}
              onChange={(v) => set("path", { ...cfg.path, pretrained_netG: v || null })} mono
              placeholder="model_zoo/swinir/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth" />
          </Accordion>

          {/* 07 — Live training charts */}
          <div className="card" style={{ padding: 18 }}>
            <div className="section-title" style={{ marginBottom: 12 }}>
              <h3>Live training</h3>
              <span className="mono" style={{ color: "var(--ink-3)", fontSize: 11 }}>
                {running ? `${progress}%` : done ? "converged" : "idle"}
              </span>
            </div>
            <div className="grid-2" style={{ gap: 18 }}>
              <div>
                <div className="mono" style={{ fontSize: 11, color: "var(--ink-3)", marginBottom: 4 }}>PSNR (dB)</div>
                <LineChart series={[{ data: psnrCurve.current.slice(0, visN), color: "var(--cobalt)" }]} height={150} xLabel="epoch" />
              </div>
              <div>
                <div className="mono" style={{ fontSize: 11, color: "var(--ink-3)", marginBottom: 4 }}>Loss</div>
                <LineChart series={[{ data: lossCurve.current.slice(0, visN), color: "var(--terracotta)" }]} height={150} xLabel="epoch" />
              </div>
            </div>
            <div style={{ marginTop: 8 }}>
              <LogStream
                lines={logs.length ? logs : [{ ts: "--:--:--", text: "training log idle", lv: "info" }]}
                running={running}
                height={130}
              />
            </div>
          </div>
        </div>

        {/* ===== RIGHT: run + benchmark + checkpoints ===== */}
        <div className="col" style={{ position: "sticky", top: 88 }}>
          <div className="card" style={{ padding: 18 }}>
            {running ? (
              <button className="btn" style={{ width: "100%", justifyContent: "center", borderColor: "var(--warn)", color: "var(--warn)" }}
                onClick={handleStop}>
                <Icons.stop size={13} /> Stop training
              </button>
            ) : (
              <button className="btn btn-primary" style={{ width: "100%", justifyContent: "center" }}
                onClick={handleStart}>
                <Icons.play size={13} /> Start training
              </button>
            )}
            {running && (
              <div style={{ marginTop: 12 }}>
                <div style={{ height: 6, background: "var(--bg-2)", borderRadius: 4, overflow: "hidden" }}>
                  <div style={{ width: progress + "%", height: "100%", background: "var(--cobalt)", borderRadius: 4, transition: "width .3s" }} />
                </div>
                <div className="mono" style={{ fontSize: 11, color: "var(--ink-3)", marginTop: 4, textAlign: "center" }}>{progress}%</div>
              </div>
            )}
          </div>

          <div className="card" style={{ padding: 18 }}>
            <div className="section-title" style={{ marginBottom: 14 }}><h3>Benchmark vs target</h3></div>
            <div className="bench">
              <BenchRow label="PSNR" value={+liveMetrics.psnr.toFixed(2)} target={45} max={50} unit="dB" color="var(--cobalt)" />
              <BenchRow label="SSIM" value={+liveMetrics.ssim.toFixed(3)} target={0.95} max={1} color="var(--sage)" />
              <BenchRow label="SAM" value={+liveMetrics.sam.toFixed(2)} target={2.5} max={6} color="var(--terracotta)" />
              <BenchRow label="SRER" value={+liveMetrics.srer.toFixed(2)} target={0.8} max={1} color="var(--cobalt-deep)" />
            </div>
          </div>

          <div className="card" style={{ padding: 18 }}>
            <div className="section-title" style={{ marginBottom: 12 }}>
              <h3>Checkpoints</h3>
              <span className="mono">{checkpoints?.length ?? 0}</span>
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {(checkpoints ?? []).slice(0, 8).map((c) => (
                <div key={c.name} className="ckpt-row">
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div className="mono" style={{ fontSize: 12, color: "var(--ink)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                      {c.name}
                      {c.is_best && <span className="chip" style={{ fontSize: 9, padding: "0 6px", marginLeft: 4, color: "var(--cobalt-deep)", borderColor: "var(--cobalt-soft)" }}>best</span>}
                    </div>
                    <span className="mono" style={{ fontSize: 10.5, color: "var(--ink-3)" }}>{c.psnr > 0 ? `psnr ${c.psnr.toFixed(2)} dB · ` : ""}{c.size_mb} MB</span>
                  </div>
                  <div style={{ display: "flex", gap: 4 }}>
                    <button className="btn btn-ghost icon-btn" title="delete" onClick={() => handleDeleteCkpt(c.name)}>
                      <Icons.trash size={14} />
                    </button>
                  </div>
                </div>
              ))}
              {(!checkpoints || checkpoints.length === 0) && (
                <div className="mono" style={{ fontSize: 12, color: "var(--ink-3)" }}>No checkpoints found in superresolution/</div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
