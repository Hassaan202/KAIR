"use client";
import { useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { getGpuStatus } from "@/lib/api";
import { useAppStore } from "@/lib/store";

export default function StatusBar() {
  const { activeJobId, activeJobName, activeJobDetail } = useAppStore();
  const { data: gpu } = useQuery({
    queryKey: ["gpu"],
    queryFn: getGpuStatus,
    refetchInterval: 2000,
    retry: false,
  });

  const setGpuStatus = useAppStore((s) => s.setGpuStatus);
  useEffect(() => { if (gpu) setGpuStatus(gpu); }, [gpu, setGpuStatus]);

  const gpuUtil = gpu?.gpu_util ?? 0;
  const vramUsed = gpu?.vram_used_gb ?? 0;
  const vramTotal = gpu?.vram_total_gb ?? 80;

  return (
    <div className="statusbar">
      {/* GPU utilisation */}
      <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
        <div className="status-row">
          <span className="lbl">GPU · CUDA</span>
          <span className="v">{Math.round(gpuUtil)}%</span>
        </div>
        <div className="meter">
          <div style={{ width: gpuUtil + "%", background: "var(--cobalt)" }} />
        </div>
      </div>

      {/* VRAM */}
      <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
        <div className="status-row">
          <span className="lbl">VRAM</span>
          <span className="v">{vramUsed.toFixed(1)} / {vramTotal.toFixed(0)} GB</span>
        </div>
        <div className="meter">
          <div style={{ width: ((vramUsed / vramTotal) * 100) + "%", background: "var(--sage)" }} />
        </div>
      </div>

      {/* Active job */}
      <div className="status-job">
        {activeJobId ? (
          <>
            <span className="pulse-dot" />
            <div style={{ display: "flex", flexDirection: "column", lineHeight: 1.3 }}>
              <span className="mono" style={{ fontSize: 11, color: "var(--ink)" }}>{activeJobName}</span>
              <span className="mono" style={{ fontSize: 10, color: "var(--ink-3)" }}>{activeJobDetail}</span>
            </div>
          </>
        ) : (
          <>
            <span style={{ width: 7, height: 7, borderRadius: "50%", background: "var(--ink-3)", opacity: 0.5, flexShrink: 0 }} />
            <span className="mono" style={{ fontSize: 11, color: "var(--ink-3)" }}>no active job</span>
          </>
        )}
      </div>
    </div>
  );
}
