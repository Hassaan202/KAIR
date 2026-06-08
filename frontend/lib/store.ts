"use client";
import { create } from "zustand";
import type { GpuStatus, LogLine } from "./types";

interface AppState {
  activeModule: "preprocessing" | "training" | "inference";
  setActiveModule: (m: AppState["activeModule"]) => void;

  activeJobId: string | null;
  activeJobName: string | null;
  activeJobDetail: string | null;
  setActiveJob: (id: string | null, name?: string, detail?: string) => void;

  runStatus: "idle" | "running" | "done" | "failed";
  setRunStatus: (s: AppState["runStatus"]) => void;

  logs: LogLine[];
  appendLog: (line: LogLine) => void;
  clearLogs: () => void;

  gpuStatus: GpuStatus | null;
  setGpuStatus: (s: GpuStatus) => void;
}

export const useAppStore = create<AppState>((set) => ({
  activeModule: "preprocessing",
  setActiveModule: (m) => set({ activeModule: m }),

  activeJobId: null,
  activeJobName: null,
  activeJobDetail: null,
  setActiveJob: (id, name, detail) => set({ activeJobId: id, activeJobName: name ?? null, activeJobDetail: detail ?? null }),

  runStatus: "idle",
  setRunStatus: (s) => set({ runStatus: s }),

  logs: [],
  appendLog: (line) => set((state) => ({ logs: [...state.logs.slice(-500), line] })),
  clearLogs: () => set({ logs: [] }),

  gpuStatus: null,
  setGpuStatus: (s) => set({ gpuStatus: s }),
}));
