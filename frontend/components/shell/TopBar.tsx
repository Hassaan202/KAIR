"use client";
import { useRouter, usePathname } from "next/navigation";
import { Icons } from "@/components/design/Primitives";
import { useAppStore } from "@/lib/store";

const MODULE_META: Record<string, { label: string; crumb: string }> = {
  preprocessing: { label: "Preprocessing", crumb: "data · tiling · degradation" },
  training:      { label: "Training",       crumb: "model · optimisation" },
  inference:     { label: "Inference",      crumb: "reconstruct · compare" },
};

export default function TopBar() {
  const router = useRouter();
  const pathname = usePathname();
  const runStatus = useAppStore((s) => s.runStatus);

  const moduleId = Object.keys(MODULE_META).find((k) => pathname.includes(k)) ?? "";
  const meta = MODULE_META[moduleId];

  const statusMap = {
    idle:    { c: "var(--ink-3)", t: "idle" },
    running: { c: "var(--cobalt)", t: "running" },
    done:    { c: "var(--ok)", t: "complete" },
    failed:  { c: "var(--bad)", t: "failed" },
  };
  const { c, t } = statusMap[runStatus];

  return (
    <header className="topbar">
      <div className="topbar-title">
        <button
          className="btn btn-ghost"
          style={{ padding: "5px 8px", marginLeft: -6 }}
          onClick={() => router.push("/")}
          title="Back to landing"
        >
          <Icons.layers size={16} />
        </button>
        <h2>{meta?.label ?? ""}</h2>
        <span className="crumb">/ {meta?.crumb ?? ""}</span>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <div className="run-status">
          <span style={{ width: 7, height: 7, borderRadius: "50%", background: c }} />
          {t}
        </div>
        <button className="btn">
          <Icons.doc size={15} /> Docs
        </button>
      </div>
    </header>
  );
}
