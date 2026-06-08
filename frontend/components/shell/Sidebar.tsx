"use client";
import { useRouter, usePathname } from "next/navigation";
import Wordmark from "@/components/design/Wordmark";
import StatusBar from "./StatusBar";
import { Icons } from "@/components/design/Primitives";

const NAV_ITEMS = [
  { id: "preprocessing", label: "Preprocessing", href: "/platform/preprocessing", icon: "preprocess", crumb: "data · tiling · degradation" },
  { id: "training",      label: "Training",       href: "/platform/training",      icon: "train",      crumb: "model · optimisation" },
  { id: "inference",     label: "Inference",      href: "/platform/inference",     icon: "inference",  crumb: "reconstruct · compare" },
] as const;

const WORKSPACE_ITEMS: Array<{ label: string; icon: string; badge?: string }> = [
  { label: "Datasets",    icon: "data",     badge: "5" },
  { label: "Checkpoints", icon: "layers",   badge: "—" },
  { label: "Reports",     icon: "doc" },
  { label: "Settings",    icon: "settings" },
];

export default function Sidebar() {
  const pathname = usePathname();
  const router = useRouter();

  return (
    <aside className="sidebar">
      <div className="sb-brand">
        <button
          onClick={() => router.push("/")}
          style={{ background: "none", border: "none", cursor: "pointer", padding: 0 }}
        >
          <Wordmark />
        </button>
      </div>

      <nav className="sb-nav scroll">
        <div className="sb-sec">Pipeline</div>
        {NAV_ITEMS.map((item) => {
          const Icon = Icons[item.icon as keyof typeof Icons];
          const active = pathname.includes(item.id);
          return (
            <button
              key={item.id}
              className={"sb-item" + (active ? " on" : "")}
              onClick={() => router.push(item.href)}
            >
              <Icon size={17} />
              {item.label}
            </button>
          );
        })}

        <div className="sb-sec">Workspace</div>
        {WORKSPACE_ITEMS.map((item) => {
          const Icon = Icons[item.icon as keyof typeof Icons];
          return (
            <button key={item.label} className="sb-item">
              <Icon size={17} />
              {item.label}
              {item.badge && <span className="badge">{item.badge}</span>}
            </button>
          );
        })}
      </nav>

      <StatusBar />
    </aside>
  );
}
