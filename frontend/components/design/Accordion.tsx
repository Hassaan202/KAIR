"use client";
import { useState, type ReactNode } from "react";

interface AccordionProps {
  title: string;
  sub?: string;
  children: ReactNode;
  open?: boolean;
  right?: ReactNode;
}

export default function Accordion({ title, sub, children, open: openProp = true, right }: AccordionProps) {
  const [open, setOpen] = useState(openProp);
  return (
    <div className="card" style={{ overflow: "hidden" }}>
      <button className="acc-head" onClick={() => setOpen(!open)}>
        <div style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
          <span className="acc-title">{title}</span>
          {sub && <span className="mono" style={{ fontSize: 11, color: "var(--ink-3)" }}>{sub}</span>}
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          {right}
          <svg width="14" height="14" viewBox="0 0 14 14"
            style={{ transform: open ? "rotate(180deg)" : "none", transition: "transform .2s", color: "var(--ink-3)" }}>
            <path d="M3 5 L7 9 L11 5" stroke="currentColor" strokeWidth="1.5" fill="none"
              strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </div>
      </button>
      {open && <div className="acc-body">{children}</div>}
    </div>
  );
}
