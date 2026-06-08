"use client";
import { type ReactNode } from "react";

// ---- Slider ----
interface SliderProps {
  label: string; value: number; min: number; max: number;
  step?: number; unit?: string; onChange: (v: number) => void;
  fmt?: (v: number) => string;
}
export function Slider({ label, value, min, max, step = 1, unit = "", onChange, fmt }: SliderProps) {
  const display = fmt ? fmt(value) : value + (unit ? " " + unit : "");
  return (
    <div>
      <div className="field-label">
        <span>{label}</span>
        <span className="val">{display}</span>
      </div>
      <input
        type="range" className="slider"
        min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
    </div>
  );
}

// ---- NumberField ----
interface NumberFieldProps { label: string; value: number; onChange: (v: number) => void; step?: number; suffix?: string; }
export function NumberField({ label, value, onChange, step = 1, suffix }: NumberFieldProps) {
  return (
    <div>
      <div className="field-label"><span>{label}</span></div>
      <div style={{ position: "relative" }}>
        <input className="num-input" type="number" value={value} step={step}
          onChange={(e) => onChange(parseFloat(e.target.value))} />
        {suffix && <span className="mono" style={{ position: "absolute", right: 10, top: 9, fontSize: 12, color: "var(--ink-3)" }}>{suffix}</span>}
      </div>
    </div>
  );
}

// ---- TextInput ----
interface TextInputProps { label: string; value: string; onChange: (v: string) => void; placeholder?: string; mono?: boolean; }
export function TextInput({ label, value, onChange, placeholder, mono }: TextInputProps) {
  return (
    <div>
      <div className="field-label"><span>{label}</span></div>
      <input className="text-input" type="text" value={value} placeholder={placeholder}
        style={mono ? { fontFamily: "var(--font-mono)" } : undefined}
        onChange={(e) => onChange(e.target.value)} />
    </div>
  );
}

// ---- Select ----
interface SelectProps { label: string; value: string; options: { value: string; label: string }[]; onChange: (v: string) => void; }
export function Select({ label, value, options, onChange }: SelectProps) {
  return (
    <div>
      <div className="field-label"><span>{label}</span></div>
      <select className="text-input" value={value} onChange={(e) => onChange(e.target.value)}>
        {options.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
      </select>
    </div>
  );
}

// ---- Segmented ----
interface SegmentedProps { options: string[]; value: string; onChange: (v: string) => void; }
export function Segmented({ options, value, onChange }: SegmentedProps) {
  return (
    <div className="seg">
      {options.map((o) => (
        <button key={o} className={value === o ? "on" : ""} onClick={() => onChange(o)}>{o}</button>
      ))}
    </div>
  );
}

// ---- Toggle switch ----
interface ToggleProps { checked: boolean; onChange: (v: boolean) => void; }
export function Toggle({ checked, onChange }: ToggleProps) {
  return (
    <label className="switch" onClick={(e) => e.stopPropagation()}>
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      <span className="track" />
    </label>
  );
}

// ---- StatPill ----
interface StatPillProps { k: string; l: string; dot?: string; delay?: number; }
export function StatPill({ k, l, dot, delay = 0 }: StatPillProps) {
  return (
    <div className="stat-pill rise" style={{ animationDelay: delay + "ms" }}>
      {dot && <span className="dot" style={{ background: dot }} />}
      <span className="k num">{k}</span>
      <span className="l">{l}</span>
    </div>
  );
}

// ---- BenchRow ----
interface BenchRowProps { label: string; value: number; target: number; max: number; unit?: string; color?: string; }
export function BenchRow({ label, value, target, max, unit, color }: BenchRowProps) {
  const pct = Math.min(100, (value / max) * 100);
  const tpct = Math.min(100, (target / max) * 100);
  const met = value >= target;
  return (
    <div className="bench-row">
      <span className="bench-label">{label}</span>
      <div className="bench-track">
        <div className="bench-fill" style={{ width: pct + "%", background: color || "var(--cobalt)" }} />
        <div className="bench-target" style={{ left: tpct + "%" }} />
      </div>
      <span className="bench-val">
        <b>{value}</b> / {target}{unit ? " " + unit : ""}{" "}
        <span style={{ color: met ? "var(--ok)" : "var(--warn)" }}>{met ? "✓" : "↑"}</span>
      </span>
    </div>
  );
}

// ---- MetricCard ----
interface MetricCardProps { label: string; value: string | number; unit?: string; sub?: string; subColor?: string; }
export function MetricCard({ label, value, unit, sub, subColor }: MetricCardProps) {
  return (
    <div className="metric-card">
      <div className="mc-label">{label}</div>
      <div className="mc-val">
        {value}<span className="mc-unit"> {unit}</span>
      </div>
      {sub && <div className="mc-sub" style={{ color: subColor }}>{sub}</div>}
    </div>
  );
}

// ---- CardToggle ----
interface CardToggleProps { on: boolean; onClick: () => void; title: string; sub: string; children?: ReactNode; }
export function CardToggle({ on, onClick, title, sub, children }: CardToggleProps) {
  return (
    <button className={"cardtoggle" + (on ? " on" : "")} onClick={onClick}>
      <span className="ct-check">
        {on && <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M4 12 l5 5 11-11" /></svg>}
      </span>
      <span className="ct-title">{title}</span>
      <span className="ct-sub">{sub}</span>
      {children}
    </button>
  );
}

// ---- Icons ----
type IconProps = { size?: number };
export const Icons = {
  preprocess: ({ size = 16 }: IconProps) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.3} strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 7 h18" /><path d="M3 12 h18" /><path d="M3 17 h18" />
      <path d="M8 3 v18" /><path d="M16 3 v18" />
    </svg>
  ),
  train: ({ size = 16 }: IconProps) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 19 V5" /><path d="M4 19 H20" /><path d="M7 16 l4-5 3 3 5-7" />
    </svg>
  ),
  inference: ({ size = 16 }: IconProps) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 5 a2 2 0 0 1 2-2 h14 a2 2 0 0 1 2 2 v14 a2 2 0 0 1-2 2 H5 a2 2 0 0 1-2-2 z" />
      <path d="M12 3 v18" />
    </svg>
  ),
  data: ({ size = 16 }: IconProps) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 3 c4.5 0 8 1.3 8 3 s-3.5 3-8 3-8-1.3-8-3 3.5-3 8-3 z" />
      <path d="M4 6 v12 c0 1.7 3.5 3 8 3 s8-1.3 8-3 V6" />
    </svg>
  ),
  upload: ({ size = 16 }: IconProps) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 16 V4" /><path d="M7 9 l5-5 5 5" /><path d="M4 20 h16" />
    </svg>
  ),
  play: ({ size = 16 }: IconProps) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor" stroke="none">
      <path d="M6 4 l14 8-14 8 z" />
    </svg>
  ),
  download: ({ size = 16 }: IconProps) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 4 v12" /><path d="M7 11 l5 5 5-5" /><path d="M4 20 h16" />
    </svg>
  ),
  check: ({ size = 16 }: IconProps) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 12 l5 5 11-11" />
    </svg>
  ),
  layers: ({ size = 16 }: IconProps) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 3 l9 5-9 5-9-5 z" /><path d="M3 13 l9 5 9-5" />
    </svg>
  ),
  settings: ({ size = 16 }: IconProps) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.2} strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 15 a3 3 0 1 0 0-6 3 3 0 0 0 0 6 z" />
      <path d="M19.4 15 a1.65 1.65 0 0 0 .33 1.82 l.06.06 a2 2 0 0 1-2.83 2.83 l-.06-.06 a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51 V21 a2 2 0 0 1-4 0 v-.09 A1.65 1.65 0 0 0 9 19.4 a1.65 1.65 0 0 0-1.82.33 l-.06.06 a2 2 0 0 1-2.83-2.83 l.06-.06 A1.65 1.65 0 0 0 4.68 15 1.65 1.65 0 0 0 3.17 14 H3 a2 2 0 0 1 0-4 h.09 A1.65 1.65 0 0 0 4.6 9 1.65 1.65 0 0 0 4.27 7.18 l-.06-.06 a2 2 0 0 1 2.83-2.83 l.06.06 A1.65 1.65 0 0 0 9 4.68 1.65 1.65 0 0 0 10 3.17 V3 a2 2 0 0 1 4 0 v.09 a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33 l.06-.06 a2 2 0 0 1 2.83 2.83 l-.06.06 A1.65 1.65 0 0 0 19.4 9 1.65 1.65 0 0 0 20.83 10 H21 a2 2 0 0 1 0 4 h-.09 a1.65 1.65 0 0 0-1.51 1 z" />
    </svg>
  ),
  doc: ({ size = 16 }: IconProps) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
      <path d="M6 2 h8 l4 4 v16 H6 z" /><path d="M14 2 v4 h4" />
    </svg>
  ),
  trash: ({ size = 16 }: IconProps) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.3} strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 6 h16" /><path d="M9 6 V4 h6 v2" /><path d="M6 6 l1 14 h10 l1-14" />
    </svg>
  ),
  spark: ({ size = 16 }: IconProps) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.2} strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 3 l2.2 6 6.3.3-5 4 1.8 6-5.3-3.6-5.3 3.6 1.8-6-5-4 6.3-.3 z" />
    </svg>
  ),
  cpu: ({ size = 16 }: IconProps) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.3} strokeLinecap="round" strokeLinejoin="round">
      <path d="M6 6 h12 v12 H6 z" />
      <path d="M9 1 v3 M15 1 v3 M9 20 v3 M15 20 v3 M1 9 h3 M1 15 h3 M20 9 h3 M20 15 h3" />
    </svg>
  ),
  stop: ({ size = 16 }: IconProps) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor" stroke="none">
      <rect x="5" y="5" width="14" height="14" rx="2" />
    </svg>
  ),
};
