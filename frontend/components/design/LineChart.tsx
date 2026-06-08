interface Series { data: [number, number][]; color: string; dim?: boolean; }
interface LineChartProps { series: Series[]; height?: number; xLabel?: string; }

export default function LineChart({ series, height = 200, xLabel }: LineChartProps) {
  const W = 560, H = height;
  const pad = { l: 38, r: 14, t: 14, b: 26 };
  const all = series.flatMap((s) => s.data);
  if (!all.length) return <div style={{ height }} />;

  const xs = all.map((d) => d[0]), ys = all.map((d) => d[1]);
  const xmin = Math.min(...xs), xmax = Math.max(...xs, 1);
  const ymin = Math.min(...ys), ymax = Math.max(...ys);
  const sx = (x: number) => pad.l + ((x - xmin) / (xmax - xmin || 1)) * (W - pad.l - pad.r);
  const sy = (y: number) => H - pad.b - ((y - ymin) / (ymax - ymin || 1)) * (H - pad.t - pad.b);
  const ticks = 4;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height }}>
      {Array.from({ length: ticks + 1 }).map((_, i) => {
        const y = pad.t + (i / ticks) * (H - pad.t - pad.b);
        const v = ymax - (i / ticks) * (ymax - ymin);
        return (
          <g key={i}>
            <line x1={pad.l} y1={y} x2={W - pad.r} y2={y} stroke="var(--line)" strokeWidth="1" />
            <text x={pad.l - 7} y={y + 3} textAnchor="end" fontSize="9"
              fontFamily="var(--font-mono)" fill="var(--ink-3)">
              {v.toFixed(v < 10 ? 1 : 0)}
            </text>
          </g>
        );
      })}
      {series.map((s, si) => {
        const d = s.data.map((p, i) => `${i ? "L" : "M"} ${sx(p[0])} ${sy(p[1])}`).join(" ");
        return (
          <path key={si} d={d} fill="none" stroke={s.color}
            strokeWidth="1.8" strokeLinejoin="round" strokeLinecap="round"
            opacity={s.dim ? 0.5 : 1} />
        );
      })}
      {xLabel && (
        <text x={W / 2} y={H - 4} textAnchor="middle" fontSize="9"
          fontFamily="var(--font-mono)" fill="var(--ink-3)">{xLabel}</text>
      )}
    </svg>
  );
}
