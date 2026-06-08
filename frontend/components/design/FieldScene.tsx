"use client";
import { useMemo } from "react";

// Region palettes from real satellite tones (from components.jsx)
const REGIONS: Record<string, { name: string; cols: number; rows: number; pal: string[] }> = {
  delta:    { name: "Indus Delta",       cols: 9,  rows: 9,  pal: ["#4f7196","#5a7a9e","#6f8a5b","#7e8b66","#a9a47e","#cdd4bd","#3d5a7a","#8aa0a8"] },
  farmland: { name: "Punjab Farmland",   cols: 11, rows: 9,  pal: ["#7e8b66","#6f8a5b","#9aa177","#c0a06a","#c0764a","#b9966a","#cdd4bd","#a9a47e"] },
  urban:    { name: "Lahore Urban Grid", cols: 13, rows: 10, pal: ["#b9966a","#a9a47e","#9c8f7a","#c0764a","#8c8475","#cdc2a8","#7e8b66","#b08a64"] },
  mountain: { name: "Karakoram Range",   cols: 9,  rows: 9,  pal: ["#b9966a","#c0a06a","#cdc2a8","#a98f6a","#8c7a5e","#e7e0cf","#9aa177","#b08a64"] },
};

export { REGIONS };

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function buildScene(seed: number, regionKey: string, dense = 1) {
  const base = REGIONS[regionKey] || REGIONS.farmland;
  const r = { ...base, cols: Math.round(base.cols * dense), rows: Math.round(base.rows * dense) };
  const rng = mulberry32(seed);
  const cells: Array<{ x: number; y: number; span: number; c: string; rot: number }> = [];
  for (let y = 0; y < r.rows; y++) {
    for (let x = 0; x < r.cols; x++) {
      const span = rng() > 0.84 ? 2 : 1;
      cells.push({ x, y, span, c: r.pal[Math.floor(rng() * r.pal.length)], rot: (rng() - 0.5) * 4 });
    }
  }
  const path: [number, number][] = [];
  let px = rng();
  for (let y = 0; y <= r.rows; y++) {
    px = Math.max(0.05, Math.min(0.95, px + (rng() - 0.5) * 0.28));
    path.push([px * 100, (y / r.rows) * 100]);
  }
  return { r, cells, path, water: regionKey === "delta" };
}

interface FieldSceneProps {
  seed: number;
  region: string;
  detail?: "hi" | "lo";
  dense?: number;
}

export default function FieldScene({ seed, region, detail = "hi", dense = 1 }: FieldSceneProps) {
  const scene = useMemo(() => buildScene(seed, region, dense), [seed, region, dense]);
  const { r, cells, path } = scene;
  const cw = 100 / r.cols;
  const ch = 100 / r.rows;
  const d = "M " + path.map((p) => `${p[0]} ${p[1]}`).join(" L ");

  return (
    <div style={{ position: "absolute", inset: 0, overflow: "hidden" }}>
      {cells.map((c, i) => (
        <div
          key={i}
          style={{
            position: "absolute",
            left: `${c.x * cw}%`,
            top: `${c.y * ch}%`,
            width: `${cw * c.span + 0.4}%`,
            height: `${ch * c.span + 0.4}%`,
            background: c.c,
            transform: detail === "hi" ? `rotate(${c.rot}deg)` : "none",
          }}
        />
      ))}
      <svg
        viewBox="0 0 100 100"
        preserveAspectRatio="none"
        style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}
      >
        <path
          d={d}
          fill="none"
          stroke={scene.water ? "#3d5a7a" : "#8c7a5e"}
          strokeWidth={scene.water ? 2.4 : 1.1}
          strokeLinejoin="round"
          strokeLinecap="round"
          opacity={0.8}
        />
      </svg>
      {detail === "hi" && (
        <div
          className="graticule-fine"
          style={{ position: "absolute", inset: 0, opacity: 0.5, backgroundSize: "12px 12px", mixBlendMode: "multiply" }}
        />
      )}
    </div>
  );
}
