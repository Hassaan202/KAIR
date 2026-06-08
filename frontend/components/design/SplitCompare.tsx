"use client";
import { useRef, useState, useEffect } from "react";
import FieldScene from "./FieldScene";

interface SplitCompareProps {
  seed?: number;
  region?: string;
  labelL?: string;
  labelR?: string;
  autoIntro?: boolean;
  height?: number | string;
}

export default function SplitCompare({
  seed = 7,
  region = "farmland",
  labelL = "LR · sensor",
  labelR = "SR · ×4",
  autoIntro = true,
  height = 400,
}: SplitCompareProps) {
  const [pos, setPos] = useState(autoIntro ? 4 : 50);
  const ref = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);
  const introDone = useRef(!autoIntro);

  useEffect(() => {
    if (!autoIntro) return;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduce) { setPos(50); introDone.current = true; return; }
    const t = setTimeout(() => {
      const start = performance.now(), dur = 2200, from = 4, to = 62;
      const tick = (now: number) => {
        const k = Math.min(1, (now - start) / dur);
        const e = 1 - Math.pow(1 - k, 3);
        setPos(from + (to - from) * e);
        if (k < 1) requestAnimationFrame(tick);
        else introDone.current = true;
      };
      requestAnimationFrame(tick);
    }, 500);
    return () => clearTimeout(t);
  }, [autoIntro]);

  const move = (clientX: number) => {
    const el = ref.current; if (!el) return;
    const rect = el.getBoundingClientRect();
    setPos(Math.max(0, Math.min(100, ((clientX - rect.left) / rect.width) * 100)));
  };

  const onDown = (clientX: number) => {
    dragging.current = true;
    introDone.current = true;
    move(clientX);
  };

  useEffect(() => {
    const mv = (e: MouseEvent | TouchEvent) => {
      if (!dragging.current) return;
      const x = "touches" in e ? e.touches[0].clientX : e.clientX;
      move(x);
    };
    const up = () => { dragging.current = false; };
    window.addEventListener("mousemove", mv);
    window.addEventListener("mouseup", up);
    window.addEventListener("touchmove", mv, { passive: false });
    window.addEventListener("touchend", up);
    return () => {
      window.removeEventListener("mousemove", mv);
      window.removeEventListener("mouseup", up);
      window.removeEventListener("touchmove", mv);
      window.removeEventListener("touchend", up);
    };
  }, []);

  return (
    <div
      ref={ref}
      className="split-compare"
      style={{ height }}
      onMouseDown={(e) => onDown(e.clientX)}
      onTouchStart={(e) => onDown(e.touches[0].clientX)}
    >
      {/* SR layer — sharp, full */}
      <div className="split-layer">
        <FieldScene seed={seed} region={region} detail="hi" />
      </div>
      {/* LR layer — clipped to left, blurred */}
      <div className="split-layer lr" style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}>
        <FieldScene seed={seed} region={region} detail="lo" />
        <div className="lr-pixels" />
      </div>
      {/* Tags */}
      <span className="split-tag left" style={{ opacity: pos > 14 ? 1 : 0 }}>{labelL}</span>
      <span className="split-tag right" style={{ opacity: pos < 86 ? 1 : 0 }}>{labelR}</span>
      {/* Divider */}
      <div className="split-divider" style={{ left: `${pos}%` }}>
        <div className="split-handle">
          <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
            <path d="M8 6 L4 10 L8 14 M12 6 L16 10 L12 14"
              stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </div>
      </div>
    </div>
  );
}
