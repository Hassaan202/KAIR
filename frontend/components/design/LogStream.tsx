"use client";
import { useEffect, useRef } from "react";
import type { LogLine } from "@/lib/types";

interface LogStreamProps {
  lines: LogLine[];
  running?: boolean;
  height?: number;
}

export default function LogStream({ lines, running = false, height = 220 }: LogStreamProps) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [lines]);

  return (
    <div ref={ref} className="logstream scroll" style={{ height }}>
      {lines.map((l, i) => (
        <div key={i} className="ln">
          <span className="ts">{l.ts}</span>
          <span className={`lv-${l.lv || "info"}`}>
            {l.lv === "step" ? "▸ " : ""}{l.text}
          </span>
        </div>
      ))}
      {running && (
        <div className="ln">
          <span className="ts">&nbsp;</span>
          <span className="cursor" />
        </div>
      )}
    </div>
  );
}
