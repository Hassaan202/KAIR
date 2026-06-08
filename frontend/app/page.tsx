"use client";
import { useRouter } from "next/navigation";
import dynamic from "next/dynamic";
import Wordmark from "@/components/design/Wordmark";
import OrbitTrack from "@/components/design/OrbitTrack";
import { StatPill } from "@/components/design/Primitives";
import FieldScene from "@/components/design/FieldScene";

const SplitCompare = dynamic(() => import("@/components/design/SplitCompare"), { ssr: false });

function ArrowIcon({ light }: { light?: boolean }) {
  return (
    <svg width="15" height="15" viewBox="0 0 15 15" fill="none"
      style={{ color: light ? "#fbfbf8" : "currentColor" }}>
      <path d="M3 7.5 H11 M7.5 4 L11 7.5 L7.5 11"
        stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export default function HeroPage() {
  const router = useRouter();

  const handleEnter = () => router.push("/platform/preprocessing");

  return (
    <div className="hero">
      {/* ---- Background ---- */}
      <div className="hero-bg">
        <div className="hero-plate">
          <div style={{ position: "absolute", inset: 0, filter: "saturate(.82) brightness(1.0)" }}>
            <FieldScene seed={42} region="delta" detail="hi" dense={2.6} />
          </div>
        </div>
        <div className="hero-wash" />
        <div className="hero-grat graticule" />
        <OrbitTrack />
      </div>

      {/* ---- Nav ---- */}
      <div className="hero-nav">
        <Wordmark />
        <div className="hero-nav-links">
          <a href="#">Platform</a>
          <a href="#">Research</a>
          <a href="#">Datasets</a>
          <button className="btn" onClick={handleEnter}>
            Enter platform <ArrowIcon />
          </button>
        </div>
      </div>

      {/* ---- Main ---- */}
      <div className="hero-main">
        <div className="eyebrow hero-eyebrow rise">SUPARCO · Satellite Image Super-Resolution</div>

        <h1 className="hero-h1 editorial rise" style={{ animationDelay: "60ms" }}>
          Seeing Earth,{"\n"}Clearly.
        </h1>

        <p className="hero-sub rise" style={{ animationDelay: "140ms" }}>
          A super-resolution research platform for SUPARCO — turning what the satellite captured
          into what the model reveals, one tile at a time.
        </p>

        <div className="hero-cta rise" style={{ animationDelay: "220ms" }}>
          <button className="btn btn-primary btn-lg" onClick={handleEnter}>
            Enter the platform <ArrowIcon light />
          </button>
          <button className="btn btn-lg btn-ghost">Read the method ↗</button>
        </div>

        <div className="hero-compare-wrap rise" style={{ animationDelay: "300ms" }}>
          <SplitCompare
            seed={9}
            region="delta"
            height={400}
            autoIntro
            labelL="LR · Sentinel-2 10m"
            labelR="SR · ×4 reconstruction"
          />
        </div>

        <div className="hero-pills">
          <StatPill k="4×" l="Upscaling" dot="var(--terracotta)" delay={200} />
          <StatPill k="46.58 dB" l="Peak PSNR" dot="var(--cobalt)" delay={400} />
          <StatPill k="SEN2VENµS" l="Trained" dot="var(--sage)" delay={600} />
        </div>

        <div className="hero-note">
          Indus Delta · scene 042 · 4× upscaling · model swinir-m · psnr 46.58 dB
        </div>
      </div>
    </div>
  );
}
