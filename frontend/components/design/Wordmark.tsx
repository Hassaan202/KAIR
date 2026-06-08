interface WordmarkProps { compact?: boolean; light?: boolean; }

export default function Wordmark({ compact = false, light = false }: WordmarkProps) {
  return (
    <div className="wordmark" style={{ color: light ? "#fbf8f2" : "var(--ink)" }}>
      <svg width="26" height="26" viewBox="0 0 26 26" fill="none" aria-hidden>
        <circle cx="13" cy="13" r="8.4" stroke="currentColor" strokeWidth="1.3" opacity={0.85} />
        <ellipse cx="13" cy="13" rx="11.4" ry="4.6" stroke="var(--cobalt)"
          strokeWidth="1.3" transform="rotate(-26 13 13)" opacity={0.95} />
        <circle cx="22.1" cy="8.2" r="1.7" fill="var(--terracotta)" />
        <circle cx="13" cy="13" r="2.1" fill="currentColor" />
      </svg>
      {!compact && (
        <div className="wordmark-txt">
          <strong>SUPARCO</strong>
          <span>Super-Resolution Lab</span>
        </div>
      )}
    </div>
  );
}
