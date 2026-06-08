export default function OrbitTrack() {
  return (
    <svg
      className="orbit-track"
      viewBox="0 0 1440 800"
      preserveAspectRatio="xMidYMid slice"
      aria-hidden
    >
      <path
        id="gt"
        className="orbit-path"
        d="M -40 540 C 260 360, 520 300, 760 360 S 1240 520, 1500 380"
        fill="none"
        stroke="var(--cobalt)"
        strokeWidth="1.1"
        opacity={0.4}
      />
      <path
        d="M -40 620 C 300 470, 640 460, 900 520 S 1320 600, 1500 500"
        fill="none"
        stroke="var(--terracotta)"
        strokeWidth="1"
        opacity={0.22}
        strokeDasharray="4 6"
      />
      <g>
        <path d="M -5 0 H 5 M 0 -5 V 5" stroke="var(--cobalt-deep)" strokeWidth="1.4" />
        <circle r="2.2" fill="none" stroke="var(--cobalt-deep)" strokeWidth="1" opacity={0.7} />
        <animateMotion dur="14s" repeatCount="indefinite" rotate="auto">
          <mpath href="#gt" />
        </animateMotion>
      </g>
    </svg>
  );
}
