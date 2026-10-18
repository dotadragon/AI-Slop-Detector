import type { SpectralLayer } from "./spectralLayers";

function Layer01Graphic() {
  return (
    <g className="canvas-layer-anim">
      <circle cx="300" cy="170" r="80" fill="none" stroke="#ff7700" strokeWidth="1.5" strokeDasharray="6 3" />
      <circle cx="300" cy="170" r="50" fill="none" stroke="#ffaa00" strokeWidth="1" />
      <line x1="300" y1="60" x2="300" y2="280" stroke="rgba(255, 119, 0, 0.3)" strokeWidth="1" />
      <line x1="190" y1="170" x2="410" y2="170" stroke="rgba(255, 119, 0, 0.3)" strokeWidth="1" />
      <rect x="220" y="90" width="160" height="160" fill="none" stroke="#ff7700" strokeWidth="1.5" />
      <text x="300" y="174" textAnchor="middle" fill="#ffaa00" fontSize="11" fontFamily="var(--mono)">
        C2PA JUMBF MANIFEST : VALIDATED
      </text>
    </g>
  );
}

function Layer02Graphic() {
  return (
    <g className="canvas-layer-anim">
      {Array.from({ length: 24 }).map((_, dotIndex) => (
        <circle
          key={dotIndex}
          cx={160 + (dotIndex % 6) * 56 + ((dotIndex * 17) % 19)}
          cy={90 + Math.floor(dotIndex / 6) * 52 + ((dotIndex * 23) % 15)}
          r={1.5 + (dotIndex % 3) * 0.8}
          fill="#10b981"
          opacity={0.35 + (dotIndex % 5) * 0.12}
        />
      ))}
      <rect x="180" y="110" width="240" height="120" rx="8" fill="rgba(16, 185, 129, 0.08)" stroke="#10b981" strokeWidth="1.5" />
      <text x="300" y="165" textAnchor="middle" fill="#10b981" fontSize="12" fontFamily="var(--mono)">
        CMOS SENSOR RESIDUAL: 0.0024 σ²
      </text>
      <text x="300" y="188" textAnchor="middle" fill="#94a3b8" fontSize="10" fontFamily="var(--mono)">
        PHYSICAL SENSOR FINGERPRINT VERIFIED
      </text>
    </g>
  );
}

function Layer03Graphic() {
  return (
    <g className="canvas-layer-anim">
      <circle cx="300" cy="170" r="110" fill="none" stroke="rgba(6, 182, 212, 0.4)" strokeWidth="1.5" />
      <circle cx="300" cy="170" r="75" fill="none" stroke="#06b6d4" strokeWidth="1.5" strokeDasharray="4 4" />
      <circle cx="300" cy="170" r="40" fill="none" stroke="#ffaa00" strokeWidth="2" />
      <circle cx="300" cy="170" r="5" fill="#ffaa00" />
      <text x="300" y="270" textAnchor="middle" fill="#06b6d4" fontSize="11" fontFamily="var(--mono)">
        AZIMUTHAL POWER LAW: 1/f DECAY (NATURAL)
      </text>
    </g>
  );
}

function Layer04Graphic() {
  return (
    <g className="canvas-layer-anim">
      <circle cx="270" cy="160" r="95" fill="url(#gradcam-salience)" />
      <circle cx="360" cy="180" r="70" fill="url(#gradcam-salience)" opacity="0.7" />
      <rect x="190" y="85" width="160" height="150" fill="none" stroke="#f43f5e" strokeWidth="1.5" strokeDasharray="4 2" />
      <text x="300" y="295" textAnchor="middle" fill="#f43f5e" fontSize="11" fontFamily="var(--mono)">
        CROSS-ATTENTION SALIENCE CONTOUR MAPPED
      </text>
    </g>
  );
}

export function InspectorCanvasGraphic({
  activeLayer,
  inspectorProgress,
  currentLayer,
}: {
  activeLayer: number;
  inspectorProgress: number;
  currentLayer: SpectralLayer;
}) {
  return (
    <div className="inspector-canvas-box">
      <div className="canvas-status-bar">
        <div className="canvas-status-left">
          <span className="canvas-live-dot" />
          <span className="canvas-channel-label">{currentLayer.channel}</span>
        </div>
        <div className="canvas-status-right">
          <span className="canvas-res-label">512 × 512 TENSOR MATRIX</span>
        </div>
      </div>

      <div className="canvas-viewport" aria-label={`Forensic layer visualization: ${currentLayer.title}`}>
        <svg className="canvas-svg" viewBox="0 0 600 340" preserveAspectRatio="xMidYMid meet">
          <defs>
            <pattern id="canvas-grid" width="30" height="30" patternUnits="userSpaceOnUse">
              <path d="M 30 0 L 0 0 0 30" fill="none" stroke="rgba(255, 255, 255, 0.04)" strokeWidth="1" />
            </pattern>
            <radialGradient id="gradcam-salience" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="#ff4500" stopOpacity="0.85" />
              <stop offset="45%" stopColor="#ffaa00" stopOpacity="0.45" />
              <stop offset="100%" stopColor="transparent" stopOpacity="0" />
            </radialGradient>
          </defs>

          <rect width="100%" height="100%" fill="url(#canvas-grid)" />
          {activeLayer === 0 && <Layer01Graphic />}
          {activeLayer === 1 && <Layer02Graphic />}
          {activeLayer === 2 && <Layer03Graphic />}
          {activeLayer === 3 && <Layer04Graphic />}
        </svg>

        <div
          className="canvas-scan-beam"
          style={{ transform: `translateY(${Math.round((inspectorProgress * 280) % 300)}px)` }}
          aria-hidden="true"
        />
      </div>
    </div>
  );
}
