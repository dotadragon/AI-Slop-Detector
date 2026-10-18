import { Link } from "react-router-dom";
import { type RefObject } from "react";
import { Radio, ChevronRight, CheckCircle2 } from "lucide-react";
import { track } from "../../utils/analytics";
import { SPECTRAL_LAYERS, type SpectralLayer } from "./spectralLayers";
import { InspectorCanvasGraphic } from "./InspectorGraphics";

interface ForensicInspectorProps {
  inspectorRef: RefObject<HTMLElement>;
  activeLayer: number;
  inspectorProgress: number;
  handleSelectStage: (idx: number, layerId: string) => void;
}

function InspectorHudPanel({ currentLayer }: { currentLayer: SpectralLayer }) {
  return (
    <div className="inspector-hud-panel">
      <div className="hud-header">
        <span className="hud-title-tag">FORENSIC TELEMETRY</span>
        <span className="hud-confidence-badge">
          <CheckCircle2 size={13} color="#10b981" />
          LIVE PIPELINE STATUS
        </span>
      </div>

      <div className="hud-desc">{currentLayer.description}</div>

      <div className="hud-metrics-list">
        {currentLayer.metrics.map((metric, idx) => (
          <div key={idx} className="hud-metric-row">
            <span className="hud-metric-name">{metric.label}</span>
            <span className={`hud-metric-val ${metric.status}`}>{metric.val}</span>
          </div>
        ))}
      </div>

      <div className="hud-terminal-log">
        <div className="hud-terminal-bar">
          <span>PYTORCH PIPELINE STREAM</span>
          <span className="terminal-live-tag">CUDA ACTIVE</span>
        </div>
        {currentLayer.telemetryLog.map((line, idx) => (
          <div key={idx} className="hud-log-line">
            {line}
          </div>
        ))}
      </div>

      <Link
        to={`/detector?tab=${currentLayer.detectorModality}&layer=${currentLayer.id}`}
        className="hud-action-btn"
        onClick={() => track("cta_clicked", { section: "inspector", layer: currentLayer.id })}
      >
        <span>Verify Content With This Layer</span>
        <ChevronRight size={16} />
      </Link>
    </div>
  );
}

function StepperDotItem({
  layer,
  idx,
  active,
  onClick,
}: {
  layer: SpectralLayer;
  idx: number;
  active: boolean;
  onClick: (idx: number, id: string) => void;
}) {
  return (
    <button
      type="button"
      role="tab"
      aria-selected={active}
      aria-label={`Stage ${layer.number}: ${layer.title}.`}
      className={`stepper-dot-item ${active ? "active" : ""}`}
      onClick={() => onClick(idx, layer.id)}
      tabIndex={0}
    >
      <div className="stepper-dot-disc">
        <span className="stepper-dot-digit">{layer.number}</span>
        {active && <span className="stepper-dot-halo" aria-hidden="true" />}
      </div>
      <div className="stepper-dot-caption">
        <span className="stepper-dot-num-tag">STAGE {layer.number}</span>
        <span className="stepper-dot-tag-text">{layer.tag}</span>
      </div>
    </button>
  );
}

function InspectorStepper({
  activeLayer,
  inspectorProgress,
  handleSelectStage,
}: {
  activeLayer: number;
  inspectorProgress: number;
  handleSelectStage: (idx: number, layerId: string) => void;
}) {
  const currentLayer = SPECTRAL_LAYERS[activeLayer];
  const lineFillRatio = Math.max(activeLayer / 3, Math.min(1, inspectorProgress * 1.08));

  return (
    <div className="inspector-stepper-wrap" aria-label="Forensic Inspection Stepper">
      <div className="stepper-track-outer">
        <svg className="stepper-dots-track-bg" viewBox="0 0 1000 28" preserveAspectRatio="none" aria-hidden="true">
          <line x1="125" y1="14" x2="875" y2="14" stroke="var(--border-hi, rgba(255, 255, 255, 0.12))" strokeWidth="3" strokeLinecap="round" />
          <line
            className="stepper-dots-track-fill"
            x1="125"
            y1="14"
            x2={125 + lineFillRatio * 750}
            y2="14"
            stroke="url(#stepper-track-gradient)"
            strokeWidth="3.5"
            strokeLinecap="round"
            style={{ transition: "x2 0.35s cubic-bezier(0.16, 1, 0.3, 1)" }}
          />
          <line
            className="stepper-shimmer-line"
            x1="125"
            y1="14"
            x2={125 + lineFillRatio * 750}
            y2="14"
            strokeWidth="3.5"
            strokeLinecap="round"
            style={{ transition: "x2 0.35s cubic-bezier(0.16, 1, 0.3, 1)" }}
          />
          <defs>
            <linearGradient id="stepper-track-gradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#10b981" />
              <stop offset="50%" stopColor="#34d399" />
              <stop offset="100%" stopColor="#10b981" />
            </linearGradient>
            <linearGradient id="stepper-shimmer-gradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="transparent" stopOpacity="0" />
              <stop offset="50%" stopColor="#ffffff" stopOpacity="0.85" />
              <stop offset="100%" stopColor="transparent" stopOpacity="0" />
            </linearGradient>
          </defs>
        </svg>

        <div className="stepper-dots-row" role="tablist" aria-label="Inspection Stages">
          {SPECTRAL_LAYERS.map((layer, idx) => (
            <StepperDotItem
              key={layer.id}
              layer={layer}
              idx={idx}
              active={activeLayer === idx}
              onClick={handleSelectStage}
            />
          ))}
        </div>
      </div>

      <div className="inspector-stepper-meta">
        <span className="stepper-stage-title">STAGE {activeLayer + 1} OF 4 : {currentLayer.title.toUpperCase()}</span>
        <div className="stepper-meta-controls">
          <span className="stepper-pct-badge">{`${Math.round(((activeLayer + 1) / 4) * 100)}% INSPECTED`}</span>
        </div>
      </div>
    </div>
  );
}

export function ForensicInspector({
  inspectorRef,
  activeLayer,
  inspectorProgress,
  handleSelectStage,
}: ForensicInspectorProps) {
  const currentLayer = SPECTRAL_LAYERS[activeLayer];

  return (
    <section className="inspector-section-outer" ref={inspectorRef} aria-label="Multi-Spectral Forensic Inspector">
      <div className="inspector-sticky-wrapper">
        <div className="inspector-spine-link" aria-hidden="true">
          <div className="inspector-spine-line" />
          <div className="inspector-pulse-node" />
        </div>

        <div className="inspector-header">
          <div className="inspector-badge">
            <Radio size={13} className="inspector-radio-icon" />
            <span>REAL-TIME MULTI-SPECTRAL DECOMPOSITION</span>
          </div>
          <h2 className="inspector-title">
            Deconstruct Synthetic Artifacts Across 4 Forensic Layers<span className="hero-amber-dot">.</span>
          </h2>
          <p className="inspector-subtitle">
            As content enters the pipeline, SLOP strips layers in real time — from cryptographic provenance to silicon sensor noise and cross-attention gradients.
          </p>
        </div>

        <InspectorStepper
          activeLayer={activeLayer}
          inspectorProgress={inspectorProgress}
          handleSelectStage={handleSelectStage}
        />

        <div className="inspector-stage-container">
          <InspectorCanvasGraphic activeLayer={activeLayer} inspectorProgress={inspectorProgress} currentLayer={currentLayer} />
          <InspectorHudPanel currentLayer={currentLayer} />
        </div>
      </div>
    </section>
  );
}
