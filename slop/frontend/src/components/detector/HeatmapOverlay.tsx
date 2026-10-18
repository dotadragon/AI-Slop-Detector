import { useState } from "react";
import { Eye, Layers, Sparkles, Activity, Maximize2 } from "lucide-react";

type HeatmapMode = "side_by_side" | "gradcam" | "ela" | "prnu";

interface ChannelConfig {
  badgeClass: string;
  badgeLabel: string;
  icon: typeof Sparkles;
  note: string;
  singleTitle: string;
  singleDesc: string;
}

const CHANNELS: Record<string, ChannelConfig> = {
  gradcam: {
    badgeClass: "gradcam",
    badgeLabel: "Neural Attention (GradCAM)",
    icon: Sparkles,
    note: "Swin/ViT activation layer overlay highlighting synthetic focal regions.",
    singleTitle: "GradCAM Neural Attention Map",
    singleDesc: "Jet colormap overlay generated directly via native PyTorch activation hooks. Red/yellow zones demarcate regions that strongly contributed to the synthetic classification.",
  },
  ela: {
    badgeClass: "ela",
    badgeLabel: "Compression Divergence (ELA)",
    icon: Activity,
    note: "Re-quantization at Q=90 exposing macroblock compression variance.",
    singleTitle: "Error Level Analysis (ELA)",
    singleDesc: "Measures 8x8 DCT compression error levels between the original file and an intentional resave. Discontinuous bright clusters point to post-processing, generative inpainting, or composite pasting.",
  },
  prnu: {
    badgeClass: "prnu",
    badgeLabel: "PRNU Sensor Noise Floor",
    icon: Eye,
    note: "Sensor noise residual isolating diffusion harmonic patterns vs camera silicon grain.",
    singleTitle: "Photo-Response Non-Uniformity (PRNU) Residual Floor",
    singleDesc: "Gaussian-filtered residual extracting physical camera sensor noise signatures. Diffusion latent models generate subtle periodic grid harmonics and lower kurtosis compared to authentic CMOS/CCD sensors.",
  },
};

function HeatmapPanel({ type, b64 }: { type: "gradcam" | "ela" | "prnu"; b64: string }) {
  const config = CHANNELS[type];
  const Icon = config.icon;

  return (
    <div className="visualizer-panel">
      <div className={`panel-badge ${config.badgeClass}`}>
        <Icon size={12} />
        <span>{config.badgeLabel}</span>
      </div>
      <div className="visualizer-frame">
        <img src={`data:image/png;base64,${b64}`} alt={config.badgeLabel} className="visualizer-img" />
      </div>
      <p className="visualizer-note">{config.note}</p>
    </div>
  );
}

function HeatmapSingleView({ type, b64 }: { type: "gradcam" | "ela" | "prnu"; b64: string }) {
  const config = CHANNELS[type];

  return (
    <div className="visualizer-single-view">
      <div className="visualizer-frame single">
        <img src={`data:image/png;base64,${b64}`} alt={config.badgeLabel} className="visualizer-img" />
      </div>
      <div className="visualizer-meta-box">
        <strong>{config.singleTitle}</strong>
        <span>{config.singleDesc}</span>
      </div>
    </div>
  );
}

function HeatmapToggleGroup({
  mode,
  setMode,
  hasGradcam,
  hasEla,
  hasPrnu,
}: {
  mode: HeatmapMode;
  setMode: (m: HeatmapMode) => void;
  hasGradcam: boolean;
  hasEla: boolean;
  hasPrnu: boolean;
}) {
  return (
    <div className="heatmap-toggle-group" role="tablist" aria-label="Forensic visualizer modes">
      <button
        type="button"
        className={`visualizer-btn ${mode === "side_by_side" ? "active" : ""}`}
        onClick={() => setMode("side_by_side")}
      >
        <Maximize2 size={13} />
        <span>Side-by-Side</span>
      </button>
      {hasGradcam && (
        <button
          type="button"
          className={`visualizer-btn ${mode === "gradcam" ? "active" : ""}`}
          onClick={() => setMode("gradcam")}
        >
          <Sparkles size={13} />
          <span>GradCAM</span>
        </button>
      )}
      {hasEla && (
        <button
          type="button"
          className={`visualizer-btn ${mode === "ela" ? "active" : ""}`}
          onClick={() => setMode("ela")}
        >
          <Activity size={13} />
          <span>ELA</span>
        </button>
      )}
      {hasPrnu && (
        <button
          type="button"
          className={`visualizer-btn ${mode === "prnu" ? "active" : ""}`}
          onClick={() => setMode("prnu")}
        >
          <Eye size={13} />
          <span>PRNU</span>
        </button>
      )}
    </div>
  );
}

function HeatmapMultiGrid({
  gradcamHeatmap,
  heatmap,
  prnuHeatmap,
}: {
  gradcamHeatmap?: string;
  heatmap?: string;
  prnuHeatmap?: string;
}) {
  return (
    <div className="visualizer-grid-multi">
      {gradcamHeatmap && <HeatmapPanel type="gradcam" b64={gradcamHeatmap} />}
      {heatmap && <HeatmapPanel type="ela" b64={heatmap} />}
      {prnuHeatmap && <HeatmapPanel type="prnu" b64={prnuHeatmap} />}
    </div>
  );
}

function HeatmapContentView({
  mode,
  heatmap,
  gradcamHeatmap,
  prnuHeatmap,
}: {
  mode: HeatmapMode;
  heatmap?: string;
  gradcamHeatmap?: string;
  prnuHeatmap?: string;
}) {
  if (mode === "side_by_side") {
    return (
      <HeatmapMultiGrid
        gradcamHeatmap={gradcamHeatmap}
        heatmap={heatmap}
        prnuHeatmap={prnuHeatmap}
      />
    );
  }
  if (mode === "gradcam" && gradcamHeatmap) {
    return <HeatmapSingleView type="gradcam" b64={gradcamHeatmap} />;
  }
  if (mode === "ela" && heatmap) {
    return <HeatmapSingleView type="ela" b64={heatmap} />;
  }
  if (mode === "prnu" && prnuHeatmap) {
    return <HeatmapSingleView type="prnu" b64={prnuHeatmap} />;
  }
  return null;
}

function resolveDefaultMode(count: number, hasGradcam: boolean, hasEla: boolean): HeatmapMode {
  if (count > 1) return "side_by_side";
  if (hasGradcam) return "gradcam";
  if (hasEla) return "ela";
  return "prnu";
}

export function HeatmapOverlay({
  heatmap,
  gradcamHeatmap,
  prnuHeatmap,
}: {
  heatmap?: string;
  gradcamHeatmap?: string;
  prnuHeatmap?: string;
}) {
  const hasEla = Boolean(heatmap);
  const hasGradcam = Boolean(gradcamHeatmap);
  const hasPrnu = Boolean(prnuHeatmap);
  const availableCount = [hasEla, hasGradcam, hasPrnu].filter(Boolean).length;

  const [mode, setMode] = useState<HeatmapMode>(() =>
    resolveDefaultMode(availableCount, hasGradcam, hasEla)
  );

  if (availableCount === 0) return null;

  return (
    <div className="forensic-visualizer-card">
      <div className="visualizer-header">
        <div className="visualizer-title">
          <Layers size={16} className="text-cyan-400" />
          <span>MULTI-SPECTRAL FORENSIC IMAGING</span>
          <span className="visualizer-tag">{availableCount} CHANNELS ACTIVE</span>
        </div>

        {availableCount > 1 && (
          <HeatmapToggleGroup
            mode={mode}
            setMode={setMode}
            hasGradcam={hasGradcam}
            hasEla={hasEla}
            hasPrnu={hasPrnu}
          />
        )}
      </div>

      <HeatmapContentView
        mode={mode}
        heatmap={heatmap}
        gradcamHeatmap={gradcamHeatmap}
        prnuHeatmap={prnuHeatmap}
      />
    </div>
  );
}
