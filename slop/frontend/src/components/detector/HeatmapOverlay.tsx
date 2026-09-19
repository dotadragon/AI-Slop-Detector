/**
 * v3.4: HeatmapOverlay — Multi-spectral cyber-forensic visualizer.
 * Supports side-by-side comparative inspection across:
 * 1. GradCAM Neural Attention (regions the vision transformer flagged as synthetic)
 * 2. ELA Error Level Analysis (JPEG compression artifact divergence)
 * 3. PRNU Sensor Noise Floor (silicon camera sensor imperfections vs diffusion latents)
 */
import { useState } from "react";
import { Eye, Layers, Sparkles, Activity, Maximize2 } from "lucide-react";

type HeatmapMode = "side_by_side" | "gradcam" | "ela" | "prnu";

export function HeatmapOverlay({
  heatmap,
  gradcamHeatmap,
  prnuHeatmap,
}: {
  heatmap?: string;
  gradcamHeatmap?: string;
  prnuHeatmap?: string;
}) {
  const hasEla = !!heatmap;
  const hasGradcam = !!gradcamHeatmap;
  const hasPrnu = !!prnuHeatmap;

  // Default to side-by-side if multiple visualizers available, else primary
  const availableCount = [hasEla, hasGradcam, hasPrnu].filter(Boolean).length;
  const defaultMode: HeatmapMode = availableCount > 1 ? "side_by_side" : hasGradcam ? "gradcam" : hasEla ? "ela" : "prnu";

  const [mode, setMode] = useState<HeatmapMode>(defaultMode);

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
                <span>GradCAM Attention</span>
              </button>
            )}
            {hasEla && (
              <button
                type="button"
                className={`visualizer-btn ${mode === "ela" ? "active" : ""}`}
                onClick={() => setMode("ela")}
              >
                <Activity size={13} />
                <span>ELA Compression</span>
              </button>
            )}
            {hasPrnu && (
              <button
                type="button"
                className={`visualizer-btn ${mode === "prnu" ? "active" : ""}`}
                onClick={() => setMode("prnu")}
              >
                <Eye size={13} />
                <span>PRNU Sensor Residual</span>
              </button>
            )}
          </div>
        )}
      </div>

      {/* Side-by-Side Multi-Inspector Grid */}
      {mode === "side_by_side" && (
        <div className="visualizer-grid-multi">
          {hasGradcam && (
            <div className="visualizer-panel">
              <div className="panel-badge gradcam">
                <Sparkles size={12} />
                <span>Neural Attention (GradCAM)</span>
              </div>
              <div className="visualizer-frame">
                <img
                  src={`data:image/png;base64,${gradcamHeatmap}`}
                  alt="GradCAM neural attention activation heatmap"
                  className="visualizer-img"
                />
              </div>
              <p className="visualizer-note">
                Swin/ViT activation layer overlay highlighting synthetic focal regions.
              </p>
            </div>
          )}

          {hasEla && (
            <div className="visualizer-panel">
              <div className="panel-badge ela">
                <Activity size={12} />
                <span>Compression Divergence (ELA)</span>
              </div>
              <div className="visualizer-frame">
                <img
                  src={`data:image/png;base64,${heatmap}`}
                  alt="Error level analysis compression heatmap"
                  className="visualizer-img"
                />
              </div>
              <p className="visualizer-note">
                Re-quantization at Q=90 exposing macroblock compression variance.
              </p>
            </div>
          )}

          {hasPrnu && (
            <div className="visualizer-panel">
              <div className="panel-badge prnu">
                <Eye size={12} />
                <span>PRNU Sensor Noise Floor</span>
              </div>
              <div className="visualizer-frame">
                <img
                  src={`data:image/png;base64,${prnuHeatmap}`}
                  alt="Photo-response non-uniformity sensor noise residual map"
                  className="visualizer-img"
                />
              </div>
              <p className="visualizer-note">
                Sensor noise residual isolating diffusion harmonic patterns vs camera silicon grain.
              </p>
            </div>
          )}
        </div>
      )}

      {/* Single View: GradCAM */}
      {mode === "gradcam" && hasGradcam && (
        <div className="visualizer-single-view">
          <div className="visualizer-frame single">
            <img
              src={`data:image/png;base64,${gradcamHeatmap}`}
              alt="GradCAM neural attention activation heatmap"
              className="visualizer-img"
            />
          </div>
          <div className="visualizer-meta-box">
            <strong>GradCAM Neural Attention Map</strong>
            <span>Jet colormap overlay generated directly via native PyTorch activation hooks. Red/yellow zones demarcate regions that strongly contributed to the synthetic classification.</span>
          </div>
        </div>
      )}

      {/* Single View: ELA */}
      {mode === "ela" && hasEla && (
        <div className="visualizer-single-view">
          <div className="visualizer-frame single">
            <img
              src={`data:image/png;base64,${heatmap}`}
              alt="Error level analysis compression heatmap"
              className="visualizer-img"
            />
          </div>
          <div className="visualizer-meta-box">
            <strong>Error Level Analysis (ELA)</strong>
            <span>Measures 8x8 DCT compression error levels between the original file and an intentional resave. Discontinuous bright clusters point to post-processing, generative inpainting, or composite pasting.</span>
          </div>
        </div>
      )}

      {/* Single View: PRNU */}
      {mode === "prnu" && hasPrnu && (
        <div className="visualizer-single-view">
          <div className="visualizer-frame single">
            <img
              src={`data:image/png;base64,${prnuHeatmap}`}
              alt="Photo-response non-uniformity sensor noise residual map"
              className="visualizer-img"
            />
          </div>
          <div className="visualizer-meta-box">
            <strong>Photo-Response Non-Uniformity (PRNU) Residual Floor</strong>
            <span>Gaussian-filtered residual extracting physical camera sensor noise signatures. Diffusion latent models generate subtle periodic grid harmonics and lower kurtosis compared to authentic CMOS/CCD sensors.</span>
          </div>
        </div>
      )}
    </div>
  );
}
