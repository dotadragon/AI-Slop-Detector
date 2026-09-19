import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ShieldCheck, Sparkles, Cpu, Eye } from "lucide-react";
import type { BoundingBox, DigitalSignatureInfo } from "../../types";

interface Props {
  boundingBox?: BoundingBox;
  signature?: DigitalSignatureInfo;
}

export function SignatureBoundingBox({ boundingBox, signature }: Props) {
  const [showTooltip, setShowTooltip] = useState(true);

  if (!boundingBox || !signature?.detected) {
    return null;
  }

  const { ymin, xmin, ymax, xmax } = boundingBox;
  const topPct = `${Math.max(0, Math.min(100, ymin * 100))}%`;
  const leftPct = `${Math.max(0, Math.min(100, xmin * 100))}%`;
  const widthPct = `${Math.max(5, Math.min(100, (xmax - xmin) * 100))}%`;
  const heightPct = `${Math.max(5, Math.min(100, (ymax - ymin) * 100))}%`;

  return (
    <div
      className="signature-bounding-box-container"
      style={{
        position: "absolute",
        top: topPct,
        left: leftPct,
        width: widthPct,
        height: heightPct,
        pointerEvents: "auto",
        zIndex: 20,
      }}
      onClick={() => setShowTooltip((prev) => !prev)}
      title="Click to toggle signature details"
    >
      {/* Neon Green Bounding Box Reticle */}
      <motion.div
        className="signature-box"
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.4, ease: "easeOut" }}
      >
        {/* Corner Reticle Accents */}
        <span className="reticle-corner top-left" />
        <span className="reticle-corner top-right" />
        <span className="reticle-corner bottom-left" />
        <span className="reticle-corner bottom-right" />

        {/* Header Label Badge */}
        <div className="signature-tag-badge">
          <ShieldCheck size={12} className="tag-icon" />
          <span className="tag-text">
            {signature.signature_type.toUpperCase()}
          </span>
          <span className="tag-conf">
            {Math.round(signature.confidence * 100)}%
          </span>
        </div>

        {/* Pulse center reticle crosshair */}
        <div className="reticle-crosshair" />
      </motion.div>

      {/* Floating Forensic Tooltip */}
      <AnimatePresence>
        {showTooltip && (
          <motion.div
            className="signature-tooltip"
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 6 }}
          >
            <div className="tooltip-header">
              <Sparkles size={14} className="text-emerald-400" />
              <span>Digital Watermark Localized</span>
            </div>
            <div className="tooltip-body">
              <div className="tooltip-row">
                <span className="tooltip-label">Type:</span>
                <span className="tooltip-val">{signature.signature_type}</span>
              </div>
              <div className="tooltip-row">
                <span className="tooltip-label">Attributed AI:</span>
                <span className="tooltip-val highlight">
                  <Cpu size={12} className="inline mr-1" />
                  {signature.generator}
                </span>
              </div>
              <div className="tooltip-row">
                <span className="tooltip-label">Coordinates:</span>
                <span className="tooltip-val mono">
                  [{Math.round(xmin * 100)}%, {Math.round(ymin * 100)}%] to [{Math.round(xmax * 100)}%, {Math.round(ymax * 100)}%]
                </span>
              </div>
              {signature.details && (
                <div className="tooltip-details">{signature.details}</div>
              )}
            </div>
            <div className="tooltip-footer">
              <Eye size={11} className="inline mr-1" />
              Click box to toggle details
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
