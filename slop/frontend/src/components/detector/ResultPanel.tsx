import { motion, AnimatePresence } from "framer-motion";
import { useEffect, useRef } from "react";
import gsap from "gsap";
import { AlertTriangle, Layers, ChevronDown } from "lucide-react";
import type { DetectionResult } from "../../types";
import { ScanningHUD } from "./ScanningHUD";
import { HeatmapOverlay } from "./HeatmapOverlay";
import { ScoreCard } from "./ScoreCard";
import { SentenceHighlight } from "./SentenceHighlight";
import { SignalGrid } from "./SignalGrid";
import { VideoTimeline } from "./VideoTimeline";
import {
  ProvenanceBadge,
  HeuristicOverrideNotice,
  AttributionCard,
} from "./ForensicBreakdowns";
import { LLMAttributionCard } from "./LLMAttributionCard";
import {
  ModelHint,
  DocumentSectionsHeatmap,
  DossierExportAction,
  CodeVerdictBadge,
  MixedAuthorshipBadge,
} from "./DocumentBreakdowns";
import { ThreeDForensicScanner } from "../ui/ThreeDForensicScanner";

function ThreeDForensicScannerFallback() {
  return <ThreeDForensicScanner size="sm" interactive={true} />;
}

function TechnicalSignalsDrawer({ scanResult }: { scanResult: DetectionResult }) {
  return (
    <details className="technical-signals-drawer">
      <summary className="technical-signals-summary">
        <div className="summary-left">
          <Layers size={14} className="summary-icon" />
          <span>Technical Signals &amp; Forensic Breakdown</span>
        </div>
        <ChevronDown size={14} className="summary-chevron" />
      </summary>
      <div className="technical-signals-content">
        <ProvenanceBadge signals={scanResult.signals} />
        <HeuristicOverrideNotice result={scanResult} />
        <LLMAttributionCard result={scanResult} />
        <AttributionCard result={scanResult} />
        <SignalGrid signals={scanResult.signals} />
      </div>
    </details>
  );
}

function EmptyState() {
  return (
    <motion.div
      key="empty"
      className="empty-state"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
    >
      <ThreeDForensicScannerFallback />
      <strong className="text-gray-300" style={{ marginTop: 12 }}>Awaiting Sample</strong>
      <span className="text-gray-500 text-sm">Upload an image, video, or paste text to initialize real-time neural detection</span>
    </motion.div>
  );
}

function ErrorState({ error }: { error: string }) {
  return (
    <motion.div
      key="error"
      className="empty-state error-state"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
    >
      <AlertTriangle size={32} className="text-red-400 mb-2" />
      <strong className="text-red-400">Analysis Error</strong>
      <span className="text-gray-400 text-sm">{error}</span>
      <span className="text-xs text-gray-500 mt-2">Please try submitting your sample again.</span>
    </motion.div>
  );
}

function ResultFlow({
  detectionResult,
  containerRef,
}: {
  detectionResult: DetectionResult;
  containerRef: React.RefObject<HTMLDivElement>;
}) {
  const isEditedCamera = Boolean(detectionResult.signals?.is_edited_camera_photo);
  const overrideExplanation =
    detectionResult.heuristic_override_explanation ||
    (detectionResult.signals?.heuristic_override_explanation as string | undefined);

  return (
    <div key="result" ref={containerRef} className="result-container-flow">
      <ScoreCard
        score={detectionResult.overall_score}
        confidence_range={detectionResult.confidence_range}
        confidence_band={detectionResult.confidence_band}
        isEditedCamera={isEditedCamera}
      />
      {overrideExplanation && (
        <div
          className="prominent-override-banner"
          style={{
            margin: "12px 0",
            padding: "12px 16px",
            borderRadius: "8px",
            background: isEditedCamera ? "rgba(16, 185, 129, 0.08)" : "rgba(59, 130, 246, 0.08)",
            border: isEditedCamera ? "1px solid rgba(16, 185, 129, 0.3)" : "1px solid rgba(59, 130, 246, 0.3)",
          }}
        >
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "8px",
              marginBottom: "4px",
              color: isEditedCamera ? "#34d399" : "#60a5fa",
              fontWeight: 600,
              fontSize: "0.85rem",
            }}
          >
            <span>{isEditedCamera ? "📷 Real Camera Photo (Edited / Color-Graded)" : "⚖️ Forensic Signal Reconciliation"}</span>
          </div>
          <p style={{ margin: 0, fontSize: "0.85rem", lineHeight: "1.4", color: "var(--text-secondary, #cbd5e1)" }}>
            {overrideExplanation}
          </p>
        </div>
      )}
      <CodeVerdictBadge codeResult={detectionResult} />
      <MixedAuthorshipBadge docResult={detectionResult} />
      <TechnicalSignalsDrawer scanResult={detectionResult} />
      <DocumentSectionsHeatmap sections={detectionResult.section_scores} />
      <HeatmapOverlay heatmap={detectionResult.heatmap_b64} gradcamHeatmap={detectionResult.gradcam_heatmap_b64} prnuHeatmap={detectionResult.prnu_heatmap_b64} />
      <VideoTimeline videoResult={detectionResult} />
      <SentenceHighlight sentences={detectionResult.sentences} />
      <ModelHint
        hint={detectionResult.model_hint}
        explanation={overrideExplanation}
      />
      <DossierExportAction jobId={detectionResult.job_id} />
    </div>
  );
}

interface ResultPanelProps {
  detectionResult?: DetectionResult;
  isAnalyzing?: boolean;
  error?: string;
  modality?: string;
}

export function ResultPanel({ detectionResult, isAnalyzing, error }: ResultPanelProps) {
  const resultContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!detectionResult || isAnalyzing || !resultContainerRef.current) return;

    const ctx = gsap.context(() => {
      const tl = gsap.timeline({ defaults: { ease: "power2.out" } });
      tl.from(".score-card", { y: 16, opacity: 0, duration: 0.4 })
        .from(".consolidated-verdict-wrap", { y: 8, opacity: 0, duration: 0.25 }, "-=0.2")
        .from(".technical-signals-drawer", { y: 12, opacity: 0, duration: 0.3 }, "-=0.15")
        .from(".heatmap, .video-timeline-panel, .sentence-highlights, .hint", {
          y: 12,
          opacity: 0,
          duration: 0.35,
          stagger: 0.08,
        }, "-=0.2");
    }, resultContainerRef);

    return () => ctx.revert();
  }, [detectionResult, isAnalyzing]);

  const renderActiveState = () => {
    if (isAnalyzing) {
      return (
        <motion.div
          key="loading"
          className="scanning-wrapper"
          initial={{ opacity: 0, scale: 0.98 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0 }}
        >
          <ScanningHUD />
        </motion.div>
      );
    }
    if (error) return <ErrorState error={error} />;
    if (detectionResult) return <ResultFlow detectionResult={detectionResult} containerRef={resultContainerRef} />;
    return <EmptyState />;
  };

  return (
    <aside className="result-panel">
      <AnimatePresence mode="wait">
        {renderActiveState()}
      </AnimatePresence>
    </aside>
  );
}
