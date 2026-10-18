import { useEffect, useRef, useState, useCallback } from "react";
import { useSearchParams } from "react-router-dom";
import type { DetectorMode, DetectionResult } from "../types";
import { ResultPanel } from "../components/detector/ResultPanel";
import { ScanHistory } from "../components/detector/ScanHistory";
import { TabBar } from "../components/detector/TabBar";
import { DetectorInputSwitcher } from "../components/detector/DetectorModalityPanes";
import { useDetection } from "../hooks/useDetection";
import { useScanHistory } from "../hooks/useScanHistory";
import { useSeo } from "../hooks/useSeo";

const VALID_MODES: DetectorMode[] = ["text", "image", "video", "live_image", "document", "audio"];

function resolveInitialMode(param: string | null): DetectorMode {
  return param && VALID_MODES.includes(param as DetectorMode) ? (param as DetectorMode) : "text";
}

function getDetectionState(detection: ReturnType<typeof useDetection>) {
  const operations = [
    detection.text,
    detection.textUpload,
    detection.image,
    detection.video,
    detection.document,
    detection.audio,
  ];

  const isPending = operations.some((op) => op.isPending);
  const activeError = operations.find((op) => Boolean(op.error))?.error as Error | undefined;

  return { isPending, errorMessage: activeError?.message };
}

export function Detector() {
  useSeo({
    title: "Detector — SLOP",
    description: "Analyze suspicious text, image, video, document, and audio samples with SLOP AI digital forensics.",
    path: "/detector",
  });

  const [searchParams, setSearchParams] = useSearchParams();
  const [mode, setMode] = useState<DetectorMode>(() => resolveInitialMode(searchParams.get("tab")));
  const [result, setResult] = useState<DetectionResult>();
  const detection = useDetection();
  const { history, addScan, clearHistory } = useScanHistory();
  const lastFileRef = useRef<File | undefined>();
  const { isPending, errorMessage } = getDetectionState(detection);

  const tabParam = searchParams.get("tab");
  useEffect(() => {
    if (tabParam && VALID_MODES.includes(tabParam as DetectorMode) && tabParam !== mode) {
      setMode(tabParam as DetectorMode);
      setResult(undefined);
    }
  }, [tabParam, mode]);

  const handleSuccessResult = useCallback(
    (scanMode: DetectorMode, scanResult: DetectionResult, file?: File, text?: string) => {
      setResult(scanResult);
      addScan(scanMode, scanResult, file, text);
    },
    [addScan]
  );

  return (
    <main className="workspace">
      <h1 className="sr-only">SLOP — Real-Time AI Content Detector</h1>
      <TabBar
        mode={mode}
        onChange={(next) => {
          setMode(next);
          setResult(undefined);
          setSearchParams({ tab: next });
        }}
      />
      <div className="detector-grid">
        <div className="detector-left-pane">
          <DetectorInputSwitcher
            mode={mode}
            loading={isPending}
            result={result}
            onSuccessResult={handleSuccessResult}
            lastFileRef={lastFileRef}
            detection={detection}
          />
        </div>
        <div className="detector-right-pane">
          <ResultPanel detectionResult={result} isAnalyzing={isPending} error={errorMessage} modality={mode} />
        </div>
      </div>
      <ScanHistory
        history={history}
        onClear={clearHistory}
        onSelect={(entry) => {
          setMode(entry.mode);
          setResult(entry.result);
          setSearchParams({ tab: entry.mode });
          window.scrollTo({ top: 0, behavior: "smooth" });
        }}
      />
    </main>
  );
}
