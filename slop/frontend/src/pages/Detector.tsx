import { useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { FlaskConical } from "lucide-react";
import type { DetectorMode, DetectionResult } from "../types";
import { ImageUpload } from "../components/detector/ImageUpload";
import { LiveImageCapture } from "../components/detector/LiveImageCapture";
import { ResultPanel } from "../components/detector/ResultPanel";
import { ScanHistory } from "../components/detector/ScanHistory";
import { TabBar } from "../components/detector/TabBar";
import { TextInput } from "../components/detector/TextInput";
import { VideoUpload } from "../components/detector/VideoUpload";
import { DocumentUpload } from "../components/detector/DocumentUpload";
import { AudioUpload } from "../components/detector/AudioUpload";
import { useDetection } from "../hooks/useDetection";
import { useScanHistory } from "../hooks/useScanHistory";
import { useSeo } from "../hooks/useSeo";
import { track } from "../utils/analytics";

function ExperimentalBanner({ label }: { label: string }) {
  return (
    <div className="experimental-heldout-banner" role="note" aria-label={`Experimental ${label} detection status`}>
      <FlaskConical size={16} className="experimental-banner-icon" />
      <div className="experimental-banner-text">
        <span className="experimental-banner-badge">Experimental Beta — Results For Evaluation Only</span>
        <span className="experimental-banner-desc">Modality model under active validation. Results intended for empirical testing and evaluation.</span>
      </div>
    </div>
  );
}

export function Detector() {
  useSeo({
    title: "Detector — SLOP",
    description: "Analyze suspicious text, image, video, document, and audio samples with SLOP AI digital forensics.",
    path: "/detector"
  });

  const [searchParams, setSearchParams] = useSearchParams();
  const tabParam = searchParams.get("tab") as DetectorMode | null;
  const initialMode: DetectorMode =
    tabParam && ["text", "image", "video", "live_image", "document", "audio"].includes(tabParam)
      ? tabParam
      : "text";

  const [mode, setMode] = useState<DetectorMode>(initialMode);
  const [result, setResult] = useState<DetectionResult>();
  const detection = useDetection();
  const { history, addScan, clearHistory } = useScanHistory();
  const lastFileRef = useRef<File | undefined>();
  const loading =
    detection.text.isPending ||
    detection.textUpload.isPending ||
    detection.image.isPending ||
    detection.video.isPending ||
    detection.document.isPending ||
    detection.audio.isPending;

  const errorMessage =
    (detection.image.error as Error)?.message ||
    (detection.text.error as Error)?.message ||
    (detection.textUpload.error as Error)?.message ||
    (detection.video.error as Error)?.message ||
    (detection.document.error as Error)?.message ||
    (detection.audio.error as Error)?.message;

  // Synchronize tab mode if URL query param changes
  useEffect(() => {
    if (tabParam && ["text", "image", "video", "live_image", "document", "audio"].includes(tabParam) && tabParam !== mode) {
      setMode(tabParam);
      setResult(undefined);
    }
  }, [tabParam, mode]);

  const activeInput = useMemo(() => {
    if (mode === "image") {
      return (
        <div className="image-mode-view">
          <div className="experimental-heldout-banner" role="note" aria-label="Experimental image detection status">
            <FlaskConical size={16} className="experimental-banner-icon" />
            <div className="experimental-banner-text">
              <span className="experimental-banner-badge">Experimental / Held-out 78.85%</span>
              <span className="experimental-banner-desc">Canonical evaluation (n=52, SDXL &amp; Unsplash). Real-world out-of-distribution performance varies.</span>
            </div>
          </div>
          <ImageUpload
            loading={loading}
            result={result}
            onFile={(file) => {
              lastFileRef.current = file;
              track("scan_started", { mode: "image" });
              detection.image.mutate(file, {
                onSuccess: (data) => {
                  const resWithJob = data.result ? { ...data.result, job_id: data.job_id } : undefined;
                  setResult(resWithJob);
                  if (resWithJob) addScan("image", resWithJob, file);
                  track("scan_completed", { mode: "image", score: resWithJob?.overall_score });
                },
              });
            }}
          />
        </div>
      );
    }
    if (mode === "live_image") {
      return (
        <LiveImageCapture
          onResult={(nextResult) => {
            setResult(nextResult);
            addScan("live_image", nextResult);
            track("scan_completed", { mode: "live_image", score: nextResult.overall_score });
          }}
        />
      );
    }
    if (mode === "video") {
      return (
        <div className="video-mode-view">
          <ExperimentalBanner label="Video" />
          <VideoUpload
            loading={loading}
            onFile={(file) => {
              lastFileRef.current = file;
              track("scan_started", { mode: "video" });
              detection.video.mutate(file, {
                onSuccess: (data) => {
                  const resWithJob = data.result ? { ...data.result, job_id: data.job_id } : undefined;
                  setResult(resWithJob);
                  if (resWithJob) addScan("video", resWithJob, file);
                  track("scan_completed", { mode: "video", score: resWithJob?.overall_score });
                },
              });
            }}
            onStreamResult={(streamResult) => {
              setResult(streamResult);
              addScan("video", streamResult, lastFileRef.current);
              track("scan_completed", { mode: "video_stream", score: streamResult.overall_score });
            }}
          />
        </div>
      );
    }
    if (mode === "document") {
      return (
        <div className="document-mode-view">
          <ExperimentalBanner label="Document" />
          <DocumentUpload
            loading={loading}
            result={result}
            onFile={(file) => {
              lastFileRef.current = file;
              track("scan_started", { mode: "document" });
              detection.document.mutate(file, {
                onSuccess: (data) => {
                  const resWithJob = data.result ? { ...data.result, job_id: data.job_id } : undefined;
                  setResult(resWithJob);
                  if (resWithJob) addScan("document", resWithJob, file);
                  track("scan_completed", { mode: "document", score: resWithJob?.overall_score });
                },
              });
            }}
          />
        </div>
      );
    }
    if (mode === "audio") {
      return (
        <div className="audio-mode-view">
          <ExperimentalBanner label="Audio" />
          <AudioUpload
            loading={loading}
            result={result}
            onFile={(file) => {
              lastFileRef.current = file;
              track("scan_started", { mode: "audio" });
              detection.audio.mutate(file, {
                onSuccess: (data) => {
                  const resWithJob = data.result ? { ...data.result, job_id: data.job_id } : undefined;
                  setResult(resWithJob);
                  if (resWithJob) addScan("audio", resWithJob, file);
                  track("scan_completed", { mode: "audio", score: resWithJob?.overall_score });
                },
              });
            }}
          />
        </div>
      );
    }
    return (
      <TextInput
        loading={loading}
        onAnalyze={(text) => {
          track("scan_started", { mode: "text", chars: text.length });
          detection.text.mutate(text, {
            onSuccess: (data) => {
              const resWithJob = data.result ? { ...data.result, job_id: data.job_id } : undefined;
              setResult(resWithJob);
              if (resWithJob) addScan("text", resWithJob, undefined, text);
              track("scan_completed", { mode: "text", score: resWithJob?.overall_score });
            },
          });
        }}
        onUploadFile={(file) => {
          lastFileRef.current = file;
          track("scan_started", { mode: "text_upload", file: file.name });
          detection.textUpload.mutate(file, {
            onSuccess: (data) => {
              const resWithJob = data.result ? { ...data.result, job_id: data.job_id } : undefined;
              setResult(resWithJob);
              if (resWithJob) addScan("text", resWithJob, file);
              track("scan_completed", { mode: "text_upload", score: resWithJob?.overall_score });
            },
          });
        }}
      />
    );
  }, [mode, loading, result, detection.image, detection.text, detection.textUpload, detection.video, detection.document, detection.audio, addScan]);

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
          {activeInput}
        </div>
        <div className="detector-right-pane">
          <ResultPanel result={result} loading={loading} error={errorMessage} />
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
