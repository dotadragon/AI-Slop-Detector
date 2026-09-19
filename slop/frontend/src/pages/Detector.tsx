import { useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import type { DetectorMode, DetectionResult } from "../types";
import { ImageUpload } from "../components/detector/ImageUpload";
import { LiveImageCapture } from "../components/detector/LiveImageCapture";
import { ResultPanel } from "../components/detector/ResultPanel";
import { ScanHistory } from "../components/detector/ScanHistory";
import { TabBar } from "../components/detector/TabBar";
import { TextInput } from "../components/detector/TextInput";
import { VideoUpload } from "../components/detector/VideoUpload";
import { useDetection } from "../hooks/useDetection";
import { useScanHistory } from "../hooks/useScanHistory";
import { useSeo } from "../hooks/useSeo";
import { track } from "../utils/analytics";

export function Detector() {
  useSeo({
    title: "Detector — SLOP",
    description: "Analyze suspicious text, image, and video samples with SLOP AI forensics.",
    path: "/detector"
  });

  const [searchParams, setSearchParams] = useSearchParams();
  const tabParam = searchParams.get("tab") as DetectorMode | null;
  const initialMode: DetectorMode =
    tabParam && ["text", "image", "video", "live_image"].includes(tabParam)
      ? tabParam
      : "text";

  const [mode, setMode] = useState<DetectorMode>(initialMode);
  const [result, setResult] = useState<DetectionResult>();
  const detection = useDetection();
  const { history, addScan, clearHistory } = useScanHistory();
  const lastFileRef = useRef<File | undefined>();
  const loading = detection.text.isPending || detection.textUpload.isPending || detection.image.isPending || detection.video.isPending;
  const errorMessage =
    (detection.image.error as Error)?.message ||
    (detection.text.error as Error)?.message ||
    (detection.textUpload.error as Error)?.message ||
    (detection.video.error as Error)?.message;

  // Synchronize tab mode if URL query param changes
  useEffect(() => {
    if (tabParam && ["text", "image", "video", "live_image"].includes(tabParam) && tabParam !== mode) {
      setMode(tabParam);
      setResult(undefined);
    }
  }, [tabParam]);

  const activeInput = useMemo(() => {
    if (mode === "image") {
      return (
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
  }, [mode, loading, result, detection.image, detection.text, detection.textUpload, detection.video, addScan]);

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
        {activeInput}
        <ResultPanel result={result} loading={loading} error={errorMessage} />
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
