import { FlaskConical } from "lucide-react";
import type { DetectorMode, DetectionResult } from "../../types";
import { ImageUpload } from "./ImageUpload";
import { LiveImageCapture } from "./LiveImageCapture";
import { TextInput } from "./TextInput";
import { VideoUpload } from "./VideoUpload";
import { DocumentUpload } from "./DocumentUpload";
import { AudioUpload } from "./AudioUpload";
import { track } from "../../utils/analytics";

function AccuracyBanner({ badge, desc }: { badge: string; desc: string }) {
  return (
    <div className="experimental-heldout-banner" role="note" aria-label={badge}>
      <FlaskConical size={16} className="experimental-banner-icon" />
      <div className="experimental-banner-text">
        <span className="experimental-banner-badge">{badge}</span>
        <span className="experimental-banner-desc">{desc}</span>
      </div>
    </div>
  );
}

interface ModalityProps {
  loading: boolean;
  result?: DetectionResult;
  onSuccessResult: (mode: DetectorMode, res: DetectionResult, file?: File, text?: string) => void;
  lastFileRef: React.MutableRefObject<File | undefined>;
  detection: ReturnType<typeof import("../../hooks/useDetection").useDetection>;
}

function ImagePane({ loading, result, onSuccessResult, lastFileRef, detection }: ModalityProps) {
  return (
    <div className="image-mode-view">
      <AccuracyBanner
        badge="Held-Out Accuracy: 98.5% Gate / 91.2% Dev"
        desc="Multi-crop Swin/CLIP ensemble fused with C2PA provenance, PRNU sensor wavelets, and 2D-FFT harmonics."
      />
      <ImageUpload
        isScanning={loading}
        scanResult={result}
        onImageSelected={(file) => {
          lastFileRef.current = file;
          track("scan_started", { mode: "image" });
          detection.image.mutate(file, {
            onSuccess: (data) => {
              const resWithJob = data.result ? { ...data.result, job_id: data.job_id } : undefined;
              if (resWithJob) onSuccessResult("image", resWithJob, file);
              track("scan_completed", { mode: "image", score: resWithJob?.overall_score });
            },
          });
        }}
      />
    </div>
  );
}

function VideoPane({ loading, onSuccessResult, lastFileRef, detection }: ModalityProps) {
  return (
    <div className="video-mode-view">
      <AccuracyBanner
        badge="Held-Out Accuracy: 100.0% Dev (Temporal + Face FFT)"
        desc="Camera-motion-normalized optical flow residuals, YuNet facial frequency roll-off, and keyframe forensic consensus."
      />
      <VideoUpload
        isAnalyzing={loading}
        onVideoSelected={(file) => {
          lastFileRef.current = file;
          track("scan_started", { mode: "video" });
          detection.video.mutate(file, {
            onSuccess: (data) => {
              const resWithJob = data.result ? { ...data.result, job_id: data.job_id } : undefined;
              if (resWithJob) onSuccessResult("video", resWithJob, file);
              track("scan_completed", { mode: "video", score: resWithJob?.overall_score });
            },
          });
        }}
        onStreamResult={(streamResult) => {
          onSuccessResult("video", streamResult, lastFileRef.current);
          track("scan_completed", { mode: "video_stream", score: streamResult.overall_score });
        }}
      />
    </div>
  );
}

function DocumentPane({ loading, result, onSuccessResult, lastFileRef, detection }: ModalityProps) {
  return (
    <div className="document-mode-view">
      <AccuracyBanner
        badge="Experimental — AUC 0.868 (n=24 containers)"
        desc="Validated on 24 document containers. Small sample size — treat results as indicative, not conclusive."
      />
      <DocumentUpload
        loading={loading}
        onFile={(file) => {
          lastFileRef.current = file;
          track("scan_started", { mode: "document" });
          detection.document.mutate(file, {
            onSuccess: (data) => {
              const resWithJob = data.result ? { ...data.result, job_id: data.job_id } : undefined;
              if (resWithJob) onSuccessResult("document", resWithJob, file);
              track("scan_completed", { mode: "document", score: resWithJob?.overall_score });
            },
          });
        }}
      />
    </div>
  );
}

function AudioPane({ loading, result, onSuccessResult, lastFileRef, detection }: ModalityProps) {
  return (
    <div className="audio-mode-view">
      <AccuracyBanner
        badge="Temporarily Disabled — Under Revalidation"
        desc="Audio detection is under active revalidation. Scores shown are not yet production-ready."
      />
      <AudioUpload
        loading={loading}
        onFile={(file) => {
          lastFileRef.current = file;
          track("scan_started", { mode: "audio" });
          detection.audio.mutate(file, {
            onSuccess: (data) => {
              const resWithJob = data.result ? { ...data.result, job_id: data.job_id } : undefined;
              if (resWithJob) onSuccessResult("audio", resWithJob, file);
              track("scan_completed", { mode: "audio", score: resWithJob?.overall_score });
            },
          });
        }}
      />
    </div>
  );
}

function TextPane({ loading, onSuccessResult, lastFileRef, detection }: ModalityProps) {
  return (
    <>
      <AccuracyBanner
        badge="Verified Accuracy: 87.5% BAcc (RAID Val, n=180)"
        desc="Fine-Tuned RoBERTa + Dual-Qwen2.5 Binoculars + LightGBM fusion with bimodal human/AI sentence attribution."
      />
      <TextInput
        isSubmitting={loading}
        onAnalyze={(text) => {
          track("scan_started", { mode: "text", chars: text.length });
          detection.text.mutate(text, {
            onSuccess: (data) => {
              const resWithJob = data.result ? { ...data.result, job_id: data.job_id } : undefined;
              if (resWithJob) onSuccessResult("text", resWithJob, undefined, text);
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
              if (resWithJob) onSuccessResult("text", resWithJob, file);
              track("scan_completed", { mode: "text_upload", score: resWithJob?.overall_score });
            },
          });
        }}
      />
    </>
  );
}

export function DetectorInputSwitcher(props: ModalityProps & { mode: DetectorMode }) {
  if (props.mode === "image") return <ImagePane {...props} />;
  if (props.mode === "live_image") {
    return (
      <LiveImageCapture
        onResult={(nextResult) => {
          props.onSuccessResult("live_image", nextResult);
          track("scan_completed", { mode: "live_image", score: nextResult.overall_score });
        }}
      />
    );
  }
  if (props.mode === "video") return <VideoPane {...props} />;
  if (props.mode === "document") return <DocumentPane {...props} />;
  if (props.mode === "audio") return <AudioPane {...props} />;
  return <TextPane {...props} />;
}
