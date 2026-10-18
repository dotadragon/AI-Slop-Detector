import { useCallback, useState } from "react";
import { AlertTriangle, Clapperboard, CheckCircle, RefreshCw } from "lucide-react";
import type { DetectionResult } from "../../types";
import { uploadVideoStream, type VideoStreamEvent } from "../../api/video";
import { useFileDrop } from "../../hooks/useFileDrop";

type VideoUploadProps = {
  onVideoSelected: (file: File) => void;
  onStreamResult?: (result: DetectionResult) => void;
  isAnalyzing?: boolean;
};

type StreamState = {
  active: boolean;
  progress: number;
  currentFrame: number;
  totalFrames: number;
  status: string;
  lastFrameScore: number | null;
  lastThumbnail: string | null;
  facesDetected: number;
  error: string | null;
};

const INITIAL_STREAM_STATE: StreamState = {
  active: false,
  progress: 0,
  currentFrame: 0,
  totalFrames: 0,
  status: "Ready",
  lastFrameScore: null,
  lastThumbnail: null,
  facesDetected: 0,
  error: null,
};

function StreamingProgressView({ stream }: { stream: StreamState }) {
  const scoreLevel =
    stream.lastFrameScore === null
      ? "idle"
      : stream.lastFrameScore >= 0.70
      ? "high"
      : stream.lastFrameScore >= 0.40
      ? "med"
      : "low";

  return (
    <div className="video-pane video-pane-scroll">
      <section className="input-panel video-progress-panel video-progress-panel-compact" role="status" aria-live="polite">
        <div className="video-progress-header">
          <RefreshCw size={16} className="animate-spin text-amber" />
          <span className="video-status-text">{stream.status}</span>
        </div>

        {stream.lastThumbnail && (
          <div className="video-thumb-preview video-thumb-preview-compact">
            <img src={`data:image/jpeg;base64,${stream.lastThumbnail}`} alt="Live scanned video frame" />
            {stream.lastFrameScore !== null && (
              <div className="video-frame-score-badge" data-level={scoreLevel}>
                {Math.round(stream.lastFrameScore * 100)}%
              </div>
            )}
          </div>
        )}

        <div className="video-progress-bar-track video-progress-bar-track-spaced">
          <div className="video-progress-bar-fill" style={{ width: `${stream.progress}%` }} />
        </div>

        {stream.error && (
          <div className="video-error video-error-spaced">
            <AlertTriangle size={16} />
            {stream.error}
          </div>
        )}
      </section>
    </div>
  );
}

function VideoCompletedPanel({
  fileInputRef,
  triggerSelect,
  handleDragOver,
  handleDragLeave,
  handleDrop,
  handleInputChange,
}: {
  fileInputRef: React.RefObject<HTMLInputElement>;
  triggerSelect: () => void;
  handleDragOver: (e: React.DragEvent<HTMLElement>) => void;
  handleDragLeave: (e: React.DragEvent<HTMLElement>) => void;
  handleDrop: (e: React.DragEvent<HTMLElement>) => void;
  handleInputChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
}) {
  return (
    <div className="video-pane video-pane-scroll">
      <section
        className="input-panel video-complete-panel cursor-pointer video-progress-panel-compact"
        onClick={triggerSelect}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        role="region"
        aria-label="Upload replacement video file"
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="video/*,.mp4,.mkv,.mov,.avi,.webm,.flv"
          className="sr-only"
          onChange={handleInputChange}
          aria-label="Upload new video file"
        />
        <CheckCircle size={34} className="video-complete-icon" aria-hidden="true" />
        <strong>Analysis complete</strong>
        <span>Drop another video or click to scan again</span>
      </section>
    </div>
  );
}

function VideoDropzonePanel({
  isDragActive,
  isAnalyzing,
  fileInputRef,
  triggerSelect,
  handleDragOver,
  handleDragLeave,
  handleDrop,
  handleInputChange,
}: {
  isDragActive: boolean;
  isAnalyzing?: boolean;
  fileInputRef: React.RefObject<HTMLInputElement>;
  triggerSelect: () => void;
  handleDragOver: (e: React.DragEvent<HTMLElement>) => void;
  handleDragLeave: (e: React.DragEvent<HTMLElement>) => void;
  handleDrop: (e: React.DragEvent<HTMLElement>) => void;
  handleInputChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
}) {
  return (
    <div className="video-pane video-pane-scroll">
      <section
        className={`dropzone video-dropzone-compact ${isDragActive ? "active" : ""}`}
        onClick={triggerSelect}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        role="region"
        aria-label="Video file upload zone"
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="video/*,.mp4,.mkv,.mov,.avi,.webm,.flv"
          className="sr-only"
          onChange={handleInputChange}
          disabled={isAnalyzing}
          aria-label="Upload video file for frame-by-frame AI detection"
        />
        <Clapperboard size={34} aria-hidden="true" />
        <strong>{isAnalyzing ? "Extracting frames..." : "Drop video sample or click to browse"}</strong>
        <span>MP4, MKV, MOV, AVI, WEBM, FLV</span>
        <span className="dropzone-hint">Real-time frame-by-frame analysis with Farnebäck motion stability</span>
      </section>
    </div>
  );
}

function handleStartedEvent(
  event: VideoStreamEvent,
  setStream: React.Dispatch<React.SetStateAction<StreamState>>
) {
  setStream((prev) => ({
    ...prev,
    totalFrames: event.total_frames ?? 0,
    status: `Analyzing ${event.total_frames} frames...`,
  }));
}

function handleFrameEvent(
  event: VideoStreamEvent,
  setStream: React.Dispatch<React.SetStateAction<StreamState>>
) {
  setStream((prev) => ({
    ...prev,
    progress: event.progress_percent,
    currentFrame: event.current_frame ?? prev.currentFrame,
    lastFrameScore: event.frame_score ?? null,
    lastThumbnail: event.thumbnail_b64 ?? null,
    facesDetected: prev.facesDetected + (event.faces_detected ?? 0),
    status: `Frame ${event.current_frame}/${event.total_frames} — ${event.progress_percent}%`,
  }));
}

function handleCompleteEvent(
  event: VideoStreamEvent,
  setStream: React.Dispatch<React.SetStateAction<StreamState>>,
  onStreamResult?: (result: DetectionResult) => void
) {
  setStream((prev) => ({ ...prev, active: false, progress: 100, status: "Analysis complete" }));
  if (event.result) {
    onStreamResult?.(event.result as DetectionResult);
  }
}

function applyStreamEvent(
  event: VideoStreamEvent,
  setStream: React.Dispatch<React.SetStateAction<StreamState>>,
  onStreamResult?: (result: DetectionResult) => void
) {
  switch (event.type) {
    case "started":
      handleStartedEvent(event, setStream);
      break;
    case "frame_result":
      handleFrameEvent(event, setStream);
      break;
    case "complete":
      handleCompleteEvent(event, setStream, onStreamResult);
      break;
    case "done":
      setStream((prev) => ({ ...prev, active: false }));
      break;
  }
}

export function VideoUpload({ onVideoSelected, onStreamResult, isAnalyzing }: VideoUploadProps) {
  const [stream, setStream] = useState<StreamState>(INITIAL_STREAM_STATE);

  const processVideoFile = useCallback(
    async (file: File) => {
      if (!file) return;
      setStream({ ...INITIAL_STREAM_STATE, active: true, status: "Uploading video..." });

      try {
        await uploadVideoStream(file, (event) => applyStreamEvent(event, setStream, onStreamResult));
      } catch {
        setStream((prev) => ({ ...prev, active: false, status: "Processing video..." }));
        onVideoSelected(file);
      }
    },
    [onVideoSelected, onStreamResult]
  );

  const videoDrop = useFileDrop({
    onDropFile: processVideoFile,
    disabled: stream.active || isAnalyzing,
  });

  if (stream.active) {
    return <StreamingProgressView stream={stream} />;
  }

  if (stream.progress === 100) {
    return (
      <VideoCompletedPanel
        fileInputRef={videoDrop.fileInputRef}
        triggerSelect={videoDrop.triggerSelect}
        handleDragOver={videoDrop.handleDragOver}
        handleDragLeave={videoDrop.handleDragLeave}
        handleDrop={videoDrop.handleDrop}
        handleInputChange={videoDrop.handleInputChange}
      />
    );
  }

  return (
    <VideoDropzonePanel
      isDragActive={videoDrop.isDragActive}
      isAnalyzing={isAnalyzing}
      fileInputRef={videoDrop.fileInputRef}
      triggerSelect={videoDrop.triggerSelect}
      handleDragOver={videoDrop.handleDragOver}
      handleDragLeave={videoDrop.handleDragLeave}
      handleDrop={videoDrop.handleDrop}
      handleInputChange={videoDrop.handleInputChange}
    />
  );
}
