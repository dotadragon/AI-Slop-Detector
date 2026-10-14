import { Clapperboard, CheckCircle, AlertTriangle } from "lucide-react";
import { useState, useCallback, useRef, type DragEvent, type ChangeEvent } from "react";
import { uploadVideoStream, type VideoStreamEvent } from "../../api/video";
import type { DetectionResult } from "../../types";

type VideoUploadProps = {
  onFile: (file: File) => void;
  onStreamResult?: (result: DetectionResult) => void;
  loading: boolean;
};

type StreamState = {
  active: boolean;
  progress: number;
  currentFrame: number;
  totalFrames: number;
  lastFrameScore: number | null;
  lastThumbnail: string | null;
  facesDetected: number;
  status: string;
  error: string | null;
};

export function VideoUpload({ onFile, onStreamResult, loading }: VideoUploadProps) {
  const [stream, setStream] = useState<StreamState>({
    active: false,
    progress: 0,
    currentFrame: 0,
    totalFrames: 0,
    lastFrameScore: null,
    lastThumbnail: null,
    facesDetected: 0,
    status: "idle",
    error: null,
  });

  const [isDragActive, setIsDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // ponytail: Rung 4 native HTML5 drag & drop, zero react-dropzone dependency
  const processVideoFile = useCallback(async (file: File) => {
    if (!file) return;

    setStream({
      active: true,
      progress: 0,
      currentFrame: 0,
      totalFrames: 0,
      lastFrameScore: null,
      lastThumbnail: null,
      facesDetected: 0,
      status: "Uploading video...",
      error: null,
    });

    try {
      await uploadVideoStream(file, (event: VideoStreamEvent) => {
        if (event.type === "started") {
          setStream((prev) => ({
            ...prev,
            totalFrames: event.total_frames ?? 0,
            status: `Analyzing ${event.total_frames} frames...`,
          }));
        } else if (event.type === "frame_result") {
          setStream((prev) => ({
            ...prev,
            progress: event.progress_percent,
            currentFrame: event.current_frame ?? prev.currentFrame,
            lastFrameScore: event.frame_score ?? null,
            lastThumbnail: event.thumbnail_b64 ?? null,
            facesDetected: prev.facesDetected + (event.faces_detected ?? 0),
            status: `Frame ${event.current_frame}/${event.total_frames} — ${event.progress_percent}%`,
          }));
        } else if (event.type === "complete" && event.result) {
          setStream((prev) => ({
            ...prev,
            active: false,
            progress: 100,
            status: "Analysis complete",
          }));
          if (onStreamResult) {
            onStreamResult(event.result as DetectionResult);
          }
        } else if (event.type === "done") {
          setStream((prev) => ({ ...prev, active: false }));
        }
      });
    } catch (err) {
      console.warn("SSE stream upload error, falling back to standard upload:", err);
      setStream((prev) => ({
        ...prev,
        active: false,
        error: null,
        status: "Processing video...",
      }));
      onFile(file);
    }
  }, [onFile, onStreamResult]);

  const handleDragOver = (e: DragEvent<HTMLElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (!stream.active && !loading) setIsDragActive(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);
  };

  const handleDrop = (e: DragEvent<HTMLElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);
    if (stream.active || loading) return;
    const file = e.dataTransfer.files?.[0];
    if (file) processVideoFile(file);
  };

  const handleInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) processVideoFile(file);
  };

  const triggerSelect = () => {
    if (!stream.active && !loading) fileInputRef.current?.click();
  };

  // Active streaming state: show progress UI
  if (stream.active) {
    return (
      <div className="video-pane" style={{ maxHeight: '68vh', overflow: 'auto' }}>
        <section className="input-panel video-progress-panel" style={{ padding: '12px 14px', margin: '0' }}>
          <div className="video-progress-header" style={{ marginBottom: '12px' }}>
            <Clapperboard size={24} />
            <span>{stream.status}</span>
          </div>

          {/* Circular progress with native SVG */}
          <div className="video-progress-ring-container" style={{ margin: '12px auto' }}>
            <svg className="video-progress-ring" viewBox="0 0 120 120">
              <circle className="video-progress-track" cx="60" cy="60" r="52" />
              <circle
                className="video-progress-fill"
                cx="60"
                cy="60"
                r="52"
                strokeDasharray={`${2 * Math.PI * 52}`}
                strokeDashoffset={`${2 * Math.PI * 52 * (1 - stream.progress / 100)}`}
              />
            </svg>
            <div className="video-progress-percent">
              <strong>{stream.progress}%</strong>
              <span>{stream.currentFrame}/{stream.totalFrames}</span>
            </div>
          </div>

          {/* Current frame thumbnail + score */}
          {stream.lastThumbnail && (
            <div className="video-current-frame" style={{ width: '140px', height: '84px', margin: '12px auto', position: 'relative' }}>
              <img
                src={`data:image/jpeg;base64,${stream.lastThumbnail}`}
                alt={`Frame ${stream.currentFrame}`}
                style={{ width: '140px', height: '84px', objectFit: 'cover', borderRadius: '8px' }}
              />
              {stream.lastFrameScore !== null && (
                <div className="video-frame-score-badge" data-level={
                  stream.lastFrameScore >= 0.72 ? "high" :
                  stream.lastFrameScore >= 0.42 ? "mid" : "low"
                }>
                  {Math.round(stream.lastFrameScore * 100)}%
                </div>
              )}
            </div>
          )}

          {/* Frame progress bar */}
          <div className="video-progress-bar-track" style={{ margin: '12px 0' }}>
            <div
              className="video-progress-bar-fill"
              style={{ width: `${stream.progress}%` }}
            />
          </div>

          {stream.error && (
            <div className="video-error" style={{ margin: '12px 0' }}>
              <AlertTriangle size={16} />
              {stream.error}
            </div>
          )}
        </section>
      </div>
    );
  }

  // Completed state
  if (stream.progress === 100 && !stream.active) {
    return (
      <div className="video-pane" style={{ maxHeight: '68vh', overflow: 'auto' }}>
        <section
          className="input-panel video-complete-panel cursor-pointer"
          style={{ padding: '12px 14px', margin: '0' }}
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
            style={{ display: "none" }}
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

  // Default dropzone
  return (
    <div className="video-pane" style={{ maxHeight: '68vh', overflow: 'auto' }}>
      <section
        className={`dropzone ${isDragActive ? "active" : ""}`}
        style={{ padding: '14px' }}
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
          style={{ display: "none" }}
          onChange={handleInputChange}
          disabled={loading}
          aria-label="Upload video file for frame-by-frame AI detection"
        />
        <Clapperboard size={34} aria-hidden="true" />
        <strong>{loading ? "Extracting frames..." : "Drop video sample or click to browse"}</strong>
        <span>MP4, MKV, MOV, AVI, WEBM, FLV</span>
        <span className="dropzone-hint">Real-time frame-by-frame analysis with Farnebäck motion stability</span>
      </section>
    </div>
  );
}
