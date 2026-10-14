import { useState, useEffect, useRef, type DragEvent, type ChangeEvent } from "react";
import { ImageUp, RefreshCw, Sparkles, Crosshair, FlaskConical } from "lucide-react";
import { motion } from "framer-motion";
import type { DetectionResult } from "../../types";
import { SignatureBoundingBox } from "./SignatureBoundingBox";

interface Props {
  onFile: (file: File) => void;
  loading: boolean;
  result?: DetectionResult;
}

export function ImageUpload({ onFile, loading, result }: Props) {
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [showReticle, setShowReticle] = useState(true);
  const [isDragActive, setIsDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // ponytail: Rung 4 native HTML5 Drag & Drop + FileReader, zero react-dropzone bloat
  const handleSelectedFile = (file?: File) => {
    if (!file) return;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(URL.createObjectURL(file));
    onFile(file);
  };

  const handleDragOver = (e: DragEvent<HTMLElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (!loading) setIsDragActive(true);
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
    if (loading) return;
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith("image/")) {
      handleSelectedFile(file);
    }
  };

  const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleSelectedFile(file);
  };

  const triggerSelect = () => {
    if (!loading) fileInputRef.current?.click();
  };

  useEffect(() => {
    return () => { if (previewUrl) URL.revokeObjectURL(previewUrl); };
  }, [previewUrl]);

  const signature = result?.digital_signature;
  const boundingBox = signature?.bounding_box;

  return (
    <section
      className={`dropzone ${isDragActive ? "active" : ""} ${previewUrl ? "has-preview" : ""}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={!previewUrl ? triggerSelect : undefined}
      role="region"
      aria-label="Image evidence upload zone"
    >
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*,.jpg,.jpeg,.png,.webp,.gif,.bmp,.tiff,.avif,.heic"
        style={{ display: "none" }}
        onChange={handleChange}
        disabled={loading}
        aria-label="Upload image file for AI detection"
      />

      {previewUrl ? (
        <motion.div
          className="image-preview-wrapper"
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
        >
          <div className="preview-image-wrapper">
            <div style={{ display: 'grid', placeItems: 'center', padding: '12px 16px' }}>
              <div className="preview-image-inner" style={{ position: 'relative', display: 'inline-block' }}>
                <img
                  src={previewUrl}
                  alt="preview"
                  style={{
                    maxWidth: '100%',
                    maxHeight: '56vh',
                    width: 'auto',
                    height: 'auto',
                    objectFit: 'contain',
                    borderRadius: '12px',
                    border: '1px solid var(--panel-border)'
                  }}
                />

                {/* Green Bounding Box Overlay for SynthID / Digital Signature */}
                {boundingBox && signature?.detected && showReticle && (
                  <SignatureBoundingBox boundingBox={boundingBox} signature={signature} />
                )}
              </div>
            </div>
          </div>

          <div className="preview-actions">
            <button
              type="button"
              className="btn btn-sm btn-ghost"
              onClick={(e) => { e.stopPropagation(); triggerSelect(); }}
              disabled={loading}
            >
              <RefreshCw size={14} className={loading ? "animate-spin" : ""} />
              <span>{loading ? "Analyzing..." : "Analyze Another Image"}</span>
            </button>

            {signature?.detected && boundingBox && (
              <button
                type="button"
                className={`btn btn-sm ${showReticle ? "btn-emerald" : "btn-ghost"}`}
                onClick={(e) => { e.stopPropagation(); setShowReticle((prev) => !prev); }}
                title="Toggle signature reticle overlay"
              >
                <Crosshair size={13} />
                <span>{showReticle ? "Reticle Visible" : "Reticle Hidden"}</span>
              </button>
            )}
          </div>
        </motion.div>
      ) : (
        <>
          <ImageUp size={42} aria-hidden="true" className="dropzone-icon" />
          <strong>{loading ? "Processing image evidence..." : "Drop image evidence here or click to browse"}</strong>
          <span className="dropzone-hint">Supports JPEG, PNG, WEBP, TIFF, BMP, GIF, HEIC, AVIF</span>
        </>
      )}
    </section>
  );
}
