import { useState, useEffect } from "react";
import { ImageUp, RefreshCw, Crosshair } from "lucide-react";
import { motion } from "framer-motion";
import type { DetectionResult, DigitalSignatureInfo, BoundingBox } from "../../types";
import { SignatureBoundingBox } from "./SignatureBoundingBox";
import { useFileDrop } from "../../hooks/useFileDrop";

interface ImageUploadProps {
  onImageSelected?: (file: File) => void;
  onFile?: (file: File) => void;
  isScanning?: boolean;
  scanResult?: DetectionResult;
}

function ImageReticleAction({
  hasSignature,
  showReticle,
  onToggleReticle,
}: {
  hasSignature: boolean;
  showReticle: boolean;
  onToggleReticle: (event: React.MouseEvent) => void;
}) {
  if (!hasSignature) return null;
  return (
    <button
      type="button"
      className="btn btn-sm btn-ghost"
      onClick={onToggleReticle}
      title={showReticle ? "Hide Digital Signature Reticle" : "Show Digital Signature Reticle"}
      aria-label={showReticle ? "Hide Digital Signature Reticle" : "Show Digital Signature Reticle"}
    >
      <Crosshair size={13} />
      <span>{showReticle ? "Reticle Visible" : "Reticle Hidden"}</span>
    </button>
  );
}

function ImagePreviewActions({
  isScanning,
  onSelectAnother,
  hasSignature,
  showReticle,
  onToggleReticle,
}: {
  isScanning?: boolean;
  onSelectAnother: (event: React.MouseEvent) => void;
  hasSignature: boolean;
  showReticle: boolean;
  onToggleReticle: (event: React.MouseEvent) => void;
}) {
  return (
    <div className="preview-actions">
      <button
        type="button"
        className="btn btn-sm btn-ghost"
        onClick={onSelectAnother}
        disabled={isScanning}
      >
        <RefreshCw size={14} className={isScanning ? "animate-spin" : ""} />
        <span>{isScanning ? "Analyzing..." : "Analyze Another Image"}</span>
      </button>
      <ImageReticleAction
        hasSignature={hasSignature}
        showReticle={showReticle}
        onToggleReticle={onToggleReticle}
      />
    </div>
  );
}

function ImagePreviewContent({
  previewUrl,
  boundingBox,
  signature,
  showReticle,
  isScanning,
  onTriggerSelect,
  onToggleReticle,
}: {
  previewUrl: string;
  boundingBox?: BoundingBox;
  signature?: DigitalSignatureInfo;
  showReticle: boolean;
  isScanning?: boolean;
  onTriggerSelect: () => void;
  onToggleReticle: () => void;
}) {
  const hasValidSignature = Boolean(signature?.detected && boundingBox);

  return (
    <motion.div
      className="image-preview-wrapper"
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <div className="preview-image-wrapper">
        <div className="preview-image-inner">
          <img src={previewUrl} alt="Forensic preview" />
          {hasValidSignature && showReticle && (
            <SignatureBoundingBox boundingBox={boundingBox} signature={signature} />
          )}
        </div>
      </div>

      <ImagePreviewActions
        isScanning={isScanning}
        onSelectAnother={(e) => {
          e.stopPropagation();
          onTriggerSelect();
        }}
        hasSignature={hasValidSignature}
        showReticle={showReticle}
        onToggleReticle={(e) => {
          e.stopPropagation();
          onToggleReticle();
        }}
      />
    </motion.div>
  );
}

export function ImageUpload({ onImageSelected, onFile, isScanning, scanResult }: ImageUploadProps) {
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [showReticle, setShowReticle] = useState(true);
  const handleSelect = onImageSelected || onFile;

  const fileDrop = useFileDrop({
    onDropFile: (file) => {
      if (file.type.startsWith("image/")) {
        if (previewUrl) URL.revokeObjectURL(previewUrl);
        setPreviewUrl(URL.createObjectURL(file));
        handleSelect?.(file);
      }
    },
    disabled: isScanning,
  });

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  return (
    <section
      className={`dropzone ${fileDrop.isDragActive ? "active" : ""} ${previewUrl ? "has-preview" : ""}`}
      onDragOver={fileDrop.handleDragOver}
      onDragLeave={fileDrop.handleDragLeave}
      onDrop={fileDrop.handleDrop}
      onClick={!previewUrl ? fileDrop.triggerSelect : undefined}
      role="region"
      aria-label="Image evidence upload zone"
    >
      <input
        ref={fileDrop.fileInputRef}
        type="file"
        accept="image/*,.jpg,.jpeg,.png,.webp,.gif,.bmp,.tiff,.avif,.heic"
        className="sr-only"
        onChange={fileDrop.handleInputChange}
        disabled={isScanning}
        aria-label="Upload image file for AI detection"
      />

      {previewUrl ? (
        <ImagePreviewContent
          previewUrl={previewUrl}
          boundingBox={scanResult?.digital_signature?.bounding_box}
          signature={scanResult?.digital_signature}
          showReticle={showReticle}
          isScanning={isScanning}
          onTriggerSelect={fileDrop.triggerSelect}
          onToggleReticle={() => setShowReticle((prev) => !prev)}
        />
      ) : (
        <>
          <ImageUp size={42} aria-hidden="true" className="dropzone-icon" />
          <strong>{isScanning ? "Processing image evidence..." : "Drop image evidence here or click to browse"}</strong>
          <span className="dropzone-hint">
            Supports JPEG, PNG, WebP, AVIF, TIFF, HEIC up to 32MB
          </span>
        </>
      )}
    </section>
  );
}
