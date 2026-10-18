import { useState, type ReactNode } from "react";
import { RefreshCw } from "lucide-react";
import { motion } from "framer-motion";
import { useFileDrop } from "../../hooks/useFileDrop";

interface MediaFileUploadProps {
  onAcceptFile: (file: File) => void;
  isProcessing: boolean;
  accept: string;
  ariaLabel: string;
  emptyIcon: ReactNode;
  emptyTitle: string;
  emptyHint: string;
  previewIcon: ReactNode;
  previewCardClass: string;
  renderPreviewExtra?: (file: File) => ReactNode;
}

function FilePreviewCard({
  file,
  previewCardClass,
  previewIcon,
  renderPreviewExtra,
  triggerSelect,
  isProcessing,
}: {
  file: File;
  previewCardClass: string;
  previewIcon: ReactNode;
  renderPreviewExtra?: (file: File) => ReactNode;
  triggerSelect: () => void;
  isProcessing: boolean;
}) {
  return (
    <motion.div
      className="image-preview-wrapper"
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <div className={`preview-image-wrapper ${previewCardClass}`}>
        {previewIcon}
        <span className="document-filename">{file.name}</span>
        <span className="document-filesize">{(file.size / 1024).toFixed(1)} KB</span>
        {renderPreviewExtra && renderPreviewExtra(file)}
      </div>

      <div className="preview-actions">
        <button
          type="button"
          className="btn btn-sm btn-ghost"
          onClick={(e) => {
            e.stopPropagation();
            triggerSelect();
          }}
          disabled={isProcessing}
        >
          <RefreshCw size={14} className={isProcessing ? "animate-spin" : ""} />
          <span>{isProcessing ? "Analyzing..." : "Analyze Another File"}</span>
        </button>
      </div>
    </motion.div>
  );
}

export function MediaFileUpload({
  onAcceptFile,
  isProcessing,
  accept,
  ariaLabel,
  emptyIcon,
  emptyTitle,
  emptyHint,
  previewIcon,
  previewCardClass,
  renderPreviewExtra,
}: MediaFileUploadProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const drop = useFileDrop({
    onDropFile: (file) => {
      setSelectedFile(file);
      onAcceptFile(file);
    },
    disabled: isProcessing,
  });

  return (
    <section
      className={`dropzone ${drop.isDragActive ? "active" : ""} ${selectedFile ? "has-preview" : ""}`}
      onDragOver={drop.handleDragOver}
      onDragLeave={drop.handleDragLeave}
      onDrop={drop.handleDrop}
      onClick={!selectedFile ? drop.triggerSelect : undefined}
      role="region"
      aria-label={ariaLabel}
    >
      <input
        ref={drop.fileInputRef}
        type="file"
        accept={accept}
        className="sr-only"
        onChange={drop.handleInputChange}
        disabled={isProcessing}
        aria-label={ariaLabel}
      />

      {selectedFile ? (
        <FilePreviewCard
          file={selectedFile}
          previewCardClass={previewCardClass}
          previewIcon={previewIcon}
          renderPreviewExtra={renderPreviewExtra}
          triggerSelect={drop.triggerSelect}
          isProcessing={isProcessing}
        />
      ) : (
        <>
          {emptyIcon}
          <strong>{isProcessing ? "Processing forensic evidence..." : emptyTitle}</strong>
          <span className="dropzone-hint">{emptyHint}</span>
        </>
      )}
    </section>
  );
}
