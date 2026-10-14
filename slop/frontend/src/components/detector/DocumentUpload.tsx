import { useState, useRef, type DragEvent, type ChangeEvent } from "react";
import { FileSearch, RefreshCw, FileText } from "lucide-react";
import { motion } from "framer-motion";
import type { DetectionResult } from "../../types";

interface Props {
  onFile: (file: File) => void;
  loading: boolean;
  result?: DetectionResult;
}

export function DocumentUpload({ onFile, loading }: Props) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isDragActive, setIsDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleSelectedFile = (file?: File) => {
    if (!file) return;
    setSelectedFile(file);
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
    if (file) handleSelectedFile(file);
  };

  const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleSelectedFile(file);
  };

  const triggerSelect = () => {
    if (!loading) fileInputRef.current?.click();
  };

  return (
    <section
      className={`dropzone ${isDragActive ? "active" : ""} ${selectedFile ? "has-preview" : ""}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={!selectedFile ? triggerSelect : undefined}
      role="region"
      aria-label="Document evidence upload zone"
    >
      <input
        ref={fileInputRef}
        type="file"
        accept=".pdf,.docx,.doc,.txt"
        style={{ display: "none" }}
        onChange={handleChange}
        disabled={loading}
        aria-label="Upload document file for AI forensics"
      />

      {selectedFile ? (
        <motion.div
          className="image-preview-wrapper"
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
        >
          <div className="preview-image-wrapper flex flex-col items-center justify-center p-8 bg-slate-900/50 rounded-lg min-h-[220px]">
            <FileText size={48} className="text-cyan-400 mb-3" />
            <span className="font-semibold text-slate-200 text-sm">{selectedFile.name}</span>
            <span className="text-xs text-slate-500 mt-1">
              {(selectedFile.size / 1024).toFixed(1)} KB
            </span>
          </div>

          <div className="preview-actions mt-4">
            <button
              type="button"
              className="btn btn-sm btn-ghost"
              onClick={(e) => {
                e.stopPropagation();
                triggerSelect();
              }}
              disabled={loading}
            >
              <RefreshCw size={14} className={loading ? "animate-spin" : ""} />
              <span>{loading ? "Analyzing Document..." : "Analyze Another Document"}</span>
            </button>
          </div>
        </motion.div>
      ) : (
        <>
          <FileSearch size={42} aria-hidden="true" className="dropzone-icon text-cyan-400" />
          <strong>{loading ? "Processing document forensics..." : "Drop PDF or DOCX file here or click to browse"}</strong>
          <span className="dropzone-hint">Supports PDF, DOCX, DOC, and TXT files for figure & stream forensics</span>
        </>
      )}
    </section>
  );
}
