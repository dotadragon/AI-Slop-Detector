import { useState, useRef, type DragEvent, type ChangeEvent } from "react";
import { ScanText, FileUp, FileText, X } from "lucide-react";
import { Button } from "../ui/Button";

type TextInputProps = {
  onAnalyze: (text: string) => void;
  onUploadFile?: (file: File) => void;
  loading: boolean;
};

export function TextInput({ onAnalyze, onUploadFile, loading }: TextInputProps) {
  const [text, setText] = useState("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isDragActive, setIsDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

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
    if (file && onUploadFile) {
      setSelectedFile(file);
      onUploadFile(file);
    }
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && onUploadFile) {
      setSelectedFile(file);
      onUploadFile(file);
    }
  };

  const handleClearFile = () => {
    setSelectedFile(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  return (
    <section
      className={`input-panel text-input-panel ${isDragActive ? "active" : ""}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <input
        ref={fileInputRef}
        type="file"
        accept=".txt,.pdf,.docx,.doc,.md,.rtf,.csv"
        style={{ display: "none" }}
        onChange={handleFileChange}
      />

      {/* Uploaded Document Status Chip */}
      {selectedFile && (
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            background: "color-mix(in srgb, var(--accent) 10%, var(--bg-card))",
            border: "1px solid color-mix(in srgb, var(--accent) 34%, var(--border-card))",
            borderRadius: "6px",
            padding: "8px 12px",
            marginBottom: "12px",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "8px", minWidth: 0 }}>
            <FileText size={16} style={{ color: "var(--accent)", flexShrink: 0 }} />
            <div style={{ display: "flex", alignItems: "baseline", gap: "8px", overflow: "hidden" }}>
              <span style={{ fontSize: "12px", fontWeight: 600, color: "var(--text-main)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                {selectedFile.name}
              </span>
              <span style={{ fontSize: "10px", color: "var(--text-muted)", fontFamily: "monospace", flexShrink: 0 }}>
                ({(selectedFile.size / 1024).toFixed(1)} KB)
              </span>
            </div>
          </div>
          <button
            type="button"
            onClick={handleClearFile}
            disabled={loading}
            style={{
              background: "none",
              border: "none",
              color: "var(--text-muted)",
              cursor: "pointer",
              padding: "2px",
              display: "flex",
              alignItems: "center",
            }}
            title="Clear selected document"
          >
            <X size={14} />
          </button>
        </div>
      )}

      <textarea
        value={text}
        onChange={(event) => {
          setText(event.target.value);
          if (selectedFile) setSelectedFile(null);
        }}
        maxLength={50000}
        placeholder={selectedFile ? `Analyzing ${selectedFile.name}...` : "Paste suspect text or drop a document (.PDF, .DOCX, .TXT, .MD)..."}
        aria-label="Suspect text to analyze for AI generation"
        disabled={loading}
      />
      <div className="panel-actions" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          <button
            type="button"
            className="btn btn-secondary"
            onClick={() => fileInputRef.current?.click()}
            disabled={loading}
            style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "12px", padding: "6px 12px" }}
          >
            <FileUp size={14} />
            <span>Upload Document</span>
          </button>
          <span style={{ fontSize: "12px", color: "var(--text-muted)" }}>
            {text.length.toLocaleString()} / 50,000
          </span>
        </div>
        <Button
          icon={<ScanText size={16} />}
          disabled={(!text.trim() && !selectedFile) || loading}
          onClick={() => onAnalyze(text)}
        >
          {loading ? "Analyzing..." : "Analyze Text"}
        </Button>
      </div>
    </section>
  );
}
