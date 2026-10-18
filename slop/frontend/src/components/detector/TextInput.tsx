import { useState } from "react";
import { ScanText, FileUp, FileText, X } from "lucide-react";
import { Button } from "../ui/Button";
import { useFileDrop } from "../../hooks/useFileDrop";

type TextInputProps = {
  onAnalyze: (text: string) => void;
  onUploadFile?: (file: File) => void;
  isSubmitting?: boolean;
};

function TextStatusChip({
  file,
  onClear,
  disabled,
}: {
  file: File;
  onClear: () => void;
  disabled: boolean;
}) {
  return (
    <div className="text-status-chip">
      <div className="text-status-info">
        <FileText size={16} className="text-status-icon" />
        <div className="text-status-label-wrap">
          <span className="text-status-filename">{file.name}</span>
          <span className="text-status-filesize">({(file.size / 1024).toFixed(1)} KB)</span>
        </div>
      </div>
      <button
        type="button"
        onClick={onClear}
        disabled={disabled}
        className="text-status-clear-btn"
        title="Clear selected document"
      >
        <X size={14} />
      </button>
    </div>
  );
}

function TextPanelActions({
  onTriggerUpload,
  textLength,
  onAnalyze,
  disabled,
  isLoading,
  hasContent,
}: {
  onTriggerUpload: () => void;
  textLength: number;
  onAnalyze: () => void;
  disabled: boolean;
  isLoading: boolean;
  hasContent: boolean;
}) {
  return (
    <div className="panel-actions text-panel-actions">
      <div className="text-panel-action-left">
        <button
          type="button"
          className="btn btn-secondary text-upload-btn"
          onClick={onTriggerUpload}
          disabled={disabled}
        >
          <FileUp size={14} />
          <span>Upload Document</span>
        </button>
        <span className="text-panel-counter">{textLength.toLocaleString()} / 50,000</span>
      </div>
      <Button
        icon={<ScanText size={16} />}
        disabled={!hasContent || disabled}
        onClick={onAnalyze}
      >
        {isLoading ? "Analyzing..." : "Analyze Text"}
      </Button>
    </div>
  );
}

export function TextInput({ onAnalyze, onUploadFile, isSubmitting }: TextInputProps) {
  const [text, setText] = useState("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const textDrop = useFileDrop({
    onDropFile: (file) => {
      if (onUploadFile) {
        setSelectedFile(file);
        onUploadFile(file);
      }
    },
    disabled: isSubmitting,
  });

  const handleClearFile = () => {
    setSelectedFile(null);
    if (textDrop.fileInputRef.current) textDrop.fileInputRef.current.value = "";
  };

  const placeholderText = selectedFile
    ? `Analyzing ${selectedFile.name}...`
    : "Paste suspect text or drop a document (.PDF, .DOCX, .TXT, .MD)...";

  return (
    <section
      className={`input-panel text-input-panel ${textDrop.isDragActive ? "active" : ""}`}
      onDragOver={textDrop.handleDragOver}
      onDragLeave={textDrop.handleDragLeave}
      onDrop={textDrop.handleDrop}
    >
      <input
        ref={textDrop.fileInputRef}
        type="file"
        accept=".txt,.pdf,.docx,.doc,.md,.rtf,.csv"
        className="sr-only"
        onChange={textDrop.handleInputChange}
      />

      {selectedFile && (
        <TextStatusChip file={selectedFile} onClear={handleClearFile} disabled={Boolean(isSubmitting)} />
      )}

      <textarea
        value={text}
        onChange={(event) => {
          setText(event.target.value);
          if (selectedFile) setSelectedFile(null);
        }}
        maxLength={50000}
        placeholder={placeholderText}
        aria-label="Suspect text to analyze for AI generation"
        disabled={isSubmitting}
      />

      <TextPanelActions
        onTriggerUpload={textDrop.triggerSelect}
        textLength={text.length}
        onAnalyze={() => onAnalyze(text)}
        disabled={Boolean(isSubmitting)}
        isLoading={Boolean(isSubmitting)}
        hasContent={Boolean(text.trim() || selectedFile)}
      />
    </section>
  );
}
