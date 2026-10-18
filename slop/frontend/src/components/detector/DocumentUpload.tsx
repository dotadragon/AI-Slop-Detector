import { FileSearch, FileText } from "lucide-react";
import { MediaFileUpload } from "./MediaFileUpload";

interface Props {
  onFile: (file: File) => void;
  loading: boolean;
}

export function DocumentUpload({ onFile, loading }: Props) {
  return (
    <MediaFileUpload
      onAcceptFile={onFile}
      isProcessing={loading}
      accept=".pdf,.docx,.doc,.txt"
      ariaLabel="Document evidence upload zone"
      emptyIcon={<FileSearch size={42} aria-hidden="true" className="dropzone-icon text-cyan-400" />}
      emptyTitle="Drop PDF or DOCX file here or click to browse"
      emptyHint="Supports PDF, DOCX, DOC, and TXT files for figure & stream forensics"
      previewIcon={<FileText size={48} className="document-preview-icon" />}
      previewCardClass="document-preview-card"
    />
  );
}
