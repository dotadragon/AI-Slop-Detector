import { FileAudio, Volume2 } from "lucide-react";
import { MediaFileUpload } from "./MediaFileUpload";

interface Props {
  onFile: (file: File) => void;
  loading: boolean;
}

export function AudioUpload({ onFile, loading }: Props) {
  return (
    <MediaFileUpload
      onAcceptFile={onFile}
      isProcessing={loading}
      accept="audio/*,.mp3,.wav,.m4a,.flac,.ogg,.aac"
      ariaLabel="Audio evidence upload zone"
      emptyIcon={<FileAudio size={42} aria-hidden="true" className="dropzone-icon text-violet-400" />}
      emptyTitle="Drop audio file here or click to browse"
      emptyHint="Supports MP3, WAV, M4A, FLAC, OGG, AAC for synthetic speech detection"
      previewIcon={<Volume2 size={48} className="audio-preview-icon" />}
      previewCardClass="audio-preview-card"
      renderPreviewExtra={(file) => {
        const audioUrl = URL.createObjectURL(file);
        return <audio controls src={audioUrl} className="audio-player-element" />;
      }}
    />
  );
}
