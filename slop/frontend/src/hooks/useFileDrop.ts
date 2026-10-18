import { useState, useRef, useCallback, type DragEvent, type ChangeEvent } from "react";

interface UseFileDropOptions {
  onDropFile?: (file: File) => void;
  disabled?: boolean;
}

export function useFileDrop({ onDropFile, disabled = false }: UseFileDropOptions) {
  const [isDragActive, setIsDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const handler = onDropFile;

  const handleDragOver = useCallback(
    (e: DragEvent<HTMLElement>) => {
      e.preventDefault();
      e.stopPropagation();
      if (!disabled) setIsDragActive(true);
    },
    [disabled]
  );

  const handleDragLeave = useCallback((e: DragEvent<HTMLElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);
  }, []);

  const handleDrop = useCallback(
    (e: DragEvent<HTMLElement>) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragActive(false);
      if (disabled) return;
      const file = e.dataTransfer.files?.[0];
      if (file && handler) handler(file);
    },
    [disabled, handler]
  );

  const handleInputChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file && handler) handler(file);
    },
    [handler]
  );

  const triggerSelect = useCallback(() => {
    if (!disabled) fileInputRef.current?.click();
  }, [disabled]);

  return {
    isDragActive,
    fileInputRef,
    handleDragOver,
    handleDragLeave,
    handleDrop,
    handleInputChange,
    triggerSelect,
  };
}
