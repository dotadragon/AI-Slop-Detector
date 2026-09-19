import { useState, useCallback, useEffect } from "react";
import type { DetectionResult, DetectorMode } from "../types";

// ponytail: lightweight localStorage history with payload sanitization against quota limits

const STORAGE_KEY = "slop_scan_history";
const MAX_ENTRIES = 50;

export interface ScanEntry {
  id: string;
  timestamp: number;
  mode: DetectorMode;
  title?: string;
  score: number;
  label: string;
  thumbnail?: string; // base64, 80x80 max
  result: DetectionResult;
}

function sanitizeResultForHistory(result: DetectionResult): DetectionResult {
  // Strip multi-megabyte base64 heatmaps to prevent localStorage QuotaExceededError
  const { heatmap_b64, gradcam_heatmap_b64, ...rest } = result;

  let frame_scores = rest.frame_scores;
  if (frame_scores) {
    frame_scores = frame_scores.slice(0, 10).map((f) => ({
      timestamp: f.timestamp,
      score: f.score,
      has_face: f.has_face,
      face_count: f.face_count,
      temporal_change: f.temporal_change,
    }));
  }

  let sentences = rest.sentences;
  if (sentences && sentences.length > 30) {
    sentences = sentences.slice(0, 30);
  }

  return {
    ...rest,
    frame_scores,
    sentences,
  };
}

function loadHistory(): ScanEntry[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveHistory(entries: ScanEntry[]) {
  // Gracefully fallback to fewer items if localStorage quota is tight
  for (const limit of [MAX_ENTRIES, 25, 10, 5, 2]) {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(entries.slice(0, limit)));
      return;
    } catch {
      // Continue to smaller slice
    }
  }
}

// ponytail: quick thumbnail from file — canvas resize to 80x80 webp
function createThumbnail(file: File): Promise<string | undefined> {
  if (!file.type.startsWith("image/")) return Promise.resolve(undefined);
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = () => {
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement("canvas");
        canvas.width = 80;
        canvas.height = 80;
        const ctx = canvas.getContext("2d");
        if (!ctx) { resolve(undefined); return; }
        const size = Math.min(img.width, img.height);
        const sx = (img.width - size) / 2;
        const sy = (img.height - size) / 2;
        ctx.drawImage(img, sx, sy, size, size, 0, 0, 80, 80);
        resolve(canvas.toDataURL("image/webp", 0.5));
      };
      img.onerror = () => resolve(undefined);
      img.src = reader.result as string;
    };
    reader.onerror = () => resolve(undefined);
    reader.readAsDataURL(file);
  });
}

function scoreLabel(score: number): string {
  if (score >= 0.75) return "AI Detected";
  if (score >= 0.50) return "Suspicious";
  if (score >= 0.30) return "Ambiguous";
  return "Authentic";
}

export function useScanHistory() {
  const [history, setHistory] = useState<ScanEntry[]>(loadHistory);

  useEffect(() => {
    saveHistory(history);
  }, [history]);

  const addScan = useCallback(async (
    mode: DetectorMode,
    result: DetectionResult,
    file?: File,
    rawText?: string
  ) => {
    const thumbnail = file ? await createThumbnail(file) : undefined;
    const title = file ? file.name : rawText ? (rawText.trim().slice(0, 45) + (rawText.length > 45 ? "…" : "")) : undefined;
    const sanitized = sanitizeResultForHistory(result);
    const entry: ScanEntry = {
      id: crypto.randomUUID(),
      timestamp: Date.now(),
      mode,
      title,
      score: result.overall_score,
      label: scoreLabel(result.overall_score),
      thumbnail,
      result: sanitized,
    };
    setHistory((prev) => {
      const updated = [entry, ...prev.filter(p => p.id !== entry.id)].slice(0, MAX_ENTRIES);
      saveHistory(updated);
      return updated;
    });
  }, []);

  const clearHistory = useCallback(() => {
    setHistory([]);
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch {
      // ignore
    }
  }, []);

  return { history, addScan, clearHistory };
}
