import { api } from "./client";
import { getCookie } from "../utils/sanitize";
import type { AnalyzeResponse } from "../types";

export async function uploadVideo(file: File) {
  const form = new FormData();
  form.append("file", file);
  const { data } = await api.post<AnalyzeResponse>("/video/upload", form);
  return data;
}

export type VideoStreamEvent = {
  type: "started" | "frame_result" | "complete" | "done";
  job_id: string;
  progress_percent: number;
  current_frame?: number;
  total_frames?: number;
  timestamp?: number;
  frame_score?: number;
  faces_detected?: number;
  temporal_change?: number;
  thumbnail_b64?: string;
  video_info?: Record<string, unknown>;
  result?: {
    overall_score: number;
    signals: Record<string, unknown>;
    frame_scores: Array<{ timestamp: number; score: number; has_face: boolean; face_count: number; temporal_change: number }>;
    video_info: Record<string, unknown>;
    model_hint: string;
  };
  error?: string;
};

async function getCsrfHeader(baseURL: string): Promise<Record<string, string>> {
  let csrf = getCookie("csrf_token");
  if (!csrf) {
    try {
      const res = await fetch(`${baseURL}/auth/csrf`, { credentials: "include" });
      if (res.ok) {
        const csrfData = await res.json();
        if (csrfData.csrf_token) csrf = csrfData.csrf_token;
      }
    } catch {
      // Ignore csrf retrieval error
    }
  }
  return csrf ? { "X-CSRF-Token": csrf } : {};
}

function processSseLines(lines: string[], onEvent: (event: VideoStreamEvent) => void) {
  for (const line of lines) {
    if (line.startsWith("data: ")) {
      try {
        const event = JSON.parse(line.slice(6)) as VideoStreamEvent;
        onEvent(event);
      } catch {
        // Skip malformed SSE lines
      }
    }
  }
}

export async function uploadVideoStream(
  file: File,
  onEvent: (event: VideoStreamEvent) => void
): Promise<void> {
  const form = new FormData();
  form.append("file", file);

  const baseURL = import.meta.env.VITE_API_URL ?? "/api/v1";
  const headers = await getCsrfHeader(baseURL);

  const response = await fetch(`${baseURL}/video/upload-stream`, {
    method: "POST",
    body: form,
    credentials: "include",
    headers,
  });

  if (!response.ok || !response.body) {
    throw new Error(`Upload failed: ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    processSseLines(lines, onEvent);
  }
}
