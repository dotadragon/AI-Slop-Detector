import { api } from "./client";
import type { AnalyzeResponse } from "../types";

export async function uploadAudio(file: File): Promise<AnalyzeResponse> {
  const form = new FormData();
  form.append("file", file);
  const { data } = await api.post<AnalyzeResponse>("/audio/upload", form);
  return data;
}
