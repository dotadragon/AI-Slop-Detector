import { api } from "./client";
import type { AnalyzeResponse } from "../types";

export async function analyzeText(text: string) {
  const { data } = await api.post<AnalyzeResponse>("/text/analyze", { text });
  return data;
}

export async function uploadText(file: File) {
  const form = new FormData();
  form.append("file", file);
  const { data } = await api.post<AnalyzeResponse>("/text/upload", form);
  return data;
}
