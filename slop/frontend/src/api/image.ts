import { api } from "./client";
import type { AnalyzeResponse } from "../types";

export async function analyzeImage(file: File) {
  const form = new FormData();
  form.append("file", file);
  const { data } = await api.post<AnalyzeResponse>("/image/analyze", form);
  return data;
}
