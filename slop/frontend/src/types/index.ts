export type DetectorMode = "text" | "image" | "live_image" | "video" | "document" | "audio";

export type BoundingBox = {
  ymin: number;
  xmin: number;
  ymax: number;
  xmax: number;
};

export type DigitalSignatureInfo = {
  detected: boolean;
  signature_type: string;
  confidence: number;
  generator: string;
  generator_confidence: number;
  bounding_box?: BoundingBox;
  details?: string;
};

export type GeneratorAttribution = {
  generator: string;
  confidence: number;
  details?: string;
  signature_type?: string;
  bounding_box?: BoundingBox;
};

// v3.0: LLM Attribution
export type LLMAttribution = {
  detected_llm: string | null;
  confidence: number;
  scores: Record<string, number>;
  total_pattern_matches: number;
  all_patterns?: string[];
};

export type SentenceScore = {
  text: string;
  ai_prob: number;
  perplexity: number;
  model_ai_prob?: number;
  pattern_matched?: boolean;
};

export type FrameScore = {
  timestamp: number;
  score: number;
  has_face: boolean;
  face_count?: number;
  temporal_change?: number;
};

// v4.1: Mixed Authorship Information
export type MixedAuthorshipInfo = {
  transitions: number;
  mixed_ratio: number;
  ai_sentence_ratio: number;
  is_mixed_authorship: boolean;
};

export type DetectionResult = {
  overall_score: number;
  signals: Record<string, unknown>;
  job_id?: string;
  confidence_band?: string;
  confidence_range?: number[];
  zone_low?: number;
  zone_high?: number;
  section_scores?: { index: number; word_count: number; score: number; preview: string }[];
  sentences?: SentenceScore[];
  frame_scores?: FrameScore[];
  heatmap_b64?: string;
  gradcam_heatmap_b64?: string;  // v3.0: GradCAM explainability
  prnu_heatmap_b64?: string;      // v3.4: PRNU sensor noise residual map
  model_hint?: string;
  heuristic_override_explanation?: string;
  video_info?: Record<string, unknown>;
  digital_signature?: DigitalSignatureInfo;
  generator_attribution?: GeneratorAttribution;
  llm_attribution?: LLMAttribution;  // v3.0: Which LLM wrote it
  mixed_authorship?: MixedAuthorshipInfo; // v4.1: Blended human+AI authorship
};

export type AnalyzeResponse = {
  job_id: string;
  status: string;
  result?: DetectionResult;
};

export type User = {
  id: string;
  email: string;
  is_active: boolean;
  is_verified: boolean;
  totp_enabled: boolean;
};
