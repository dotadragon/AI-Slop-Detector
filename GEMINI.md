# Gemini Operational Directives — SLOP Project

## 1. Zero-Mock & Anti-Shortcut Doctrine
- NEVER mock inference outputs, return hardcoded heuristic strings, or generate synthetic evaluation scores.
- Every ML pipeline modification MUST execute real forward passes using PyTorch and report live outputs.
- If a model is not downloaded or CUDA is uninitialized, raise an explicit exception (`RuntimeError`). Never degrade to a silent fallback or fake placeholder without logging.
- Verify every file modification on disk immediately after generation. Check syntax and imports via the CLI before declaring a task complete.

## 2. Hardware Budget & Memory Management (RTX 4050 6GB)
- Strict VRAM ceiling: 5.0 GB total allocation across all active models.
- Never hold all three modalities (Text, Image, Video) in GPU memory simultaneously.
- Enforce dynamic context eviction and cache clearing (`torch.cuda.empty_cache()`).
- Large text models (e.g., Qwen 1.5B or Falcon) MUST use `bitsandbytes` 4-bit (NF4) quantization.
- Heavy generative inversion (e.g., DDIM/DIRE) is strictly banned due to VRAM thrashing.

## 3. Architecture & Verification Workflow
- Backend: FastAPI async endpoints under `slop/backend/app/api/v1/`.
- Frontend: React 18 + Vite under `slop/frontend/src/` adhering to the cyber-slate palette (`#07090e`, `#f43f5e`, `#10b981`).
- Database: Async SQLAlchemy 2.0 sessions (`sqlite+aiosqlite`). Always commit sessions inside context managers.
- Ponytail Full Mode: Intentional simplifications marked with `# ponytail:` comments. Standard library and native features prioritized over bloat.
- Mandatory CLI verification: Execute tests via CLI and check exit codes before completing any step.

## 4. Frontend Performance & Accessibility Directives
- Landmark & Hierarchy: Every page view MUST mount a `<main>` landmark with strictly sequential heading levels (`<h1>` -> `<h2>` -> `<h3>`).
- Compositor-Only Animations: Animations MUST animate only GPU-composited properties (`transform`, `opacity`). Never animate `top`, `left`, `margin`, `color`, or `box-shadow`.
- Zero-Shift Font Loading: Web fonts MUST use preconnected stylesheets with `font-display: swap` without print-media hacks that trigger CLS.
- On-Demand Observability: Heavy tracking/monitoring SDKs (Sentry, PostHog) MUST be dynamically loaded only when credentials exist and user consent is confirmed.

## 5. AI Agent & Pipeline Security Directives
- Least-Privilege Agent Guardrails: AI Agents MUST NEVER read, edit, or expose `.env`, `.pem`, `.key`, or RSA private keys in `keys/`. Never write files outside the workspace root.
- Input Normalization & Sanitization: Every incoming text analysis MUST execute Unicode NFKC normalization and strip invisible/zero-width formatting characters (`\u200b`, `\u200c`, `\ufeff`, `\u2060`) before stylometric or neural processing.
- Injection Awareness & Non-Execution: Pipelines MUST detect and flag prompt injection or jailbreak payloads in metadata without executing or propagating them to downstream agents or tools.
- Multi-Tenant & Session Hygiene: User session contexts must be scoped (`user_id:key`) and explicitly wiped on session termination to prevent data leakage.

## 6. Multi-Modal Frontier Calibration & Spectral Visualization Directives
- Frontier LLM Calibration: Text analysis pipelines MUST calibrate neural outputs against 2025/2026 frontier LLM signatures (GPT-4o, Claude 3.5 Sonnet, Gemini 1.5/2.0, DeepSeek-R1) using Binoculars-style cross-ratio variance and lexical entropy. Old classifier suppression must never produce a false negative on high-evidence synthetic text.
- PRNU Sensor Noise Mapping: Image detection endpoints MUST generate a normalized Photo-Response Non-Uniformity (PRNU) sensor residual heatmap alongside GradCAM and ELA to distinguish real camera sensor silicon imperfections from synthetic diffusion latents.
- HUD Performance: All interactive forensic visualizers (radar gauges, token highlights, frequency spectrograms) must animate exclusively via GPU-composited properties (`transform`, `opacity`) with zero cumulative layout shift (CLS).
- Neural-First Consensus Authority: In multi-modal ensembles combining deep neural networks (Swin Transformer, RoBERTa) with forensic heuristics (Bayer CFA, FFT, ELA, orthogonal edge ratios), the deep neural model MUST hold primary authority (80%+). Heuristic classifiers for diagrams or authentic camera photos must NEVER override or clamp an image if the neural vision transformer detects synthetic generative features with confidence (>= 0.70). Heuristics serve exclusively as corroborating evidence or as authentic confirmation when neural AI probability is below 0.35.
