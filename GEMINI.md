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
- Frontend: React 18 + Vite under `slop/frontend/src/`.
- Database: Async SQLAlchemy 2.0 sessions (`sqlite+aiosqlite`). Always commit sessions inside context managers.
- Ponytail Full Mode: Intentional simplifications marked with `# ponytail:` comments. Standard library and native features prioritized over bloat.
- Mandatory CLI verification: Execute tests via CLI and check exit codes before completing any step.

## 4. Frontend Dual-Theme & Accessibility Directives
- Semantic Token System: All components MUST use CSS theme variables (`--bg-page`, `--bg-card`, `--text-main`, `--border-card`) instead of hardcoded dark hex codes.
  * Cyber Slate (Dark): Background `#07090e`, Cards `#0d121c`, Borders `#1e293b`, Text `#f8fafc`.
  * Studio Clean (Light): Background `#f8fafc`, Cards `#ffffff`, Borders `#e2e8f0`, Text `#0f172a`.
  * Both modes must strictly maintain WCAG AA contrast (no white text on light backgrounds or dark boxes in light mode).
- Sticky Dual-Pane Invariant: The detector layout (`.detector-grid`) must use `align-items: start !important`. Ancestor containers MUST NOT use `overflow: hidden` or `overflow-x: clip`, which disable sticky scrolling.
- Compositor-Only Animations: Animate exclusively GPU-composited properties (`transform`, `opacity`). Never animate layout dimensions or colors.

## 5. AI Agent & Pipeline Security Directives
- Least-Privilege Agent Guardrails: AI Agents MUST NEVER read, edit, or expose `.env`, `.pem`, `.key`, or RSA private keys in `keys/`. Never write files outside the workspace root.
- Input Normalization & Sanitization: Every incoming text analysis MUST execute Unicode NFKC normalization and strip invisible/zero-width formatting characters (`\u200b`, `\u200c`, `\ufeff`, `\u2060`) before processing.
- Injection Awareness & Non-Execution: Pipelines MUST detect and flag prompt injection or jailbreak payloads in metadata without executing or propagating them to downstream agents or tools.

## 6. Multi-Modal Calibration & Forensic Dual-Track Directives
- Text vs. Code Dual-Track: Incoming text must undergo code syntax parsing.
  * If syntax exhibits code structures (Python, JS, C++), route to the **Code Forensics Track**: evaluate identifier entropy, formatting regularity, and comment density. Bypasses natural prose Zipf's law and burstiness penalties.
  * Categorize code explicitly as **"Manual Coded (Human / Competitive)"** (< 0.20 AI) or **"Vibe Coded (AI / Assistant Generated)"** (> 0.70 AI).
- Neural-First Consensus Authority: In multi-modal ensembles (Swin Transformer, RoBERTa, Heuristics), the deep neural model holds primary authority (80%+).
  * Heuristic classifiers for diagrams or authentic camera photos must NEVER override or clamp an image if the neural vision model detects synthetic generative features with confidence (`>= 0.55`).
  * Diagram suppression is strictly limited to low-evidence neural ranges (`< 0.40`).
- PRNU Sensor Noise Mapping: Image detection endpoints MUST generate a normalized Photo-Response Non-Uniformity (PRNU) sensor residual heatmap alongside GradCAM and ELA to distinguish real camera silicon imperfections from synthetic diffusion latents.