---
name: slop-production-engine
description: Enforces production-grade ML implementation, strict VRAM budgeting, zero-mock execution, and automated verification for the SLOP detector.
---

# SLOP Production Engine Directives

## 1. Zero-Mock & Anti-Shortcut Doctrine
- NEVER mock inference outputs, return hardcoded heuristic strings, or generate synthetic evaluation scores.
- Every ML pipeline modification MUST execute real forward passes using PyTorch and report live outputs.
- If a model is not downloaded or CUDA is uninitialized, raise an explicit exception (`RuntimeError`). Never degrade to a silent fallback or fake placeholder without logging.
- Verify every file modification on disk immediately after generation. Check syntax and imports via the CLI before declaring a task complete.

## 2. Hardware Budget & Memory Management (RTX 4050 6GB)
- Strict VRAM ceiling: 5.0 GB total allocation across all active models.
- Never hold all three modalities (Text, Image, Video) in GPU memory simultaneously.
- Enforce explicit context eviction and cache clearing:
  ```python
  import gc
  import torch

  def evict_model(model):
      del model
      gc.collect()
      torch.cuda.empty_cache()
  ```
- Large text models (e.g., Qwen 1.5B or Falcon) MUST use `bitsandbytes` 4-bit (NF4) quantization.
- Heavy generative inversion (e.g., DDIM/DIRE) is strictly banned due to VRAM thrashing.

## 3. Architecture & Contract Enforcement
- Backend: FastAPI async endpoints under `slop/backend/app/api/v1/`.
- Frontend: React 18 + Vite under `slop/frontend/src/` adhering to the cyber-slate palette (`#07090e`, `#f43f5e`, `#10b981`).
- Database: Async SQLAlchemy 2.0 sessions (`sqlite+aiosqlite`). Always commit sessions inside context managers.
- Error handling: Every endpoint must return RFC 7807 compliant error payloads.

## 4. Verification & Testing Workflow
- For any backend route change:
  1. Write or update a test in `tests/`.
  2. Execute the test using `pytest` via the terminal.
  3. Validate that `torch.cuda.memory_allocated()` returns a non-zero value during active inference.
- For UI updates:
  1. Use the `webapp-testing` skill to trigger a headless Playwright run.
  2. Verify that `/api/v1/image/analyze` renders the GradCAM heatmap canvas without layout shift.

## 5. Multi-Spectral Image & Modern LLM Forensics Pipeline
- **PRNU Sensor Noise Residual**:
  Extract normalized sensor residual $R = I - \text{Filter}(I)$ to isolate physical CMOS/CCD silicon grain from diffusion latents.
- **Farnebäck Temporal Coherence**:
  Compute optical flow inter-frame acceleration variance across keyframe sequences to catch generative temporal morphing with zero VRAM overhead.
- **Frontier LLM Calibration**:
  Evaluate token transitional entropy consistency and calibrate against 2025/2026 models (GPT-4o, Claude 3.5 Sonnet, Gemini 1.5/2.0, DeepSeek-R1) to prevent false-negative classifier suppression.

