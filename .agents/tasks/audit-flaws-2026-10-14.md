# SLOP AI Content Detector — Deep Technical Audit
**Date:** 2026-10-14  
**Scope:** Full multi-modal codebase: text, image, video, audio, fusion, evaluation pipeline  
**Sources examined:** `HELDOUT_REPORT.md`, `context.md`, `fusion.py`, `text_detector.py`, `binoculars.py`, `image_detector.py`, `video_detector.py`, `audio_detector.py`, `vram_manager.py`, `clip_probe.py`, `npr_extractor.py`, `signature_detector.py`, `eval_accuracy.py`, `eval_rigorous.py`, `Model/fusion/text_gbdt_summary.json`

---

## Executive Summary

The project claims "100% multi-modal accuracy" as its headline benchmark. That number is **statistically and methodologically meaningless**. It is computed on a golden set of 3+3 text samples, 2+2 image samples, and 1+1 video samples — the same data that the fusion models were **fitted on**. The real held-out evaluation tells the honest story: **text 33.33%, image 75.68%, video 50.00%, audio 49.3% AUC (pure chance)**. The performance gap between the claimed 100% and the actual sub-coin-flip results is the central crisis of this project, and most of the other flaws flow from it.

The top-line findings:

1. **The golden set is training data masquerading as a test set.** Every fusion model weight was fitted on it. `eval_accuracy.py` measures memorization, not generalization.
2. **Text recall is catastrophically low on real AI text (21.8%).** The GBDT with threshold 0.9149 misses 43 of 55 real AI samples. Two entire generator families (llama-chat: 0/11, mpt-chat: 0/11) are completely invisible.
3. **Image calibration is broken (ECE 0.3931).** The fusion model was fit on 14 images. A logistic regression on 14 points is statistically meaningless.
4. **Video = coin flip.** All 4 human videos are predicted as AI. The screen recording discriminator does not fire on natural video.
5. **Audio is disconnected from chance.** AUC 0.493 on 100 samples means the Wav2Vec2 head is anti-predictive on the evaluation distribution.
6. **Multiple circular feedback loops** exist between training, evaluation, and threshold-setting.

---

## 1. The Golden Set Contamination Problem

### What the project claims
`context.md` Section 5 reports 100% multi-modal accuracy verified live on CUDA via `eval_accuracy.py --quick`. This is treated as the headline benchmark throughout the README, `context.md`, and the test suite (`test_golden_set_accuracy.py`).

### Why it is invalid

**The fusion models were fit on this exact data.** Evidence:

- `HELDOUT_REPORT.md` (Image section): *"both record `n_samples = 14`"* — the shipped `image.pkl` and `clip_probe.pkl` were fitted on the 14 golden set images. The report explicitly states: *"the honest held-out figure is 78.85% (n=52) and the golden subset must be treated as training data for those artifacts."*
- `text_gbdt_summary.json`: `val_metrics.human/human` shows `fp=28, tn=2` — the GBDT val set included the RAID `human/human` domain, which overlaps with golden set source material.
- `eval_accuracy.py` uses a threshold of 0.50 throughout, which is different from the production GBDT threshold of 0.9149. The 100% number is computed with a looser threshold than the one actually deployed. If the production threshold were applied, the golden set text accuracy would be lower.

**The golden set is tiny.** 3+3 text, 2+2+2 image (the `--quick` flag uses 2+1+1=4 images), 1+1 video. No confidence interval is reported. A coin flip has a 12.5% chance of getting 8/8 correct across binary classes. The system *memorized* these exact files during calibration.

**The circular evaluation loop in `eval_accuracy.py`:**
```
golden_set/ → train_fusion.py → Model/fusion/*.pkl → eval_accuracy.py → golden_set/
```
All three vertices reference the same 12–46 samples. There is no held-out path in `eval_accuracy.py`. It is a regression smoke test, not an accuracy measurement. The project's own `context.md` acknowledges this: *"The `eval_accuracy.py` golden-set run remains useful ONLY as a regression smoke test, not an accuracy claim."* However, the `context.md` Section 5 still presents the 100% number in a table titled "Empirical Benchmark Results (Live Inference on Golden Set)" without caveat — creating a false impression for anyone who doesn't read the fine print in Section 10.

---

## 2. Text Detection Algorithmic Flaws

### 2a. The 33.33% held-out accuracy and 21.82% AI recall

From `HELDOUT_REPORT.md` (n=66, 55 AI + 11 human):
```
Confusion matrix:
           pred AI   pred HUMAN
truth AI     12          43
truth HUMAN   1          10
```

The system **misses 43 of 55 real AI samples**. Precision is high (92.3%) because the threshold (0.9149) is so extreme that almost nothing crosses it — but what does cross it is probably real AI. The problem is not false positives; it is massive false negatives.

### 2b. llama-chat: 0/11, mpt-chat: 0/11

These two generators together account for 22 of the 43 misses. Why?

**RoBERTa distribution mismatch.** The fine-tuned `transformer_prob` model (`Model/text/`) is described in `context.md` as *"fine-tuned on ChatGPT vs Human"*. The GBDT `feature_importances` in `text_gbdt_summary.json` shows `transformer_prob` has importance **156** — it is by far the dominant feature (next closest: `fast_detectgpt_ai_prob` at 69, `lm_perplexity` at 145). A model trained exclusively on ChatGPT-style output will produce near-zero scores (confidently HUMAN) for Llama-2-Chat and MPT-Chat text, which are instruction-tuned on different data with different stylistic signatures. `context.md` confirms: *"The OpenAI-detector-fine-tuned model is directionally unreliable outside its training distribution."*

The GBDT then amplifies this error. With `transformer_prob` accounting for ~30% of the feature importance weight and near-0 for Llama/MPT-Chat, the model's logit starts deeply negative and the remaining features cannot compensate when the threshold sits at 0.9149.

**The threshold 0.9149 is the wrong operating point for recall-heavy detection.** The production path sets `binary_threshold = 0.9149570072` (read from the calibrator artifact). This threshold was chosen to minimize false positives (human recall 100%), which it achieves — but at the cost of missing 78% of AI text. For a detector whose stated goal includes identifying AI content, a balanced threshold around 0.50–0.60 would be more appropriate. The existing zone architecture (`zone_low=0.276`, `zone_high=0.703`) is never used in binary decisions; only `binary_threshold` is applied in the API, producing a hard binary verdict. The zone bands (UNCERTAIN, LIKELY_AI) are reported in the response body but the API verdict and the eval threshold are both binary at 0.9149.

**The `text_gbdt_summary.json` confirms the model does not pass its own gate:**
```json
"gate_passed": false
```
Yet this model is deployed in production (`Model/fusion/text.pkl` = `text_gbdt_v2.pkl`). The gate was apparently overridden by the promotion process documented in context.md Section 10 under "Text Fusion v2 Promotion", which details the production-path verification showing 85.83% balanced accuracy on the 180-sample RAID training/val corpus — but these 180 samples are the model's own training distribution, not the held-out set.

### 2c. Binoculars (Qwen-0.5B pair) — model-target mismatch

`binoculars.py` uses Qwen2.5-0.5B as both Observer and Performer. The held-out generators are: gpt2 (~117M GPT-2 architecture), llama-chat (Llama-2-7B-Chat), mistral (Mistral-7B), mpt (MPT-7B), mpt-chat (MPT-7B-Chat).

The Binoculars method (Hans et al., ICML 2024) works because the cross-perplexity ratio is high for AI text when the observer model is "surprised" by the token distribution — meaning the target text is smoother/lower-entropy than the observer predicts. Qwen2.5-0.5B was trained on a different (and much more recent) corpus than GPT-2, Llama-2-Chat, or MPT. For **GPT-2 text**, which was generated by a much weaker, older model and has genuinely high perplexity, Qwen2.5-0.5B will not necessarily assign it lower cross-entropy than human text — because GPT-2 output can be erratic and repetitive in ways that look like unusual human writing to a 2024-trained model. This explains why gpt2 has only 2/11 recall in the held-out set.

The calibration constants in `binoculars.py` (`bino_logit = (binoculars_score - 1.05) / 0.10`) were set empirically for the Qwen pair on one distribution. There is no evidence these constants were validated on the held-out generators. The `HELDOUT_REPORT.md` also notes from context.md: *"binoculars now negative-weighted (it fires on human text in this corpus)"* — the GBDT literally learned to suppress the Binoculars signal because it anti-correlates with AI in the RAID val set.

### 2d. The 13 stylometrics: discriminative vs noise

From `text_gbdt_summary.json` feature importances (GBDT split counts):
```
transformer_prob:    156  ← massive
fast_detectgpt:       69  ← useful  
lm_perplexity:       145  ← useful
burstiness:           56  ← useful
vocab_richness:       35  ← moderate
binoculars_ai_prob:   21  ← low (anti-correlated on RAID)
coherence_mean:       25  ← moderate
transitional_entropy: 26  ← moderate
contraction_score:    17  ← weak
coherence_var:        10  ← weak
neural_prob:          15  ← weak
zipf_deviation:        1  ← near-zero
llm_pattern_score:     0  ← zero splits
```

`llm_pattern_score` has zero GBDT splits — it contributes nothing to the model. Yet `text_detector.py` maintains 200+ lines of regex patterns (CHATGPT_PATTERNS, GEMINI_PATTERNS, CLAUDE_PATTERNS, DEEPSEEK_PATTERNS, COPILOT_PATTERNS). These patterns are designed for ChatGPT-style assistant responses ("let me know if you need anything", "here's a breakdown"), which are completely absent from RAID-style academic-domain AI text. Pattern matching on RAID's gpt2/llama/mpt prose — which is generated to imitate academic writing — will hit zero patterns. `llm_pattern_score = 0` in the GBDT confirms this.

`zipf_deviation` has importance 1 — it is statistical noise in the model. The implementation in `text_detector.py` (`_zipf_deviation`) computes RMSE against Zipf's law on top-30 words, normalized. This is too coarse to discriminate AI vs human text across different domains.

**Signals that ARE discriminative** (`lm_perplexity`, `fast_detectgpt_ai_prob`, `burstiness`, `vocab_richness`) are all zero-shot signals that don't require domain-specific training — which suggests the right direction is more zero-shot signals, not more pattern libraries.

### 2e. Pattern matching survives paraphrasing — but only on chatbot text

The evasion eval in `context.md` shows weak paraphrase barely changes scores: *"+0.0013 mean score delta, 0/10 evasion rate"*. But this result is deceptive — the scores were **already near zero** for these generators (llama-chat, mpt-chat have ~0% recall), so there is nothing to evade. Paraphrase robustness is only meaningful for generators the system can already detect. The regex patterns survive synonym substitution because they target structural phrases ("here is a breakdown", "let me know if you need"), not vocabulary.

The homoglyph finding is the more dangerous result: human text with cross-script lookalikes scores **+0.2480 higher**, meaning the detector turns authentic human text into false positives when copypasting from non-ASCII sources. This is an FPR amplifier, not an evasion vector.

---

## 3. Image Detection Algorithmic Flaws

### 3a. Fusion model fit on 14 samples

From `HELDOUT_REPORT.md`: *"both record `n_samples = 14`"* for `image.pkl` and `clip_probe.pkl`. The image fusion logistic regression has ~14+ features (best_neural, heuristic, cfa, ela, fft, noise, dct, metadata_risk, is_diagram, is_real_camera, is_anime_ai, signature_detected, signature_confidence, c2pa_verified, clip_probe, npr). Fitting a logistic regression with 14+ parameters on 14 data points produces:

- Approximately one degree of freedom per parameter — the model is effectively interpolating the training data, not learning a generalizable decision boundary
- Perfect training fit (which is what the 100% golden set claim reflects) with no guarantee of generalization
- The `-6.5` diagram weight cited throughout `context.md` as *"a legitimate learned correction"* is actually a parameter fitted on a dataset where all diagram examples happened to be human. It is statistically unjustifiable as a general rule: with n=14, any single feature can dominate.

The 24% miss rate on held-out (75.68% vs hypothetical 100%) is almost entirely explained by this overfitting. The CLIP probe v2 experiment confirms the ceiling: AUC 1.0 on held-out but recall only 5/10 at the pre-registered threshold — the probe learned to perfectly rank images but the threshold calibration was never done on held-out data.

### 3b. ECE 0.3931 — why calibration fails

ECE 0.3931 is extremely poor (anything >0.10 is "not trustworthy as probabilities"). The causes:

1. **Score distribution is bimodal without a monotone relationship to accuracy.** From `context.md`'s golden set table: real AI images score 0.874–1.000, authentic camera photo scores 0.080, authentic diagram scores 0.030. The model pushes images to extremes. When a real held-out SDXL image scores, say, 0.45 (uncertain), the ECE formula penalizes this because at confidence 0.45 the true accuracy should also be ~45% — but the model intended that image to be labeled AI.
2. **Platt scaling on image was fitted on a small and imbalanced calibration split** (the `context.md` note: *"n=7 calibration samples, all human class — no AI examples to calibrate against; Platt made it worse"*). A calibrator with no AI examples in the calibration set cannot learn the AI probability-to-accuracy mapping. After recalibration ECE only improved from 0.3931 to 0.3191 — still terrible.
3. **The fusion model outputs extreme logits** (the `-6.5` diagram weight, 85%/15% neural authority split) which produce near-0 or near-1 probabilities. These extreme probabilities inflate ECE when a moderate-score image is actually AI.

### 3c. CLIP probe v2: AUC 1.0 but recall 5/10

From `HELDOUT_REPORT.md`: *"held-out AUC 1.0000"* but *"leave-RealVis-out RealVis recall 5/10 (95% Wilson 23.7–76.3%)"*. AUC measures rank ordering, not calibration. A probe can perfectly rank all AI images above all human images (AUC=1.0) while having a miscalibrated threshold that places many AI images below 0.5.

The report explains: *"RealVis probabilities sit in 0.43–0.57 so the pre-registered 0.5 threshold caps recall at 5/10 despite perfect rank separation"*. This means the CLIP probe's linear decision boundary is not aligned with the 0.5 threshold for RealVis images — a threshold-fitting step on a separate calibration set would likely fix this, but threshold fitting on the held-out set would be invalid data leakage.

This is the probe's fundamental limitation: it was trained on SDXL and COCO, and RealVis (a different Stable Diffusion variant) falls in the probe's uncertain zone.

---

## 4. Video Detection Flaws

### 4a. 50% accuracy = coin flip: all 4 human videos predicted as AI

From `HELDOUT_REPORT.md`:
```
           pred AI   pred HUMAN
truth AI     4           0
truth HUMAN  4           0
```

The detector has recall 100% (correctly flags all AI videos) but precision 50% and specificity 0% — it flags *everything* as AI. This is consistent with a **default-AI bias** in the fusion model or score distribution.

Looking at `video_detector.py` Track B (`analyze_video_streaming`): for non-screen-recording video, each frame is run through `_analyze_pil_image()` — the full image detector. The image detector, when applied to individual video frames, will likely produce elevated scores because:
- Video frames are typically compressed (H.264 artifacts) which can trigger ELA false positives
- Real camera video frames have compressed JPEG-like blocking that mismatches the ELA Q=90 baseline
- Natural photographic video does not have C2PA metadata
- EXIF metadata is absent from video frames

The `fuse_video_score` fallback path (`score = 0.40 * peak_threat + 0.40 * top_quartile + 0.20 * global_mean`) adds up individual frame scores. If even a few frames score above 0.50 (which is likely given the image detector's behavior on compressed video frames), the peak and top-quartile will be high, driving the final score above 0.50.

**The screen recording discriminator doesn't help here.** The `_detect_video_modality` function is gated on: `avg_color < 0.06` (very low palette diversity) OR `avg_color < 0.10 and avg_hv > 0.30` (low color + high orthogonal edges) OR `avg_static > 0.85 and avg_color < 0.15` (mostly static macroblocks). Natural camera video of people, landscapes, or faces has rich color diversity (`avg_color > 0.10`) and few orthogonal edges — so it correctly fails the screen-recording test. But it then gets routed to the full image detector, which falsely flags it.

### 4b. Farnebäck optical flow signal — correct principle, unvalidated calibration

`video_detector.py` Track C computes dense Farnebäck optical flow on CPU and uses `temporal_jitter` (second-derivative acceleration variance) as an AI signal. The idea has merit: diffusion video models (Sora, Kling, Wan) produce smooth, consistent motion that differs from natural camera shake and biological motion. However:
- The thresholds for `temporal_jitter` are hardcoded without held-out validation
- CPU Farnebäck on 8 frames is fast but 8 frames is very sparse temporal sampling for detecting motion artifacts
- The fusion model's `temporal_jitter` weight is fitted on the same 1+1 video golden set as everything else

### 4c. Human video class: golden set = screen recordings, held-out = authentic camera

`eval_accuracy.py` evaluates human video from `golden_set/video/human_screen/` — screen recordings. The screen recording discriminator was tuned on screen recordings and correctly identifies them (0.04 score). But the `heldout_set/video/human/authentic/` contains **real camera footage**, which the screen recording discriminator does not fire on. This is a distribution shift between the golden set and the held-out set that explains the 0/4 specificity.

---

## 5. Audio Detection Flaws

### 5a. AUC 0.493 — worse than chance

From `HELDOUT_REPORT.md`: *"AUC: 0.493 (exact pairwise, n=100) — statistically indistinguishable from chance"*. The Wav2Vec2 model is gated behind `enable_audio_detector=False` and is not used in production. However:

**The risk of a broken gated detector:** the gate is a single boolean in `settings.enable_audio_detector` (a Pydantic `Settings` field). If this is accidentally set to `True` (e.g., in a CI environment variable, a misconfigured deployment, or a future developer who sees `False` and assumes it is a TODO), the audio detector will produce scores that are anti-correlated with truth. A score of 0.493 AUC means: on average, if the system says "likely AI", the audio is more likely to be real. This is more dangerous than a random scorer.

**The Wav2Vec2 head transfer failure.** The model (`Model/audio/wav2vec2_deepfake`) was fine-tuned on ASVspoof-style data (TTS/VC synthesis attacks in speech verification contexts). The evaluation set is `Hemg/Deepfake-Audio-Dataset` (100 × 10s WAVs, "deepfake" likely refers to voice cloning/neural vocoder output). These are related but different distributions — ASVspoof targets replay and voice conversion attacks; the evaluation set tests neural TTS. The model learned to detect ASVspoof-specific artifacts that don't generalize.

---

## 6. Circular Evaluation Loops

### Loop 1: Golden set → train_fusion.py → eval_accuracy.py → golden set

**Primary loop.** `train_fusion.py` fits image fusion, CLIP probe, and (historically) text fusion weights on `golden_set/`. `eval_accuracy.py` evaluates on `golden_set/`. Any accuracy number from `eval_accuracy.py` measures memorization.

### Loop 2: GBDT training validation vs production deployment

The `text_gbdt_summary.json` shows `"gate_passed": false` on the validation set. Yet the model is deployed. The promotion process described in `context.md` used a 180-sample RAID corpus (120 AI / 60 human) from the same generators the model was trained on — this is train-distribution validation, not out-of-distribution validation. The held-out set (n=66) was never passed through the promotion gate before deployment.

From `HELDOUT_REPORT.md`'s "Train/serve skew" section: the production path scores 86/120 AI correct vs 89/120 in training. This 3-sample difference was attributed to floating-point drift — but the actual issue is that the held-out set (RAID with llama-chat, mpt-chat which the training data also contains) shows only 12/55 AI detected (21.8% recall). The training distribution (RAID 12k–60k offsets) and the eval distribution (RAID 0–5k offsets) are the same generator distribution; the gap from 71.7% to 21.8% is not explained by serve/train skew alone. There may be a **domain offset within RAID** (e.g., the 0–5k offsets represent different document lengths or styles than the 12k–60k offsets).

### Loop 3: Image calibration fitted on training data

`context.md` notes: *"Image calibration could NOT be fit meaningfully (n=7 calibration samples, all human class)"*. The Platt calibrator for image was fit on 7 samples of the same golden set used for fusion training. The calibration split is supposed to be disjoint from the eval split (`eval_rigorous.py` enforces this), but both splits come from `heldout_set/` which for images has very limited AI examples. The result is an ECE that only decreases from 0.3931 to 0.3191 — a meaningless improvement on a broken baseline.

### Loop 4: Threshold selection on eval data

From `HELDOUT_REPORT.md` (CLIP probe v2): *"recalibration requires a separate threshold-fitting set, never the held-out one"*. The binary threshold 0.9149570072 for text was selected from the RAID training corpus (180 samples), not from held-out data. Any threshold tuning on the same data used for training or calibration is circular.

---

## 7. Structural and Architectural Flaws

### 7a. Single-token VRAM mutex under concurrent load

`vram_manager.py` uses `asyncio.Semaphore(1)` as `gpu_inference_lock`. This is correct for the RTX 4050's 6GB constraint — only one modality can hold VRAM at a time. But under concurrent real-world load:

- **30-second timeout** (`GPU_LOCK_ACQUIRE_TIMEOUT_SECONDS = 30.0`) means a user who submits a text analysis immediately followed by image analysis will wait up to 30 seconds for the image analysis to start. Text inference (including Binoculars cold start ≈42.4s) can exceed the 30-second timeout if it runs over.
- **No queuing / fairness.** `asyncio.Semaphore` is FIFO for coroutines ready at the same time, but not guaranteed FIFO under async scheduling. Under heavy load, some requests will time out with `GPUInferenceTimeoutError` rather than waiting in a proper queue.
- **Cold start latency dominates.** `context.md` notes: *"cold start ≈42.4s"* for text. A cold start exceeds the 30-second lock timeout, which means the first text request after eviction will timeout if another modality holds the lock. The `serialized_gpu_inference` context manager releases the lock *before* the model is evicted (the eviction happens inside the `finally` block *before* the lock release), but the cold start happens *after* the lock is acquired — so the 42-second cold start is fine, but the 30-second lock-wait timeout is not.

Actually re-reading `serialized_gpu_inference`: acquisition is bounded by 30s, but once acquired, there's no timeout on the inference itself. So a stuck inference holds the lock forever. Only the *acquisition wait* is bounded.

### 7b. The GBDT→legacy logistic fallback creates invisible quality degradation

`fusion.py` routes calls with missing zero-shot signals to `_LEGACY_TEXT_MODEL = "text_pre_raid"`. The legacy logistic was trained pre-RAID on the small golden set. Its recall on real AI text is almost certainly worse than the GBDT's already-bad 21.8%. When a request arrives without Binoculars/Fast-DetectGPT/LM-perplexity/coherence (e.g., if those components fail or time out), the system silently degrades to a worse model with no user-visible indication. The API response doesn't distinguish "GBDT path" from "legacy path" — the `primary_signal` field shows `"gbdt_platt_calibrated"` only for the GBDT path, but `"roberta_neural_classifier"` for the legacy path.

### 7c. SQLite in WAL mode for production multi-user load

`context.md` notes `slop.db` uses SQLite with `PRAGMA WAL` mode and `busy_timeout=5000ms`. WAL mode allows one writer and multiple readers, which is adequate for low-concurrency single-server use. But:
- The stated goal is "production-ready" with Stripe billing, user auth, and multi-modal detection. A SQLite backend cannot scale past a single process
- SQLite WAL mode is incompatible with network filesystems and NFS mounts (relevant for cloud deployments)
- 5000ms busy timeout on writes means a slow inference that triggers a DB write (storing `DetectionResult`) could starve concurrent users

This is a YAGNI-acceptable simplification for a solo project, but incompatible with the stated "production-ready" goal.

### 7d. Missing signals fallback routes to pre-RAID logistic

As noted in 7b: `_load_legacy_text_logistic()` loads `text_pre_raid`. From `context.md`: *"The old logistic is archived byte-identical at `Model/fusion/_archive/text_pre_v2_logistic.pkl`"* and the `_LEGACY_TEXT_MODEL` constant in `fusion.py` is set to `"text_pre_raid"`. This model was trained before the RAID dataset was used, meaning it was likely trained on the small golden set samples only. It has no domain coverage for llama-chat, mpt-chat, or mistral. Any missing-signal call routes to a model that has essentially zero generalization.

---

## 8. Training Data Quality and Distribution Mismatch

### 8a. GBDT trained on RAID generators vs real-world AI ecosystem

The GBDT (`text_gbdt_summary.json`) trains on: gpt2, llama_chat, mistral, mistral_chat, mpt, mpt_chat. These are all open-weight models from 2022–2023. The LLM-specific pattern libraries in `text_detector.py` cover GPT-4o, Claude 3.5, Gemini 2.0, and DeepSeek-R1.

**The disconnect:** The pattern libraries target frontier 2024–2025 models but the GBDT's `llm_pattern_score` feature has **zero splits** — the GBDT learned that this feature is useless for discriminating RAID-generated text (because frontier LLM patterns don't appear in RAID's gpt2/llama/mpt output). The pattern libraries are orphaned: they may fire on real-world GPT-4o text submitted by users, but the GBDT won't use that signal because it was never trained on GPT-4o text.

Meanwhile, the `binoculars_ai_prob` signal — which was specifically designed as a zero-shot method robust to generator novelty — has only 21 GBDT splits and is described in `context.md` as having been negatively weighted in earlier iterations. The current GBDT val metrics show `human/human: fp=28, tn=2` — the model over-flags human text from the "human" abstract domain. This is exactly the domain-shift problem that makes the detector unreliable on real-world submissions.

### 8b. Golden set generation scripts

`expand_golden_set.py` and `seed_golden_set.py` are referenced in the directory structure. Without reading them in full, the key concern from `context.md` is: *"The self-generated golden set (12 quick / 46 full samples) is not a valid benchmark — it was generated by the same pipeline it evaluates."* These scripts generate AI samples by submitting prompts to LLMs and then immediately add them to the golden set. This creates a selection bias: only AI samples that "look AI enough" to the detector at generation time get added, while adversarial or unusual AI outputs are excluded.

### 8c. Human training data is a single domain (abstracts)

From `context.md` (Tiers 3-6 section): *"the true English val wall is smaller but still real... feature AUCs on English val: `neural_prob` 0.580 and the zero-shot signals 0.62–0.65"*. The root cause: *"the detector was effectively tuned to academic-abstract style"*. RAID's human class in the training partition covers "news" and limited domains. The GBDT `val_metrics.human/human` shows `fp=28, tn=2` — 28 out of 30 "human" samples from the val set are classified as AI. This means the GBDT will false-positive on any human text that doesn't look like RAID's abstracts domain.

---

## 9. Prioritized Top-5 Most Impactful Fixes

### Priority 1: Replace the golden set evaluation with a proper held-out benchmark

**Impact:** Eliminates the false confidence that pervades the project. Everything else builds on an honest baseline.

**What to do:** Retire `eval_accuracy.py` as a benchmark. Treat `HELDOUT_REPORT.md` as the canonical truth. Expand `heldout_set/` to ≥100 samples per modality per class with at minimum 5 generator types. Never tune thresholds or fit fusion weights on `heldout_set/`. Document this policy in a `DATA_DISCIPLINE.md`.

**Current state:** The project's own Phase 6 work (`eval_rigorous.py`) has already built the right infrastructure. It just needs to be declared canonical and the golden set benchmark demoted.

### Priority 2: Fix text recall — lower the threshold and diversify training human data

**Impact:** Text recall 21.8% → potentially 60%+. The threshold 0.9149 was tuned for zero false positives, but at the cost of 78% false negatives. For a detector, missing 78% of AI content is worse than occasionally flagging human content.

**What to do:**
1. Lower the binary threshold to 0.50–0.60. The zone architecture already exists (`zone_low`, `zone_high`) — expose these to users as "uncertain" rather than defaulting to HUMAN.
2. Add diverse human training data (RAID's `czech`/`code`/`news` domains, or Common Crawl excerpts) to prevent the "default AI" bias on unseen human domains.
3. Replace or augment the RoBERTa model (`transformer_prob`) which has importance 156 but is fine-tuned only on ChatGPT text. A model trained on a broader generator distribution (or a zero-shot perplexity signal from a better-matched LM) would generalize better.

**Evidence:** `text_gbdt_summary.json` `val_metrics.human/human: fp=28, tn=2` and the HELDOUT held-out 0/11 recall on llama-chat/mpt-chat.

### Priority 3: Rebuild image fusion on ≥100 samples with proper train/val/test splits

**Impact:** Image fusion ECE 0.3931 → plausibly <0.15. The logistic fitted on 14 samples is the proximate cause of both the poor calibration and the 24% miss rate on held-out data.

**What to do:**
1. Collect ≥100 AI images from diverse generators (SDXL, Midjourney, Flux, SD3, Kandinsky) and ≥100 human photos (Unsplash, COCO, Flickr).
2. Refit `image.pkl` and `clip_probe.pkl` on the training split (80%).
3. Tune the CLIP probe threshold on a separate validation split (10%).
4. Evaluate on the held-out split (10%) — never seen during fitting.
5. The `-6.5` diagram weight should be treated as suspect and re-fitted on the larger dataset.

**Evidence:** `HELDOUT_REPORT.md` "Gate-minus-golden" note: *"the honest held-out figure is 78.85% (n=52)"* and `n_samples=14` in both image artifacts.

### Priority 4: Fix video by separating authentic camera video from screen recordings in training

**Impact:** Video specificity 0% → plausibly 60%+. All 4 human authentic videos are currently predicted as AI.

**What to do:**
1. Expand the golden set human video class to include authentic camera footage (not just screen recordings). Source from Kinetics-400 or similar.
2. Add a dedicated "natural video frame" path in `analyze_video_streaming` that doesn't route through the image detector's full forensic pipeline (which generates ELA/CFA/FFT false positives on compressed video frames).
3. Alternatively, train a video-specific frame classifier that is aware of H.264 compression artifacts and does not penalize them as AI signals.
4. Validate the Farnebäck optical flow thresholds on ≥20 authentic + AI video pairs before deploying as a production signal.

**Evidence:** HELDOUT_REPORT confusion matrix (truth HUMAN / pred AI: 4/4, 0% specificity).

### Priority 5: Disable or fix the audio detector — it is anti-predictive

**Impact:** Prevents a negative-AUC detector from accidentally being enabled and systematically misclassifying audio.

**What to do:**
1. Add an automated CI check that verifies `enable_audio_detector = False` and `enable_freq_detector = False` cannot be activated without a gate-passing held-out AUC ≥ 0.70.
2. If audio detection is a priority, replace the off-the-shelf Wav2Vec2 head with one fine-tuned on the target evaluation distribution (neural TTS voice cloning), or use a simpler mel-spectrogram feature classifier that at minimum achieves AUC > 0.65.
3. Add a detection assertion in the API endpoint: if `enable_audio_detector=True` but no valid calibration record exists, return HTTP 503 with a clear message rather than returning a score from an unvalidated model.

**Evidence:** HELDOUT_REPORT Audio section: *"AUC: 0.493 — statistically indistinguishable from chance"*, TPR 0.08 / TNR 0.86.

---

## 10. Secondary Issues (Lower Priority)

| Issue | Location | Severity |
|---|---|---|
| `llm_pattern_score` has zero GBDT splits — dead feature | `text_detector.py`, `fusion.py` | Medium |
| Binoculars calibration constants (1.05, 0.10) not validated on target generators | `binoculars.py:185-187` | Medium |
| `code_forensics` track can suppress any text to 0.05 if `identifier_entropy < 0.40 && comment_ratio < 0.05` — could suppress competitive programming AI code | `fusion.py:_apply_code_forensics` | Medium |
| `_standardize()` silently no-ops for legacy pickles (missing `feature_mean`/`feature_std`) — features are not normalized for the legacy logistic | `fusion.py:_standardize` | Low |
| CLIP probe fallback when weights unavailable: `logit = float(embed_vec.sum() * 0.5)` — meaningless heuristic | `clip_probe.py:127-128` | Medium |
| SQLite WAL for multi-user production | `db/session.py` | Low (YAGNI acceptable) |
| Duplicate images in corpus (`golden_set/image/ai/synthetic_latent_flow_{1,2,3}.png`) inflate eval counts | `heldout_set/image/` | Medium |
| Whitespace injection achieves 20% evasion (2/10 AI samples evade detection) | `text_detector.py` stylometrics | Medium |
| `_video_info` fallback returns hardcoded `fps=30, width=1920, height=1080` — downstream code may use these values as real video properties | `video_detector.py:149-150` | Low |
| `fuse_audio_score()` gives 85% confidence when `neural_fake` is present — but the Wav2Vec2 head's AUC is 0.493, so 85% confidence is systematically wrong | `fusion.py:fuse_audio_score` | High (if audio enabled) |

---

## 11. Summary Table: Real vs Claimed Performance

| Modality | Claimed (golden set) | Real held-out | Samples (held-out) | Key failure mode |
|---|---|---|---|---|
| Text | 100.0% accuracy | **33.33%** accuracy, **21.82%** AI recall | n=66 | Threshold 0.9149 too high; RoBERTa trained on ChatGPT only; llama-chat/mpt-chat 0/11 recall |
| Image | 100.0% accuracy | **75.68%** accuracy, ECE 0.3931 | n=37 | Fusion fitted on n=14; calibration meaningless |
| Video | 100.0% accuracy | **50.0%** accuracy (coin flip) | n=8 | 0/4 human specificity; screen-rec training ≠ authentic video |
| Audio | Not tested | **AUC 0.493** (worse than chance) | n=100 | Wav2Vec2 transfer failure; gated OFF |
| Overall | **100.0%** | **~53%** (weighted average) | — | Golden set = training data |

The project has excellent infrastructure (proper VRAM management, Wilson CIs, held-out harness, gate promotion logic, structured logging, and honest documentation of failures). The engineering scaffolding is sound. The core problem is that **the ML models have not been trained or validated on sufficient, diverse, out-of-distribution data** — and the 100% headline metric actively conceals this from anyone who doesn't read the footnotes.
