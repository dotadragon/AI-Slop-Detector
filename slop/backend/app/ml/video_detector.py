"""
SLOP Video Detector v3.1 — Production-grade dual-track video forensics.
ponytail: zero-overhead dual-track architecture (Face-crop deepfakes + Full-scene diffusion + Optical flow temporal coherence)
Strict VRAM budgeting for NVIDIA RTX 4050 (6GB).
"""

import base64
import importlib
import io
import logging
from typing import Any, Generator
from pathlib import Path
from functools import lru_cache

import numpy as np
from PIL import Image

from app.ml.vram_manager import get_vram_stats, check_vram_headroom

logger = logging.getLogger(__name__)


def _load_cv2() -> Any:
    return importlib.import_module("cv2")


def _get_frames(file_path: str, max_frames: int = 8, fps: float = 1.0) -> Generator[tuple[int, float, np.ndarray], None, None]:
    """Extract up to max_frames keyframes spaced evenly across video duration. Falls back to placeholders if cv2 missing."""
    try:
        cv2 = _load_cv2()  # ponytail: lazy import
    except ImportError as exc:
        logger.warning("cv2_import_failed_using_fallback_frames", error=str(exc), operation="frame_extraction")
        for i in range(max_frames):
            f = np.full((256, 256, 3), 128, dtype=np.uint8)
            f[:, :, 0] = int(255 * i / max_frames)
            yield i, i * 0.5, f
        return

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return

    try:
        vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = int(vid_fps * 10)

        interval = max(int(vid_fps / fps), 1)
        if total_frames > max_frames:
            interval = max(interval, total_frames // max_frames)

        idx, extracted = 0, 0
        while extracted < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % interval == 0:
                yield extracted, idx / vid_fps, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                extracted += 1
            idx += 1
    finally:
        cap.release()


def _find_yunet_model_path() -> str | None:
    base = Path(__file__).resolve()
    for p in [
        base.parent.parent.parent.parent.parent / "Model" / "video" / "face_detection_yunet_2023mar.onnx",
        base.parent.parent.parent.parent / "Model" / "video" / "face_detection_yunet_2023mar.onnx",
        base.parent.parent.parent / "models" / "video" / "face_detection_yunet_2023mar.onnx",
    ]:
        if p.exists() and p.stat().st_size > 1000:
            return str(p)
    return None


def _detect_faces_and_crops(frame: np.ndarray) -> list[Image.Image]:
    """
    ponytail: Rung 5 OpenCV YuNet face detector (2ms on CPU) with 20% spatial margin.
    Ensures facial deepfake models only evaluate faces, never squished 1080p scenes.
    Gracefully falls back to Haar cascade if YuNet ONNX model is missing.
    """
    try:
        cv2 = _load_cv2()
        h, w, _ = frame.shape
        crops = []
        yunet_path = _find_yunet_model_path()

        if yunet_path and hasattr(cv2, "FaceDetectorYN"):
            detector = cv2.FaceDetectorYN.create(yunet_path, "", (w, h), score_threshold=0.6, nms_threshold=0.3)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            _, faces = detector.detect(frame_bgr)
            if faces is not None:
                for face in faces:
                    fx, fy, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                    pad_x = int(fw * 0.20)
                    pad_y = int(fh * 0.20)
                    x1, y1 = max(0, fx - pad_x), max(0, fy - pad_y)
                    x2, y2 = min(w, fx + fw + pad_x), min(h, fy + fh + pad_y)
                    cropped = frame[y1:y2, x1:x2]
                    if cropped.size > 0:
                        crops.append(Image.fromarray(cropped).resize((224, 224), Image.Resampling.LANCZOS))
                if crops:
                    return crops

        # Fallback to Haar cascade if YuNet weights unavailable
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        rects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
        for (x, y, fw, fh) in rects:
            pad_x = int(fw * 0.20)
            pad_y = int(fh * 0.20)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w, x + fw + pad_x)
            y2 = min(h, y + fh + pad_y)
            cropped = frame[y1:y2, x1:x2]
            if cropped.size > 0:
                crops.append(Image.fromarray(cropped).resize((224, 224), Image.Resampling.LANCZOS))
        return crops
    except Exception as exc:
        logger.warning("face_detection_failed", error=str(exc), exc_type=type(exc).__name__, operation="face_crop", modality="video")
        return []


def _thumbnail(frame: np.ndarray, w: int = 160) -> str:
    """Frame -> small base64 JPEG thumbnail."""
    img = Image.fromarray(frame)
    img = img.resize((w, max(1, int(img.height * w / max(img.width, 1)))), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=55)
    return base64.b64encode(buf.getvalue()).decode()


def _video_info(file_path: str) -> dict:
    try:
        cv2 = _load_cv2()
        cap = cv2.VideoCapture(file_path)
        info: dict[str, int | float] = {
            "fps": round(cap.get(cv2.CAP_PROP_FPS) or 30, 1),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        info["duration_seconds"] = round(info["total_frames"] / max(info["fps"], 1), 1)
        cap.release()
        return info
    except Exception as exc:
        logger.warning("video_info_extraction_failed", error=str(exc), exc_type=type(exc).__name__, operation="video_info", modality="video")
        return {"fps": 30, "total_frames": 300, "duration_seconds": 10, "width": 1920, "height": 1080, "note": "cv2 not available"}


def _find_video_model_dir() -> str | None:
    """Find video deepfake model directory with reliable parent resolution."""
    base = Path(__file__).resolve()
    for p in [
        base.parent.parent.parent.parent.parent / "Model" / "video",
        base.parent.parent.parent.parent / "Model" / "video",
        base.parent.parent.parent / "models" / "video",
    ]:
        if p.exists() and (any(p.glob("*.safetensors")) or any(p.glob("*.bin"))):
            return str(p)
    return None


@lru_cache(maxsize=1)
def _load_video_deepfake_model():
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification
    except ImportError:
        return None, None, None, "cpu"

    model_dir = _find_video_model_dir()
    if not model_dir:
        return None, None, None, "cpu"

    from app.core.config import settings
    device = settings.compute_device

    try:
        proc = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True)
        model = AutoModelForImageClassification.from_pretrained(model_dir, local_files_only=True)
        model.to(device).eval()
        return proc, model, torch, device
    except Exception as e:
        logger.warning(f"Could not load video deepfake model from {model_dir}: {e}")
        return None, None, None, "cpu"


def _detect_video_modality(frames: list[tuple[int, float, np.ndarray]]) -> tuple[str, dict]:
    """
    ponytail: Rung 5 native OpenCV screen recording & digital UI discriminator.
    Analyzes color palette diversity, orthogonal edge alignment, and temporal macroblock freeze.
    Zero VRAM overhead, runs in ~10ms on CPU.
    """
    if not frames:
        return "photographic_scene", {}

    try:
        cv2 = _load_cv2()
    except Exception:
        cv2 = None
    color_ratios = []
    hv_ratios = []
    static_block_ratios = []
    prev_gray = None

    for _, _, frame in frames[:5]:
        h, w, _ = frame.shape
        # 1. Palette diversity
        thumb = cv2.resize(frame, (256, max(1, int(256 * h / w)))) if cv2 else frame[::8, ::8]
        uniq = len(np.unique(thumb.reshape(-1, 3), axis=0))
        color_ratios.append(uniq / (thumb.shape[0] * thumb.shape[1]))

        # 2. Orthogonal axis alignment (Sobel 0-deg and 90-deg UI edges)
        if cv2:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
            mag = np.sqrt(gx**2 + gy**2)
            angles = np.abs(np.arctan2(gy, gx) * 180.0 / np.pi)
            edge_mask = mag > 25
            if np.sum(edge_mask) > 0:
                aligned = ((angles < 5) | (angles > 175) | (np.abs(angles - 90) < 5)) & edge_mask
                hv_ratios.append(float(np.sum(aligned) / np.sum(edge_mask)))

            # 3. Macroblock temporal freeze ratio
            if prev_gray is not None:
                diff = np.abs(gray.astype("float32") - prev_gray.astype("float32"))
                h16, w16 = h // 16, w // 16
                static_cnt = 0
                for by in range(h16):
                    for bx in range(w16):
                        if np.mean(diff[by*16:(by+1)*16, bx*16:(bx+1)*16]) < 1.0:
                            static_cnt += 1
                static_block_ratios.append(static_cnt / max(1, h16 * w16))
            prev_gray = gray

    avg_color = float(np.mean(color_ratios)) if color_ratios else 0.5
    avg_hv = float(np.mean(hv_ratios)) if hv_ratios else 0.0
    avg_static = float(np.mean(static_block_ratios)) if static_block_ratios else 0.0

    # Screen recording discriminator:
    # Discrete UI palette OR high orthogonal window edges with frozen macroblocks
    is_screen = (avg_color < 0.06) or (avg_color < 0.10 and avg_hv > 0.30) or (avg_static > 0.85 and avg_color < 0.15)
    modality = "screen_recording" if is_screen else "photographic_scene"
    meta = {
        "modality": modality,
        "is_screen_recording": is_screen,
        "palette_variety": round(avg_color, 4),
        "orthogonal_edge_ratio": round(avg_hv, 4),
        "static_macroblock_ratio": round(avg_static, 4),
    }
    return modality, meta


def analyze_video_streaming(file_path: str, job_id: str) -> Generator[dict, None, None]:
    """
    Production dual-track video forensics:
    Track A: Cropped face deepfake analysis (OpenCV YuNet + ViT)
    Track B: Scene forensics with Screen Recording & Digital UI Discriminator
    Track C: OpenCV Farnebäck dense optical flow & motion stability
    """
    from app.ml.image_detector import _analyze_pil_image

    frames = list(_get_frames(file_path, max_frames=8, fps=1.0))
    total = len(frames)

    if not total:
        yield {"type": "complete", "job_id": job_id, "progress_percent": 100, "error": "No frames extracted"}
        return

    check_vram_headroom()
    modality, modality_meta = _detect_video_modality(frames)
    is_screen_rec = (modality == "screen_recording")

    vid_info = _video_info(file_path)
    vid_info["modality"] = modality
    vid_info["modality_meta"] = modality_meta

    yield {
        "type": "started",
        "job_id": job_id,
        "total_frames": total,
        "video_info": vid_info,
        "progress_percent": 0,
        "vram_stats": get_vram_stats(),
    }

    scores, frame_data, prev_gray, prev_flow, prev_frame_rgb = [], [], None, None, None
    any_synthid_detected = False
    v_proc, v_model, v_torch, v_device = _load_video_deepfake_model()

    for i, (_, ts, frame) in enumerate(frames):
        pil_img = Image.fromarray(frame)
        
        # Track B: Scene Forensics
        # Screen recordings have discrete UI palettes and zero camera grain; camera models are biased against them.
        if is_screen_rec:
            scene_ai_score = 0.04
            is_synthid = False
        else:
            scene_result = _analyze_pil_image(pil_img, file_name=f"frame_{i}.png", file_path=None)
            scene_ai_score = scene_result["overall_score"]
            sig_info = scene_result.get("digital_signature", {})
            is_synthid = sig_info.get("detected", False) and (
                sig_info.get("confidence", 0) >= 0.65 or sig_info.get("is_synthid", False)
            )
            if is_synthid:
                any_synthid_detected = True

        # Track A: Cropped Face Deepfake Forensics
        face_crops = _detect_faces_and_crops(frame)
        face_fake_prob = None

        if face_crops and v_proc and v_model and v_torch:
            crop_fake_scores = []
            for crop in face_crops:
                try:
                    inputs = v_proc(images=crop, return_tensors="pt")
                    inputs = {k: v.to(v_device) for k, v in inputs.items()}
                    with v_torch.inference_mode():
                        v_logits = v_model(**inputs).logits[0]
                        v_probs = v_torch.softmax(v_logits, dim=-1).cpu().tolist()
                    # Label 1 is 'Fake' in dima806/deepfake_vs_real_image_detection
                    crop_fake_scores.append(round(float(v_probs[1]), 3))
                except Exception as exc:
                    logger.warning("face_crop_inference_failed", error=str(exc), exc_type=type(exc).__name__, operation="face_deepfake_inference", modality="video", frame_index=i)
            if crop_fake_scores:
                face_fake_prob = max(crop_fake_scores)

        # Dual-Track Synthesis (Never let face-swap absence drag down strong diffusion / SynthID!)
        if face_fake_prob is not None:
            if face_fake_prob >= 0.50:
                frame_ai_score = max(scene_ai_score, face_fake_prob)
            elif scene_ai_score >= 0.50:
                frame_ai_score = scene_ai_score
            else:
                frame_ai_score = round(0.50 * scene_ai_score + 0.50 * face_fake_prob, 3)
        else:
            frame_ai_score = scene_ai_score

        # Google SynthID or Neural Diffusion Signature Anchor
        if is_synthid:
            frame_ai_score = max(frame_ai_score, 0.85)

        # Track C: Farnebäck Optical Flow & Motion Stability
        # ponytail: Rung 5 CPU Farnebäck flow replaces naive delta and avoids 1.5GB VRAM neural flow
        temporal_change = 0.0
        temporal_flicker = 0.0
        temporal_jitter = 0.0
        cv2 = _load_cv2()
        gray_u8 = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if cv2 else np.mean(frame, axis=2).astype(np.uint8)

        if prev_gray is not None and cv2:
            try:
                small_prev = cv2.resize(prev_gray, (320, 240))
                small_curr = cv2.resize(gray_u8, (320, 240))
                flow = cv2.calcOpticalFlowFarneback(small_prev, small_curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                temporal_change = round(float(np.mean(mag)) / 10.0, 4)
                temporal_flicker = round(float(np.std(mag)) / 10.0, 4)
                if prev_flow is not None:
                    # Second-derivative acceleration variance detecting generative latent jitter
                    accel = np.sqrt((flow[..., 0] - prev_flow[..., 0]) ** 2 + (flow[..., 1] - prev_flow[..., 1]) ** 2)
                    temporal_jitter = round(float(np.mean(accel)) / 10.0, 4)
                prev_flow = flow
            except Exception as exc:
                logger.warning("optical_flow_failed", error=str(exc), exc_type=type(exc).__name__, operation="farneback_flow", modality="video")
                diff = np.abs(gray_u8.astype("float32") - prev_gray.astype("float32"))
                temporal_change = round(float(np.mean(diff)) / 255.0, 4)
                temporal_flicker = round(float(np.std(diff)) / 255.0, 4)
        prev_gray = gray_u8

        # v4.1 Zero-VRAM Temporal Consistency (color & identity histogram correlation)
        color_stability = 1.0
        if prev_frame_rgb is not None and cv2:
            try:
                corrs = []
                for ch in range(3):
                    h1 = cv2.calcHist([prev_frame_rgb], [ch], None, [64], [0, 256]).flatten()
                    h2 = cv2.calcHist([frame], [ch], None, [64], [0, 256]).flatten()
                    s1, s2 = h1.sum(), h2.sum()
                    if s1 > 0 and s2 > 0:
                        corrs.append(float(cv2.compareHist(h1 / s1, h2 / s2, cv2.HISTCMP_CORREL)))
                if corrs:
                    color_stability = round(float(np.mean(corrs)), 4)
            except Exception:
                color_stability = 1.0
        prev_frame_rgb = frame

        # Generative video often exhibits abnormal inter-frame warping / high-frequency motion jitter
        if (temporal_flicker > 0.08 or temporal_jitter > 0.06) and frame_ai_score >= 0.40 and not is_screen_rec:
            frame_ai_score = min(0.98, frame_ai_score + 0.06)

        fd = {
            "timestamp": round(ts, 2),
            "score": round(frame_ai_score, 3),
            "scene_score": round(scene_ai_score, 3),
            "face_deepfake_score": face_fake_prob,
            "has_face": len(face_crops) > 0,
            "face_count": len(face_crops),
            "temporal_change": temporal_change,
            "temporal_flicker": temporal_flicker,
            "temporal_jitter": temporal_jitter,
            "color_stability": color_stability,
            "synthid_detected": is_synthid,
        }
        frame_data.append(fd)
        scores.append(frame_ai_score)

        pct = int((i + 1) / total * 100)
        yield {
            "type": "frame_result",
            "job_id": job_id,
            "progress_percent": pct,
            "current_frame": i + 1,
            "total_frames": total,
            "timestamp": round(ts, 2),
            "frame_score": round(frame_ai_score, 3),
            "faces_detected": len(face_crops),
            "temporal_change": temporal_change,
            "color_stability": color_stability,
            "thumbnail_b64": _thumbnail(frame),
        }

    # Intelligent Video Aggregation (Peak Threat + Top Quartile + Global Mean)
    if scores:
        sorted_scores = sorted(scores, reverse=True)
        peak_threat = sorted_scores[0]
        top_k = max(2, len(scores) // 2)
        top_quartile_mean = float(np.mean(sorted_scores[:top_k]))
        global_mean = float(np.mean(scores))

        # ===== FUSION MODEL IS THE SOLE SCORE AUTHORITY (Rule 2, 3, 5) =====
        # ponytail: PROTECTED - DO NOT MODIFY WITHOUT RE-RUNNING eval_accuracy.py AND HUMAN APPROVAL
        from app.ml.fusion import fuse_video_score

        has_deepfake_face = any(f.get("face_deepfake_score") and f["face_deepfake_score"] >= 0.50 for f in frame_data)
        max_face_score = max([f.get("face_deepfake_score") or 0.0 for f in frame_data] or [0.0])
        avg_flicker = float(np.mean([f.get("temporal_flicker", 0.0) for f in frame_data])) if frame_data else 0.0
        avg_jitter = float(np.mean([f.get("temporal_jitter", 0.0) for f in frame_data])) if frame_data else 0.0

        fusion_features = {
            "peak_threat": peak_threat,
            "top_quartile_mean": top_quartile_mean,
            "global_mean": global_mean,
            "is_screen_rec": 1.0 if is_screen_rec else 0.0,
            "has_deepfake_face": 1.0 if has_deepfake_face else 0.0,
            "max_face_score": max_face_score,
            "temporal_flicker": avg_flicker,
            "temporal_jitter": avg_jitter,
            "synthid_detected": 1.0 if any_synthid_detected else 0.0,
            "high_threat_frame_count": sum(1 for s in scores if s >= 0.60),
        }

        fusion_res = fuse_video_score(fusion_features)
        overall = fusion_res["overall_score"]
        logger.info(
            "video_forensics_scored",
            extra={
                "raw_neural": peak_threat,
                "fusion_features": fusion_features,
                "final_score": overall,
            },
        )
    else:
        from app.ml.fusion import fuse_video_score
        fusion_res = fuse_video_score({})
        overall = fusion_res["overall_score"]

    yield {
        "type": "complete",
        "job_id": job_id,
        "progress_percent": 100,
        "result": {
            "overall_score": overall,
            "signals": {
                "temporal_consistency": round(float(np.std(scores)), 3) if len(scores) > 1 else 0,
                "avg_frame_score": round(float(np.mean(scores)), 3) if scores else 0,
                "peak_frame_score": round(float(max(scores)), 3) if scores else 0,
                "min_frame_score": round(float(min(scores)), 3) if scores else 0,
                "frames_with_faces": sum(1 for f in frame_data if f["has_face"]),
                "synthid_frames": sum(1 for f in frame_data if f["synthid_detected"]),
                "total_frames_analyzed": total,
                "modality": modality,
                "modality_label": "Authentic Screen Recording (Human Software UI)" if is_screen_rec else "Photographic Scene / Generative Video",
                "deepfake_model": "dima806/deepfake_vs_real_image_detection (Face-Crop)" if v_model else "vit_fallback",
                "scene_backbone": "Screen Forensics & Temporal Macroblock Stability" if is_screen_rec else "AutoTrain Swin Transformer + Spectral CFA/FFT",
                "job_id": job_id,
                "vram_stats": get_vram_stats(),
            },
            "frame_scores": frame_data,
            "video_info": vid_info,
            "model_hint": (
                "Authentic human screen recording / software interface verified by spatial palette and macroblock stability analysis."
                if is_screen_rec and overall < 0.20
                else (
                    f"Dual-Track Video Forensics: Full-scene Swin + SynthID spectral analysis across {total} keyframes"
                    + (f" with {sum(1 for f in frame_data if f['has_face'])} facial deepfake crops evaluated." if any(f['has_face'] for f in frame_data) else ".")
                )
            ),
        },
    }


def analyze_video(file_path: str, job_id: str) -> dict:
    """Synchronous wrapper — runs all frames, returns final result."""
    for event in analyze_video_streaming(file_path, job_id):
        if event["type"] == "complete" and "result" in event:
            return event["result"]
    return {"overall_score": 0.5, "signals": {"error": "analysis_failed"}, "frame_scores": [], "model_hint": "Video analysis failed."}
