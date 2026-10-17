"""
Fine-tune RoBERTa-base end-to-end on the RAID training dataset (multi-generator AI + multi-register human text).
Strict VRAM budget (<3.5 GB on RTX 4050 6GB) using PyTorch AMP FP16 and gradient accumulation.
Evaluates on sealed raid_val/text/ after each epoch and exports best checkpoint.
"""

import argparse
import json
import logging
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

root = Path(__file__).resolve().parent.parent
backend_path = root / "slop" / "backend"
sys.path.insert(0, str(backend_path))

from app.ml.text_detector import _normalize_and_sanitize_text  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TextDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], ids: list[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.ids = ids
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index: int):
        clean_text = _normalize_and_sanitize_text(self.texts[index])
        encoding = self.tokenizer(
            clean_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "id": self.ids[idx],
        }


def load_dataset_from_dir(base_dir: Path):
    texts, labels, ids = [], [], []
    for label_dir, y in (("ai", 1), ("human", 0)):
        d = base_dir / label_dir
        if not d.exists():
            continue
        for sub_dir in sorted(p for p in d.iterdir() if p.is_dir()):
            for f in sorted(sub_dir.glob("*.txt")):
                sid = f"{label_dir}/{sub_dir.name}/{f.name}"
                try:
                    content = f.read_text(encoding="utf-8")
                    if len(content.strip()) > 10:
                        texts.append(content)
                        labels.append(y)
                        ids.append(sid)
                except Exception as e:
                    logger.warning(f"Failed to read {f}: {e}")
    return texts, labels, ids


def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds, all_probs, all_labels, all_ids = [], [], [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            with torch.amp.autocast("cuda"):  # type: ignore[bad-context-manager]
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = criterion(logits, labels)

            total_loss += loss.item() * len(labels)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_ids.extend(batch["id"])

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    tp = float(np.sum((all_preds == 1) & (all_labels == 1)))
    tn = float(np.sum((all_preds == 0) & (all_labels == 0)))
    fp = float(np.sum((all_preds == 1) & (all_labels == 0)))
    fn = float(np.sum((all_preds == 0) & (all_labels == 1)))

    rec1 = tp / (tp + fn) if (tp + fn) else 0.0
    rec0 = tn / (tn + fp) if (tn + fp) else 0.0
    prec1 = tp / (tp + fp) if (tp + fp) else 0.0
    prec0 = tn / (tn + fn) if (tn + fn) else 0.0

    bacc = 0.5 * (rec1 + rec0)
    acc = (tp + tn) / len(all_labels) if len(all_labels) else 0.0
    f1_ai = 2 * prec1 * rec1 / (prec1 + rec1) if (prec1 + rec1) else 0.0
    f1_hu = 2 * prec0 * rec0 / (prec0 + rec0) if (prec0 + rec0) else 0.0
    macro_f1 = 0.5 * (f1_ai + f1_hu)

    per_gen = {}
    for sid, p, y in zip(all_ids, all_preds.tolist(), all_labels.tolist()):
        parts = sid.split("/")
        gen_key = f"{parts[0]}/{parts[1]}"
        g = per_gen.setdefault(gen_key, {"tp": 0, "fp": 0, "tn": 0, "fn": 0})
        if y == 1:
            if p == 1:
                g["tp"] += 1
            else:
                g["fn"] += 1
        else:
            if p == 0:
                g["tn"] += 1
            else:
                g["fp"] += 1

    return {
        "val_loss": total_loss / len(all_labels) if len(all_labels) else 0.0,
        "balanced_accuracy": round(float(bacc), 4),
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "ai_recall": round(float(rec1), 4),
        "human_recall": round(float(rec0), 4),
        "per_gen": per_gen,
        "predictions": dict(zip(all_ids, all_probs)),
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune text classifier on RAID dataset")
    parser.add_argument("--model-path", type=str, default="Model/text", help="Base model path or HF repo ID")
    parser.add_argument("--output-dir", type=str, default="Model/text/fine_tuned", help="Checkpoint output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    train_dir = root / "raid_train" / "text"
    val_dir = root / "raid_val" / "text"
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading base tokenizer and model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=2)
    model.to(device)

    logger.info("Loading training and validation datasets...")
    train_texts, train_labels, train_ids = load_dataset_from_dir(train_dir)
    val_texts, val_labels, val_ids = load_dataset_from_dir(val_dir)
    logger.info(f"Loaded {len(train_texts)} train samples ({sum(train_labels)} AI, {len(train_labels) - sum(train_labels)} Human)")
    logger.info(f"Loaded {len(val_texts)} val samples ({sum(val_labels)} AI, {len(val_labels) - sum(val_labels)} Human)")

    train_dataset = TextDataset(train_texts, train_labels, train_ids, tokenizer, max_length=args.max_length)
    val_dataset = TextDataset(val_texts, val_labels, val_ids, tokenizer, max_length=args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Class-balanced loss weights
    n_total = len(train_labels)
    n_ai = sum(train_labels)
    n_hu = n_total - n_ai
    w_hu = n_total / (2.0 * n_hu)
    w_ai = n_total / (2.0 * n_ai)
    class_weights = torch.tensor([w_hu, w_ai], dtype=torch.float, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    logger.info(f"Class weights: Human={w_hu:.3f}, AI={w_ai:.3f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    scaler = torch.amp.GradScaler("cuda")

    best_val_bacc = 0.0
    best_metrics = None

    logger.info("Starting fine-tuning...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            with torch.amp.autocast("cuda"):  # type: ignore[bad-context-manager]
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels) / args.grad_accum

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * args.grad_accum * len(labels)

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        avg_train_loss = epoch_loss / len(train_texts)
        val_res = evaluate_model(model, val_loader, device)

        logger.info(
            f"Epoch {epoch}/{args.epochs}: Train Loss={avg_train_loss:.4f} | "
            f"Val Loss={val_res['val_loss']:.4f} | "
            f"Val BAcc={val_res['balanced_accuracy']:.4f} | "
            f"AI Rec={val_res['ai_recall']:.4f} | "
            f"Hu Rec={val_res['human_recall']:.4f} | "
            f"Macro F1={val_res['macro_f1']:.4f}"
        )

        if val_res["balanced_accuracy"] > best_val_bacc:
            best_val_bacc = val_res["balanced_accuracy"]
            best_metrics = val_res
            logger.info(f"==> New best validation balanced accuracy: {best_val_bacc:.4f}. Saving checkpoint...")
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

    logger.info("Fine-tuning completed. Generating full predictions for train and val splits...")
    # Load best checkpoint
    model = AutoModelForSequenceClassification.from_pretrained(output_dir)
    model.to(device)

    # Generate predictions on train split (for GBDT training)
    train_eval_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    train_res = evaluate_model(model, train_eval_loader, device)
    (train_dir / "_transformer_preds.json").write_text(json.dumps(train_res["predictions"], indent=2))
    logger.info(f"Wrote train predictions to {train_dir / '_transformer_preds.json'}")

    # Generate predictions on val split
    val_res = evaluate_model(model, val_loader, device)
    (val_dir / "_transformer_preds.json").write_text(json.dumps(val_res["predictions"], indent=2))
    logger.info(f"Wrote val predictions to {val_dir / '_transformer_preds.json'}")

    summary_file = output_dir / "eval_summary.json"
    summary_file.write_text(
        json.dumps(
            {
                "best_val_balanced_accuracy": best_val_bacc,
                "val_metrics": {k: v for k, v in val_res.items() if k != "predictions"},
                "train_metrics": {k: v for k, v in train_res.items() if k != "predictions"},
            },
            indent=2,
        )
    )
    logger.info(f"Saved evaluation summary to {summary_file}")

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
