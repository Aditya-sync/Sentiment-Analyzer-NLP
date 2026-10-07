"""
Evaluation Script for Fine-Tuned RoBERTa V2 on Held-Out TweetEval Test Split.

Evaluates the saved model checkpoint on the official TweetEval sentiment test set.
Ensures zero retraining, strict evaluation mode (torch.no_grad()), and verifies
that model weights on disk remain completely unmodified.
"""

import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from ml_transformer.src.config import (
    DATASET_CONFIG,
    DATASET_NAME,
    MAX_LENGTH,
    SAVED_MODEL_DIR,
)


def evaluate_test_set(checkpoint_dir: Path = SAVED_MODEL_DIR) -> Dict[str, Any]:
    """
    Run full test evaluation on the official held-out test split of TweetEval sentiment.
    """
    checkpoint_path = Path(checkpoint_dir).resolve()
    print(f"[*] Checkpoint path: {checkpoint_path}")

    weights_file = checkpoint_path / "model.safetensors"
    config_file = checkpoint_path / "config.json"

    assert checkpoint_path.exists(), f"Checkpoint directory not found: {checkpoint_path}"
    assert weights_file.exists(), f"Weights file not found at: {weights_file}"
    assert config_file.exists(), f"Config file not found at: {config_file}"

    # Record initial modification time to verify weights are never altered
    mtime_before = weights_file.stat().st_mtime

    print(f"[*] Loading tokenizer and fine-tuned RoBERTa model from: {checkpoint_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint_path))
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"[*] Inference Device: {device}")

    print(f"[*] Loading official held-out test split of '{DATASET_NAME}' ('{DATASET_CONFIG}')...")
    raw_test_dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="test")
    num_test_samples = len(raw_test_dataset)
    print(f"[+] Loaded {num_test_samples} test samples.")

    # Tokenize test split
    print("[*] Tokenizing test split with dynamic padding preparation...")
    def tokenize_func(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    tokenized_dataset = raw_test_dataset.map(tokenize_func, batched=True, remove_columns=["text"])
    tokenized_dataset.set_format("torch")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    data_loader = DataLoader(tokenized_dataset, batch_size=64, shuffle=False, collate_fn=data_collator)

    all_preds = []
    all_labels = []

    print("[*] Running inference on held-out test set (torch.no_grad())...")
    t0 = time.time()
    total_batches = len(data_loader)

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            inputs = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask"]}
            labels = batch["labels"].cpu().numpy()
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

            if (i + 1) % 40 == 0 or (i + 1) == total_batches:
                elapsed = time.time() - t0
                pct = ((i + 1) / total_batches) * 100
                print(f"    Batch {i+1}/{total_batches} ({pct:.1f}%) | Elapsed: {elapsed:.1f}s")

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute classification metrics
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    # Order of classes: 0 = NEGATIVE, 1 = NEUTRAL, 2 = POSITIVE
    prec_per_class = precision_score(all_labels, all_preds, average=None, labels=[0, 1, 2], zero_division=0)
    rec_per_class = recall_score(all_labels, all_preds, average=None, labels=[0, 1, 2], zero_division=0)
    f1_per_class = f1_score(all_labels, all_preds, average=None, labels=[0, 1, 2], zero_division=0)

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])

    # Verify model weights were untouched
    mtime_after = weights_file.stat().st_mtime
    assert mtime_before == mtime_after, "Verification FAILED: Model weights file modification time changed!"

    # Clean formatted report
    print("\n" + "=" * 60)
    print("FINAL RoBERTa V2 TEST SET EVALUATION")
    print("=" * 60)
    print(f"\nTest Samples: {num_test_samples}\n")
    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"Macro F1: {macro_f1 * 100:.2f}%\n")
    print(f"{'':20}{'Precision':>12}{'Recall':>12}{'F1':>10}")
    print(f"{'NEGATIVE':20}{prec_per_class[0]*100:>11.2f}%{rec_per_class[0]*100:>11.2f}%{f1_per_class[0]*100:>9.2f}%")
    print(f"{'NEUTRAL':20}{prec_per_class[1]*100:>11.2f}%{rec_per_class[1]*100:>11.2f}%{f1_per_class[1]*100:>9.2f}%")
    print(f"{'POSITIVE':20}{prec_per_class[2]*100:>11.2f}%{rec_per_class[2]*100:>11.2f}%{f1_per_class[2]*100:>9.2f}%")
    print("\nConfusion Matrix:")
    print(cm)
    print("\n" + "=" * 60)
    print("Verification Details:")
    print(f"- checkpoint used: {checkpoint_path}")
    print(f"- test split used: {DATASET_NAME} ({DATASET_CONFIG}) -> 'test' split ({num_test_samples} samples)")
    print(f"- no retraining occurred: True (torch.no_grad(), eval mode)")
    print(f"- model weights were not modified: True (file mtime unchanged)")
    print("=" * 60)

    return {
        "num_test_samples": num_test_samples,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "precision": prec_per_class,
        "recall": rec_per_class,
        "f1": f1_per_class,
        "confusion_matrix": cm,
    }


def main():
    evaluate_test_set()


if __name__ == "__main__":
    main()
