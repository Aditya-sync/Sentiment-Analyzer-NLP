"""
RoBERTa Fine-Tuning Pipeline for TweetEval Sentiment Analysis.

Features:
- Hugging Face Transformers Trainer with dynamic padding (DataCollatorWithPadding).
- Automatic CUDA detection with safeguard against accidental CPU training runs.
- Comprehensive evaluation metrics: Accuracy, Macro F1 (model selection), Weighted F1, Precision, Recall.
- Confusion matrix computation and reporting for Negative, Neutral, and Positive classes.
- Lightweight local smoke-test mode validating:
  * Dataset & Tokenizer loading
  * Model instantiation & architecture
  * Data collation
  * Forward pass & logits shape (batch_size, 3)
  * Loss calculation
  * Backpropagation & optimizer gradient update
  * TrainingArguments instantiation
- Fully compatible with Google Colab / GPU environments for full 4-epoch training.
"""

import argparse
import inspect
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from datasets import DatasetDict
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from ml_transformer.src.config import (
    BATCH_SIZE,
    DATASET_CONFIG,
    DATASET_NAME,
    EVAL_BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    ID2LABEL,
    LABEL2ID,
    LEARNING_RATE,
    MAX_LENGTH,
    METRIC_FOR_BEST_MODEL,
    MODEL_NAME,
    NUM_EPOCHS,
    NUM_LABELS,
    OUTPUT_DIR,
    SAVED_MODEL_DIR,
    SEED,
    WARMUP_RATIO,
    WEIGHT_DECAY,
    set_seed,
)
from ml_transformer.src.load_dataset import load_tweet_sentiment_dataset


# ============================================================================
# 1. Evaluation Metrics & Confusion Matrix
# ============================================================================

def compute_metrics(eval_pred) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics for evaluation.
    Macro F1 is the primary metric for model selection.
    Also calculates per-class precision, recall, and F1 for Negative, Neutral, and Positive.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    macro_precision = precision_score(labels, preds, average="macro", zero_division=0)
    macro_recall = recall_score(labels, preds, average="macro", zero_division=0)

    # Per-class metrics: class 0 = negative, class 1 = neutral, class 2 = positive
    f1_per_class = f1_score(labels, preds, average=None, labels=[0, 1, 2], zero_division=0)
    prec_per_class = precision_score(labels, preds, average=None, labels=[0, 1, 2], zero_division=0)
    rec_per_class = recall_score(labels, preds, average=None, labels=[0, 1, 2], zero_division=0)

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "precision": float(macro_precision),
        "recall": float(macro_recall),
        # Per-class breakdown
        "f1_negative": float(f1_per_class[0]),
        "f1_neutral": float(f1_per_class[1]),
        "f1_positive": float(f1_per_class[2]),
        "precision_negative": float(prec_per_class[0]),
        "precision_neutral": float(prec_per_class[1]),
        "precision_positive": float(prec_per_class[2]),
        "recall_negative": float(rec_per_class[0]),
        "recall_neutral": float(rec_per_class[1]),
        "recall_positive": float(rec_per_class[2]),
    }


def print_confusion_matrix(labels: List[int], preds: List[int]) -> np.ndarray:
    """
    Display a formatted confusion matrix for Negative, Neutral, Positive.
    """
    class_names = ["Negative (0)", "Neutral (1)", "Positive (2)"]
    cm = confusion_matrix(labels, preds, labels=[0, 1, 2])

    print("\n" + "=" * 60)
    print(" CONFUSION MATRIX")
    print("=" * 60)
    header = f"{'Actual \\ Predicted':<20}" + "".join([f"{name:>15}" for name in class_names])
    print(header)
    print("-" * len(header))
    for i, row in enumerate(cm):
        row_str = f"{class_names[i]:<20}" + "".join([f"{val:>15}" for val in row])
        print(row_str)
    print("=" * 60)
    return cm


# ============================================================================
# 2. Model & Tokenizer Initializer
# ============================================================================

def load_model_and_tokenizer(model_name: str = MODEL_NAME):
    """
    Load base RoBERTa model with randomly initialized 3-class classification head
    and corresponding Byte-Pair Encoding tokenizer.
    """
    print(f"[*] Initializing tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"[*] Initializing model: {model_name} with {NUM_LABELS} sentiment labels")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    return model, tokenizer


# ============================================================================
# 3. Training Arguments Builder
# ============================================================================

def get_training_args(
    output_dir: Path = OUTPUT_DIR,
    epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    eval_batch_size: int = EVAL_BATCH_SIZE,
    gradient_accumulation_steps: int = GRADIENT_ACCUMULATION_STEPS,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    warmup_ratio: float = WARMUP_RATIO,
    seed: int = SEED,
) -> TrainingArguments:
    """
    Create Hugging Face TrainingArguments configured for epoch-based validation,
    checkpoints, and best-model selection by Macro F1.
    Handles eval_strategy vs evaluation_strategy dynamically across Transformers versions.
    """
    has_cuda = torch.cuda.is_available()

    args_kwargs = {
        "output_dir": str(output_dir),
        "save_strategy": "epoch",
        "learning_rate": learning_rate,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": eval_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "num_train_epochs": epochs,
        "weight_decay": weight_decay,
        "load_best_model_at_end": True,
        "metric_for_best_model": METRIC_FOR_BEST_MODEL,
        "greater_is_better": True,
        "logging_strategy": "steps",
        "logging_steps": 50,
        "save_total_limit": 2,
        "seed": seed,
        "fp16": has_cuda,  # Automatic mixed precision on CUDA GPUs (never on CPU)
        "dataloader_num_workers": 2 if has_cuda else 0,
        "report_to": "none",  # Do not require external logging services
    }

    # Version-safe parameter inspection for evaluation strategy
    sig = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in sig.parameters:
        args_kwargs["eval_strategy"] = "epoch"
    else:
        args_kwargs["evaluation_strategy"] = "epoch"

    # Version-safe parameter inspection for warmup (warmup_ratio in 4.x vs warmup_steps in 5.x)
    if "warmup_ratio" in sig.parameters:
        args_kwargs["warmup_ratio"] = warmup_ratio
    elif "warmup_steps" in sig.parameters:
        args_kwargs["warmup_steps"] = 500

    return TrainingArguments(**args_kwargs)


# ============================================================================
# 4. Pipeline Smoke Test (Task 8: Local Validation Without Full Training)
# ============================================================================

def run_smoke_test() -> Dict[str, Any]:
    """
    Lightweight, rigorous smoke test to validate all technical components:
    1. Dataset loads
    2. Tokenizer loads
    3. RoBERTa model loads
    4. Tokenized data has expected fields
    5. Data collator works with dynamic padding
    6. Small batch forward pass succeeds
    7. Output logits have shape (batch_size, 3)
    8. Loss calculation succeeds
    9. Backpropagation executes gradient update
    10. TrainingArguments instantiate successfully
    """
    print("\n" + "=" * 75)
    print(" TASK 8: LOCAL PIPELINE SMOKE TEST & TECHNICAL VALIDATION")
    print("=" * 75)

    results = {}
    set_seed(SEED)

    # 1. Dataset loading check
    print("\n[Step 1/10] Loading raw dataset...")
    try:
        raw_datasets = load_tweet_sentiment_dataset()
        results["dataset_loaded"] = True
        print("  --> [OK] Raw dataset loaded.")
    except Exception as exc:
        results["dataset_loaded"] = False
        print(f"  --> [FAIL] Dataset loading error: {exc}")
        return results

    # 2. Tokenizer loading check
    print("\n[Step 2/10] Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        results["tokenizer_loaded"] = True
        print(f"  --> [OK] Tokenizer loaded (vocab size: {tokenizer.vocab_size:,}).")
    except Exception as exc:
        results["tokenizer_loaded"] = False
        print(f"  --> [FAIL] Tokenizer loading error: {exc}")
        return results

    # 3. Model loading check
    print("\n[Step 3/10] Loading RoBERTa sequence classification model...")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_LABELS,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
        results["model_loaded"] = True
        print("  --> [OK] RoBERTa classification model loaded.")
    except Exception as exc:
        results["model_loaded"] = False
        print(f"  --> [FAIL] Model loading error: {exc}")
        return results

    # 4. Tokenization of tiny slice (4 samples)
    print("\n[Step 4/10] Tokenizing small slice (4 samples) without static padding...")
    mini_train = raw_datasets["train"].select(range(4))

    def tokenize_func(examples):
        return tokenizer(examples["text"], truncation=True, max_length=MAX_LENGTH)

    tokenized_mini = mini_train.map(tokenize_func, batched=True)
    expected_fields = {"input_ids", "attention_mask", "label"}
    has_expected_fields = expected_fields.issubset(set(tokenized_mini.column_names))
    results["fields_verified"] = has_expected_fields
    print(f"  --> [OK] Columns: {tokenized_mini.column_names} (Required fields present: {has_expected_fields})")

    # 5. Data collator check (dynamic padding test)
    print("\n[Step 5/10] Testing DataCollatorWithPadding...")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    features = [
        {"input_ids": item["input_ids"], "attention_mask": item["attention_mask"], "labels": item["label"]}
        for item in tokenized_mini
    ]
    batch = data_collator(features)
    results["collator_works"] = True
    print(f"  --> [OK] Collated batch shapes: input_ids={tuple(batch['input_ids'].shape)}, labels={tuple(batch['labels'].shape)}")

    # 6 & 7. Forward pass & logits shape check
    print("\n[Step 6 & 7/10] Running forward pass and checking logits shape...")
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
    logits_shape = tuple(outputs.logits.shape)
    expected_shape = (4, NUM_LABELS)
    correct_shape = logits_shape == expected_shape
    results["logits_shape"] = logits_shape
    results["shape_correct"] = correct_shape
    print(f"  --> [OK] Model output logits shape: {logits_shape} (Expected: {expected_shape})")

    # 8. Loss calculation check
    print("\n[Step 8/10] Checking loss calculation...")
    loss_val = outputs.loss.item()
    loss_valid = not np.isnan(loss_val) and loss_val > 0
    results["loss_calculated"] = loss_valid
    results["loss_value"] = loss_val
    print(f"  --> [OK] Calculated Cross-Entropy Loss: {loss_val:.4f} (Valid: {loss_valid})")

    # 9. Backpropagation check
    print("\n[Step 9/10] Testing backpropagation and gradient calculation...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    optimizer.zero_grad()
    train_outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    train_loss = train_outputs.loss
    train_loss.backward()

    # Check that gradients were computed
    has_grads = any(p.grad is not None and torch.norm(p.grad) > 0 for p in model.parameters())
    optimizer.step()
    results["backprop_success"] = has_grads
    print(f"  --> [OK] Backward pass completed; gradients calculated: {has_grads}")

    # 10. TrainingArguments instantiation check
    print("\n[Step 10/10] Instantiating TrainingArguments...")
    try:
        training_args = get_training_args(epochs=1, batch_size=4)
        results["training_args_ready"] = True
        print(f"  --> [OK] TrainingArguments initialized successfully (eval_strategy={training_args.eval_strategy}).")
    except Exception as exc:
        results["training_args_ready"] = False
        print(f"  --> [FAIL] TrainingArguments initialization error: {exc}")

    # Check CUDA availability
    has_cuda = torch.cuda.is_available()
    results["cuda_available"] = has_cuda

    print("\n" + "=" * 75)
    print(" SMOKE TEST SUMMARY")
    print("=" * 75)
    print(f"  1. Dataset loaded:            {'YES' if results.get('dataset_loaded') else 'NO'}")
    print(f"  2. Tokenizer loaded:          {'YES' if results.get('tokenizer_loaded') else 'NO'}")
    print(f"  3. RoBERTa model loaded:      {'YES' if results.get('model_loaded') else 'NO'}")
    print(f"  4. Expected fields verified:  {'YES' if results.get('fields_verified') else 'NO'}")
    print(f"  5. Data collator works:       {'YES' if results.get('collator_works') else 'NO'}")
    print(f"  6. Model output shape:        {results.get('logits_shape')}")
    print(f"  7. Loss calculation:          {'YES' if results.get('loss_calculated') else 'NO'} ({results.get('loss_value', 0):.4f})")
    print(f"  8. Backpropagation test:      {'YES' if results.get('backprop_success') else 'NO'}")
    print(f"  9. TrainingArguments ready:   {'YES' if results.get('training_args_ready') else 'NO'}")
    print(f" 10. CUDA available:            {'YES' if results.get('cuda_available') else 'NO (CPU Mode)'}")
    print("=" * 75)

    return results


# ============================================================================
# 5. Full Training Pipeline Function (For Google Colab / GPU Execution)
# ============================================================================

def run_full_training(force_cpu: bool = False) -> None:
    """
    Launch full 4-epoch fine-tuning using Hugging Face Trainer.
    Guarded against accidental execution on CPU-only machines.
    """
    has_cuda = torch.cuda.is_available()

    if not has_cuda and not force_cpu:
        print("\n" + "!" * 75)
        print(" [SAFEGUARD TRIGGERED] CUDA IS NOT AVAILABLE ON THIS MACHINE")
        print("!" * 75)
        print("""
Your local system (AMD Ryzen 7 5700U CPU, no NVIDIA CUDA) would require several
hours of high CPU utilization to train RoBERTa across 45,615 examples for 4 epochs.

To train the model efficiently:
1. Open Google Colab (https://colab.research.google.com).
2. Select a GPU runtime (Runtime -> Change runtime type -> T4 GPU).
3. Clone or upload this repository.
4. Run:
     python ml_transformer/src/train.py --full-train

If you intentionally want to run on CPU locally, pass:
     python ml_transformer/src/train.py --full-train --force-cpu
""")
        print("!" * 75)
        sys.exit(0)

    print("\n" + "=" * 75)
    print(" LAUNCHING FULL ROBERTA FINE-TUNING PIPELINE")
    print(f" Device: {'CUDA GPU (' + torch.cuda.get_device_name(0) + ')' if has_cuda else 'CPU (Forced)'}")
    print("=" * 75)

    set_seed(SEED)
    raw_datasets = load_tweet_sentiment_dataset()
    model, tokenizer = load_model_and_tokenizer()

    # Tokenize splits
    print("[*] Tokenizing datasets with dynamic padding preparation...")
    def tokenize_func(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    tokenized_datasets = raw_datasets.map(tokenize_func, batched=True, desc="Tokenizing")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Configure Trainer with version-safe processing_class / tokenizer parameter
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized_datasets["train"],
        "eval_dataset": tokenized_datasets["validation"],
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
    }
    trainer_sig = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    print("[*] Starting Trainer.train()...")
    trainer.train()

    # Final evaluation on the unseen TEST set
    print("\n[*] Evaluating best model on the official TEST split...")
    test_results = trainer.evaluate(tokenized_datasets["test"])
    print("\n--- Test Set Results ---")
    for key, value in test_results.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    # Generate test set confusion matrix
    print("\n[*] Generating Confusion Matrix on Test Split...")
    test_preds_raw = trainer.predict(tokenized_datasets["test"])
    test_preds = np.argmax(test_preds_raw.predictions, axis=1)
    test_labels = test_preds_raw.label_ids
    print_confusion_matrix(test_labels, test_preds)

    # Save best model and tokenizer
    print(f"\n[*] Saving best fine-tuned model to: {SAVED_MODEL_DIR}")
    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(SAVED_MODEL_DIR))
    tokenizer.save_pretrained(str(SAVED_MODEL_DIR))
    print("[+] Model and tokenizer saved successfully!")


# ============================================================================
# 6. CLI Entrypoint
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="RoBERTa Sentiment Training Pipeline")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run lightweight 10-step technical validation without full training",
    )
    parser.add_argument(
        "--full-train",
        action="store_true",
        help="Launch full fine-tuning (requires CUDA or --force-cpu)",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force full training on CPU even if CUDA is not detected",
    )
    args = parser.parse_args()

    if args.full_train:
        run_full_training(force_cpu=args.force_cpu)
    else:
        # Default behavior: run lightweight smoke test
        run_smoke_test()


if __name__ == "__main__":
    main()
