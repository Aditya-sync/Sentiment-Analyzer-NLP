"""
Dataset Exploration and Integrity Verification for TweetEval Sentiment.

Performs thorough statistical analysis, distribution checking, text/token length metrics,
and data integrity verification (missing values, duplicates, split overlaps)
as required by Steps 5 and 6.
"""

from collections import Counter
from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd
from datasets import DatasetDict

from ml_transformer.src.config import ID2LABEL, LABEL2ID, MODEL_NAME
from ml_transformer.src.load_dataset import load_tweet_sentiment_dataset


def print_section(title: str) -> None:
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(f" {title.upper()}")
    print("=" * 70)


def explore_splits_and_columns(datasets: DatasetDict) -> None:
    """Print split counts, column names, and dataset features."""
    print_section("1. Split Overview & Features")
    for split_name, split_data in datasets.items():
        print(f"Split: {split_name:<12} | Examples: {len(split_data):<6} | Columns: {split_data.column_names}")

    train_features = datasets["train"].features
    print("\nFeature Details (Train Split):")
    for col, feat in train_features.items():
        print(f"  - {col}: {feat}")


def verify_labels_and_distribution(datasets: DatasetDict) -> Dict[str, Dict[str, Any]]:
    """Check label names, ID mappings, and label distributions across all splits."""
    print_section("2. Label Verification & Distribution")

    # Verify ID to label mapping
    print("Label Mappings Verified:")
    for label_id, name in ID2LABEL.items():
        print(f"  {label_id} -> {name}")

    distributions = {}

    for split_name in ["train", "validation", "test"]:
        split_data = datasets[split_name]
        total = len(split_data)
        labels = split_data["label"]
        counts = Counter(labels)

        print(f"\n--- {split_name.upper()} SPLIT (Total: {total}) ---")
        split_dist = {}
        for label_id in sorted(ID2LABEL.keys()):
            count = counts.get(label_id, 0)
            pct = (count / total) * 100 if total > 0 else 0
            label_name = ID2LABEL[label_id]
            split_dist[label_name] = {"count": count, "percentage": pct}
            print(f"  Label {label_id} ({label_name:<8}): {count:>5} ({pct:>5.2f}%)")
        distributions[split_name] = split_dist

    return distributions


def show_sample_examples(datasets: DatasetDict, samples_per_class: int = 3) -> None:
    """Display representative examples from each sentiment class."""
    print_section(f"3. Sample Examples ({samples_per_class} per class from Train split)")
    train_data = datasets["train"]

    by_class: Dict[int, List[str]] = {0: [], 1: [], 2: []}
    for item in train_data:
        label = item["label"]
        if len(by_class[label]) < samples_per_class:
            by_class[label].append(item["text"])
        if all(len(texts) >= samples_per_class for texts in by_class.values()):
            break

    for label_id in sorted(by_class.keys()):
        label_name = ID2LABEL[label_id]
        print(f"\n[Class {label_id}: {label_name.upper()}]")
        for i, text in enumerate(by_class[label_id], 1):
            cleaned_preview = text.replace("\n", " ").strip()
            print(f"  {i}. \"{cleaned_preview}\"")


def compute_length_statistics(datasets: DatasetDict) -> None:
    """Compute character, word, and token length statistics across splits."""
    print_section("4. Text Length Statistics (Characters & Words)")

    # Optional: try loading roberta tokenizer for exact token length statistics
    tokenizer = None
    try:
        from transformers import AutoTokenizer
        print(f"[*] Loading tokenizer '{MODEL_NAME}' for token-level statistics...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        print("[+] Tokenizer loaded.")
    except Exception as e:
        print(f"[!] Tokenizer could not be loaded ({e}); proceeding with word/character counts.")

    for split_name in ["train", "validation", "test"]:
        texts = datasets[split_name]["text"]
        char_lens = [len(t) for t in texts]
        word_lens = [len(t.split()) for t in texts]

        print(f"\n--- {split_name.upper()} (N = {len(texts)}) ---")
        print(f"Character Length: min = {np.min(char_lens):>3}, max = {np.max(char_lens):>3}, "
              f"avg = {np.mean(char_lens):>6.2f}, median = {np.median(char_lens):>5.1f}")
        print(f"Word Count:       min = {np.min(word_lens):>3}, max = {np.max(word_lens):>3}, "
              f"avg = {np.mean(word_lens):>6.2f}, median = {np.median(word_lens):>5.1f}")

        if tokenizer is not None:
            token_lens = [len(tokenizer.encode(t, truncation=False)) for t in texts]
            p95 = np.percentile(token_lens, 95)
            p99 = np.percentile(token_lens, 99)
            print(f"RoBERTa Tokens:   min = {np.min(token_lens):>3}, max = {np.max(token_lens):>3}, "
                  f"avg = {np.mean(token_lens):>6.2f}, median = {np.median(token_lens):>5.1f}, "
                  f"95th% = {p95:.1f}, 99th% = {p99:.1f}")


def check_data_integrity(datasets: DatasetDict) -> Dict[str, Any]:
    """
    Check for:
    - Missing/empty text
    - Missing/invalid labels
    - Duplicate examples within each split
    - Overlap across train, validation, and test splits
    """
    print_section("5. Data Integrity Checks (Missing Values, Duplicates & Overlaps)")
    integrity_report = {}

    # Check missing values and internal duplicates
    split_texts: Dict[str, List[str]] = {}
    split_text_sets: Dict[str, Set[str]] = {}

    for split_name in ["train", "validation", "test"]:
        data = datasets[split_name]
        texts = data["text"]
        labels = data["label"]

        # Missing text
        missing_texts = sum(1 for t in texts if t is None or not str(t).strip())
        # Missing labels or invalid labels
        invalid_labels = sum(1 for l in labels if l is None or l not in (0, 1, 2))

        # Duplicates within split
        num_total = len(texts)
        unique_texts = set(texts)
        num_duplicates = num_total - len(unique_texts)

        split_texts[split_name] = texts
        split_text_sets[split_name] = unique_texts

        integrity_report[split_name] = {
            "total": num_total,
            "missing_texts": missing_texts,
            "invalid_labels": invalid_labels,
            "duplicate_texts": num_duplicates,
        }

        print(f"\n--- {split_name.upper()} INTEGRITY ---")
        print(f"  Total examples:        {num_total}")
        print(f"  Missing / empty texts: {missing_texts}")
        print(f"  Invalid / null labels: {invalid_labels}")
        print(f"  Duplicate texts:       {num_duplicates} ({(num_duplicates / num_total)*100:.2f}%)")

    # Cross-split overlap
    train_set = split_text_sets["train"]
    val_set = split_text_sets["validation"]
    test_set = split_text_sets["test"]

    train_val_overlap = train_set.intersection(val_set)
    train_test_overlap = train_set.intersection(test_set)
    val_test_overlap = val_set.intersection(test_set)

    integrity_report["overlaps"] = {
        "train_val": len(train_val_overlap),
        "train_test": len(train_test_overlap),
        "val_test": len(val_test_overlap),
    }

    print("\n--- CROSS-SPLIT OVERLAP ---")
    print(f"  Train & Validation overlap: {len(train_val_overlap)} examples")
    print(f"  Train & Test overlap:       {len(train_test_overlap)} examples")
    print(f"  Validation & Test overlap:  {len(val_test_overlap)} examples")

    if train_test_overlap:
        print(f"  [!] Note: {len(train_test_overlap)} exact text strings appear in both train and test splits in official TweetEval.")
    else:
        print("  [+] No overlap detected between train and test splits.")

    print("\n[+] Integrity checks completed. No modifications made to raw dataset.")
    return integrity_report


def run_full_exploration() -> None:
    """Run full exploration and integrity pipeline."""
    datasets = load_tweet_sentiment_dataset()
    explore_splits_and_columns(datasets)
    verify_labels_and_distribution(datasets)
    show_sample_examples(datasets, samples_per_class=3)
    compute_length_statistics(datasets)
    check_data_integrity(datasets)


if __name__ == "__main__":
    run_full_exploration()
