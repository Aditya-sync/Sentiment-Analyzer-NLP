"""
RoBERTa Tokenization Pipeline for TweetEval Sentiment.

Handles:
- Subword BPE tokenization using FacebookAI/roberta-base
- Educational inspection of token representations, IDs, and decoding
- Sequence length distribution and truncation analysis on TweetEval train split
- Batched tokenization without premature padding (preserving dynamic padding capability)
- Deep educational breakdown of Transformer input representations
"""

from typing import Any, Dict, List, Tuple

import numpy as np
from datasets import DatasetDict
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from ml_transformer.src.config import MAX_LENGTH, MODEL_NAME
from ml_transformer.src.load_dataset import load_tweet_sentiment_dataset


def get_tokenizer(model_name: str = MODEL_NAME) -> PreTrainedTokenizerFast:
    """
    Load the pre-trained RoBERTa tokenizer.

    Args:
        model_name: Hugging Face model identifier.

    Returns:
        PreTrainedTokenizerFast: Fast tokenizer instance.
    """
    print(f"[*] Loading tokenizer '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"[+] Tokenizer loaded successfully (vocab size: {tokenizer.vocab_size:,}).")
    return tokenizer


def inspect_sample_sentences(tokenizer: PreTrainedTokenizerFast) -> None:
    """
    Inspect and display detailed tokenization behavior on canonical test sentences.
    Demonstrates subword token splitting, special tokens (<s>, </s>), IDs, attention mask, and decoding.
    """
    sample_sentences = [
        "I absolutely loved this movie!",
        "The movie was okay, nothing special.",
        "I hated every minute of it.",
        "It arrived on time.",
        "I expected more, but the camera is usable.",
        "I do not dislike it.",
    ]

    print("\n" + "=" * 75)
    print(" TASK 2: TOKENIZATION INSPECTION ON EXAMPLE SENTENCES")
    print("=" * 75)

    for idx, sentence in enumerate(sample_sentences, 1):
        encoded = tokenizer(sentence, truncation=True, max_length=MAX_LENGTH)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        decoded_text = tokenizer.decode(input_ids, skip_special_tokens=False)
        decoded_clean = tokenizer.decode(input_ids, skip_special_tokens=True)

        print(f"\n[Example {idx}]")
        print(f"  Original Text:    \"{sentence}\"")
        print(f"  Tokens:           {tokens}")
        print(f"  Token IDs:        {input_ids}")
        print(f"  Attention Mask:   {attention_mask}")
        print(f"  Token Count:      {len(input_ids)}")
        print(f"  Decoded (with special tokens): \"{decoded_text}\"")
        print(f"  Decoded (clean text):          \"{decoded_clean}\"")


def analyze_sequence_lengths(
    dataset_split,
    tokenizer: PreTrainedTokenizerFast,
    max_length: int = MAX_LENGTH,
) -> Dict[str, Any]:
    """
    Calculate sequence length statistics on raw text using the RoBERTa tokenizer
    and determine the impact of truncation at max_length.
    """
    print("\n" + "=" * 75)
    print(" TASK 3: TOKEN LENGTH DISTRIBUTION & TRUNCATION ANALYSIS")
    print("=" * 75)
    print(f"[*] Computing exact RoBERTa token lengths for {len(dataset_split):,} examples...")

    # Tokenize without truncation to inspect raw distribution
    token_lengths = [
        len(tokenizer.encode(text, truncation=False))
        for text in dataset_split["text"]
    ]

    token_lengths_arr = np.array(token_lengths)
    total_examples = len(token_lengths_arr)

    min_len = int(np.min(token_lengths_arr))
    max_len = int(np.max(token_lengths_arr))
    mean_len = float(np.mean(token_lengths_arr))
    median_len = float(np.median(token_lengths_arr))
    p90 = float(np.percentile(token_lengths_arr, 90))
    p95 = float(np.percentile(token_lengths_arr, 95))
    p99 = float(np.percentile(token_lengths_arr, 99))

    truncated_count = int(np.sum(token_lengths_arr > max_length))
    truncated_pct = (truncated_count / total_examples) * 100

    print(f"\n--- RoBERTa Sequence Length Metrics (Train Split: {total_examples:,} tweets) ---")
    print(f"  Minimum token length:    {min_len}")
    print(f"  Maximum token length:    {max_len}")
    print(f"  Mean token length:       {mean_len:.2f}")
    print(f"  Median token length:     {median_len:.1f}")
    print(f"  90th percentile:         {p90:.1f}")
    print(f"  95th percentile:         {p95:.1f}")
    print(f"  99th percentile:         {p99:.1f}")
    print(f"\n--- Truncation Threshold Check (MAX_LENGTH = {max_length}) ---")
    print(f"  Examples exceeding {max_length} tokens: {truncated_count} ({truncated_pct:.4f}%)")
    if truncated_count == 0:
        print(f"  [+] Excellent: 100% of training samples fit comfortably within MAX_LENGTH={max_length}.")
    else:
        print(f"  [!] Note: {truncated_count} samples will be truncated at {max_length} tokens.")

    return {
        "min": min_len,
        "max": max_len,
        "mean": mean_len,
        "median": median_len,
        "p90": p90,
        "p95": p95,
        "p99": p99,
        "truncated_count": truncated_count,
        "truncated_pct": truncated_pct,
    }


def create_tokenized_dataset(
    raw_datasets: DatasetDict,
    tokenizer: PreTrainedTokenizerFast,
    max_length: int = MAX_LENGTH,
) -> DatasetDict:
    """
    Apply batched tokenization across all splits (train, validation, test).
    Truncates at max_length but does NOT statically pad to max_length.
    Preserves original columns ('text', 'label') alongside 'input_ids' and 'attention_mask'.
    """
    print("\n" + "=" * 75)
    print(" TASK 4: CREATING TOKENIZED DATASET (BATCHED)")
    print("=" * 75)

    def tokenize_batch(batch: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Tokenize a batch of text samples without static padding."""
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

    print(f"[*] Applying batched tokenization (truncation=True, max_length={max_length}, no static padding)...")
    tokenized_datasets = raw_datasets.map(
        tokenize_batch,
        batched=True,
        batch_size=1000,
        desc="Tokenizing splits",
    )
    print("[+] Batched tokenization completed for all splits.")
    return tokenized_datasets


def verify_tokenized_dataset(tokenized_datasets: DatasetDict) -> None:
    """
    Verify structure, features, column integrity, and sample content of the tokenized dataset.
    """
    print("\n" + "=" * 75)
    print(" TASK 5: VERIFYING TOKENIZED DATASET OUTPUT")
    print("=" * 75)

    for split_name, split_data in tokenized_datasets.items():
        print(f"\nSplit: '{split_name}' (Total examples: {len(split_data):,})")
        print(f"  Available Columns: {split_data.column_names}")
        print(f"  Features:          {split_data.features}")

        # Check required fields
        required_fields = ["input_ids", "attention_mask", "label"]
        missing = [f for f in required_fields if f not in split_data.column_names]
        if missing:
            raise ValueError(f"Split '{split_name}' is missing required fields: {missing}")

    # Inspect first train sample
    sample_0 = tokenized_datasets["train"][0]
    print("\n--- Detailed Inspection of Train Sample [0] ---")
    print(f"  text:           \"{sample_0['text']}\"")
    print(f"  label:          {sample_0['label']}")
    print(f"  input_ids:      {sample_0['input_ids']}")
    print(f"  attention_mask: {sample_0['attention_mask']}")
    print(f"  Sequence Length (unpadded): {len(sample_0['input_ids'])} tokens")
    print(f"  Confirmed: All required transformer inputs present.")


def print_educational_summary() -> None:
    """
    Detailed conceptual breakdown of Transformer representations, token IDs,
    attention masking, truncation, and dynamic padding.
    """
    print("\n" + "=" * 75)
    print(" TASK 6: TECHNICAL & ARCHITECTURAL FOUNDATIONS (EDUCATIONAL)")
    print("=" * 75)

    print("""
1. WHAT `input_ids` REPRESENT:
   `input_ids` are discrete integer indices mapped from text using RoBERTa's Byte-Pair
   Encoding (BPE) vocabulary of 50,265 tokens. They represent positions in the model's
   learnable word embedding matrix:
       E in R^{50265 x 768}
   Special tokens are added automatically:
   - 0 (<s>): Beginning-of-sequence classification token (corresponds to [CLS] in BERT).
   - 2 (</s>): End-of-sequence delimiter token (corresponds to [SEP] in BERT).
   The leading character 'Ġ' represents a preceding whitespace boundary in RoBERTa's BPE.

2. WHAT `attention_mask` REPRESENTS:
   A binary 1D tensor of the same length as `input_ids`:
   - 1: Real token that should participate in multi-head self-attention.
   - 0: Padding token that must be ignored.
   During Scaled Dot-Product Attention:
       Attention(Q, K, V) = softmax( (Q K^T / sqrt(d_k)) + M ) V
   where M[i, j] = -infinity for padding positions (mask = 0). This drives their softmax
   weights to 0.0, ensuring pad tokens contribute zero influence to contextual hidden states.

3. WHY TRANSFORMERS REQUIRE TOKEN IDs:
   Neural networks operate strictly on numerical tensors, not raw text strings. Token IDs
   act as discrete row-lookup indices into the embedding matrix. Once retrieved, these 768-D
   static embeddings are combined with learned positional embeddings:
       h_0 = Embedding(input_ids) + PositionEmbedding(positions)
   The Transformer's stacked multi-head self-attention blocks then project these vectors
   through Query, Key, and Value weights across 12 layers to build deep contextual representations.

4. WHY WE USE TRUNCATION (max_length = 128):
   Standard multi-head self-attention exhibits quadratic computational and memory complexity:
   O(N^2) where N is sequence length. Truncation bounds sequences to a strict ceiling,
   guaranteeing bounded tensor dimensions and preventing Out-Of-Memory (OOM) errors.

5. WHY WE DO NOT PAD THE ENTIRE DATASET TO 128 TOKENS YET:
   If we statically padded all 45,615 examples to 128 tokens right now:
   - Our average tweet length is ~29 tokens.
   - Padded to 128, ~77% of all computed tokens in matrix multiplications would be meaningless
     padding tokens (zeros), wasting enormous memory and CPU/GPU compute cycles.
   Instead, we use DYNAMIC PADDING at training time via `DataCollatorWithPadding`:
   Each mini-batch (e.g. 16 or 32 samples) is padded ONLY to the maximum length of that specific
   batch (often 35-45 tokens), yielding 2x to 3x training speedup and minimal memory consumption.

6. WHAT THE MODEL WILL RECEIVE DURING TRAINING:
   During each forward step, the PyTorch DataLoader will pass a dictionary of tensors:
   - `input_ids`: LongTensor of shape (batch_size, batch_max_len)
   - `attention_mask`: LongTensor of shape (batch_size, batch_max_len)
   - `labels`: LongTensor of shape (batch_size,) containing class IDs {0, 1, 2}
   The RoBERTa classification head (`RobertaForSequenceClassification`) will extract the pooled
   representation of the <s> token, pass it through a linear projection layer (768 -> 3),
   and compute cross-entropy loss against `labels`.
""")
    print("=" * 75)


def run_tokenization_pipeline() -> Tuple[PreTrainedTokenizerFast, DatasetDict]:
    """Execute complete tokenization inspection, distribution analysis, and dataset creation."""
    tokenizer = get_tokenizer()
    raw_datasets = load_tweet_sentiment_dataset()

    inspect_sample_sentences(tokenizer)
    analyze_sequence_lengths(raw_datasets["train"], tokenizer, max_length=MAX_LENGTH)
    tokenized_datasets = create_tokenized_dataset(raw_datasets, tokenizer, max_length=MAX_LENGTH)
    verify_tokenized_dataset(tokenized_datasets)
    print_educational_summary()

    return tokenizer, tokenized_datasets


if __name__ == "__main__":
    run_tokenization_pipeline()
