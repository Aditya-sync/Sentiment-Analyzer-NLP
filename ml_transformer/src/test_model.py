"""
Standalone Inference Script for RoBERTa Sentiment Classification.

Loads fine-tuned weights from ml_transformer/models/roberta-tweeteval/ (or base model
for local verification if fine-tuned checkpoint is not yet generated) and runs inference
on canonical test sentences using real model logits and softmax probabilities.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Union

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ml_transformer.src.config import ID2LABEL, LABEL2ID, MAX_LENGTH, MODEL_NAME, NUM_LABELS, SAVED_MODEL_DIR

TEST_SENTENCES = [
    "I absolutely loved this movie!",
    "The movie was okay, nothing special.",
    "I hated every minute of it.",
    "It arrived on time.",
    "I expected more, but the camera is usable.",
    "I do not dislike it.",
    "This product is fantastic and I would definitely recommend it.",
    "The product works, but there is nothing impressive about it.",
    "This is the worst purchase I have ever made.",
]


def load_inference_pipeline(model_path: Union[str, Path] = SAVED_MODEL_DIR):
    """
    Load model and tokenizer for inference.
    If fine-tuned weights exist at model_path, loads them.
    Otherwise, gracefully falls back to base RoBERTa with a clear notification.
    """
    model_path = Path(model_path)
    if model_path.exists() and (model_path / "config.json").exists():
        print(f"[*] Loading fine-tuned model checkpoint from: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
        is_fine_tuned = True
    else:
        print(f"[!] Fine-tuned checkpoint not found at: {model_path}")
        print(f"[*] Loading base model '{MODEL_NAME}' for inference verification...")
        print(f"    (Note: Base weights will produce untrained classification probabilities until Colab fine-tuning is completed.)")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_LABELS,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
        is_fine_tuned = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return model, tokenizer, is_fine_tuned


def predict_sentiment(text: str, model, tokenizer) -> Dict[str, Union[str, float]]:
    """
    Compute real forward pass logits and softmax probabilities for a text input.
    """
    device = next(model.parameters()).device
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

    pred_idx = int(torch.argmax(logits).item())
    pred_label = ID2LABEL[pred_idx]

    return {
        "text": text,
        "predicted_class": pred_label,
        "predicted_id": pred_idx,
        "negative_prob": float(probs[0]),
        "neutral_prob": float(probs[1]),
        "positive_prob": float(probs[2]),
        "logits": [round(float(x), 4) for x in logits.tolist()],
    }


def run_inference_suite(model_path: Union[str, Path] = SAVED_MODEL_DIR, sentences: List[str] = TEST_SENTENCES) -> None:
    """Run sentiment inference on test sentences and display formatted results."""
    model, tokenizer, is_fine_tuned = load_inference_pipeline(model_path)

    status_tag = "FINE-TUNED MODEL CHECKPOINT" if is_fine_tuned else "BASE ROBERTA (PRE-FINE-TUNING)"
    print("\n" + "=" * 75)
    print(f" SENTIMENT INFERENCE RESULTS ({status_tag})")
    print("=" * 75)

    for i, sentence in enumerate(sentences, 1):
        res = predict_sentiment(sentence, model, tokenizer)
        print(f"\n[Sentence {i}]")
        print(f"  Text:                 \"{res['text']}\"")
        print(f"  Predicted Class:      {res['predicted_class'].upper()} (id={res['predicted_id']})")
        print(f"  Negative Probability: {res['negative_prob'] * 100:>6.2f}%")
        print(f"  Neutral Probability:  {res['neutral_prob'] * 100:>6.2f}%")
        print(f"  Positive Probability: {res['positive_prob'] * 100:>6.2f}%")
        print(f"  Raw Logits:           {res['logits']}")

    print("\n" + "=" * 75)


def main():
    parser = argparse.ArgumentParser(description="RoBERTa Sentiment Inference Script")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(SAVED_MODEL_DIR),
        help="Path to saved fine-tuned model checkpoint directory",
    )
    args = parser.parse_args()
    run_inference_suite(model_path=args.model_dir)


if __name__ == "__main__":
    main()
