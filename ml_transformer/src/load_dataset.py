"""
Dataset Loader for TweetEval Sentiment.

Loads the official train, validation, and test splits of 'cardiffnlp/tweet_eval' (config: 'sentiment')
directly from Hugging Face Datasets.
"""

from datasets import DatasetDict, load_dataset
from ml_transformer.src.config import DATASET_CONFIG, DATASET_NAME


def load_tweet_sentiment_dataset() -> DatasetDict:
    """
    Load the official TweetEval sentiment dataset from Hugging Face.

    Returns:
        DatasetDict: Contains 'train', 'validation', and 'test' splits.
    """
    print(f"[*] Loading dataset '{DATASET_NAME}' (config: '{DATASET_CONFIG}')...")
    raw_datasets = load_dataset(DATASET_NAME, DATASET_CONFIG)

    splits = list(raw_datasets.keys())
    print(f"[+] Successfully loaded splits: {splits}")
    for split in splits:
        print(f"    - {split}: {len(raw_datasets[split])} samples")

    return raw_datasets


if __name__ == "__main__":
    datasets = load_tweet_sentiment_dataset()
