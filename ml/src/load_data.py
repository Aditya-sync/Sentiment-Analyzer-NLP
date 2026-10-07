from pathlib import Path
import pandas as pd


# Project root / IMDb dataset location
DATASET_PATH = Path("dataset.gz/aclImdb")


def load_reviews(split: str) -> pd.DataFrame:
    """
    Load IMDb reviews from the specified split.

    Args:
        split: "train" or "test"

    Returns:
        DataFrame containing review text and sentiment label.
    """

    data = []

    split_path = DATASET_PATH / split

    # Positive reviews
    positive_path = split_path / "pos"

    for file_path in positive_path.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")
        data.append({
            "text": text,
            "label": 1
        })

    # Negative reviews
    negative_path = split_path / "neg"

    for file_path in negative_path.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")
        data.append({
            "text": text,
            "label": 0
        })

    return pd.DataFrame(data)


if __name__ == "__main__":

    print("Loading training data...")

    train_df = load_reviews("train")

    print(f"Training samples: {len(train_df)}")

    print("\nLoading test data...")

    test_df = load_reviews("test")

    print(f"Test samples: {len(test_df)}")

    print("\nFirst 5 reviews:")
    print(train_df.head())

    print("\nLabel distribution:")
    print(train_df["label"].value_counts())