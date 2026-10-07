from load_data import load_reviews
import pandas as pd


# ----------------------------------------
# 1. Load the dataset
# ----------------------------------------

print("Loading IMDb dataset...\n")

train_df = load_reviews("train")
test_df = load_reviews("test")


# ----------------------------------------
# 2. Basic dataset information
# ----------------------------------------

print("===== DATASET SHAPE =====")

print(f"Training samples: {len(train_df)}")
print(f"Testing samples:  {len(test_df)}")


# ----------------------------------------
# 3. Label distribution
# ----------------------------------------

print("\n===== LABEL DISTRIBUTION =====")

print(train_df["label"].value_counts())

print("\nPercentage distribution:")

print(
    train_df["label"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ----------------------------------------
# 4. Missing values
# ----------------------------------------

print("\n===== MISSING VALUES =====")

print(train_df.isnull().sum())


# ----------------------------------------
# 5. Duplicate reviews
# ----------------------------------------

print("\n===== DUPLICATES =====")

duplicates = train_df["text"].duplicated().sum()

print(f"Duplicate reviews: {duplicates}")


# ----------------------------------------
# 6. Character length
# ----------------------------------------

train_df["character_count"] = train_df["text"].str.len()

print("\n===== CHARACTER COUNT =====")

print(f"Average: {train_df['character_count'].mean():.2f}")
print(f"Minimum: {train_df['character_count'].min()}")
print(f"Maximum: {train_df['character_count'].max()}")


# ----------------------------------------
# 7. Word count
# ----------------------------------------

train_df["word_count"] = train_df["text"].str.split().str.len()

print("\n===== WORD COUNT =====")

print(f"Average: {train_df['word_count'].mean():.2f}")
print(f"Minimum: {train_df['word_count'].min()}")
print(f"Maximum: {train_df['word_count'].max()}")


# ----------------------------------------
# 8. Example positive review
# ----------------------------------------

print("\n===== POSITIVE REVIEW EXAMPLE =====")

positive_review = train_df[train_df["label"] == 1].iloc[0]

print(positive_review["text"])


# ----------------------------------------
# 9. Example negative review
# ----------------------------------------

print("\n===== NEGATIVE REVIEW EXAMPLE =====")

negative_review = train_df[train_df["label"] == 0].iloc[0]

print(negative_review["text"])


# ----------------------------------------
# 10. Dataset summary
# ----------------------------------------

print("\n===== DATASET INFO =====")

print(train_df.info())