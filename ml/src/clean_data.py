from load_data import load_reviews


print("Loading datasets...")

train_df = load_reviews("train")
test_df = load_reviews("test")

print(f"\nOriginal training samples: {len(train_df)}")
print(f"Original testing samples:  {len(test_df)}")


# --------------------------------------------------
# 1. Remove duplicate reviews within training data
# --------------------------------------------------

before_train = len(train_df)

train_df = train_df.drop_duplicates(
    subset="text",
    keep="first"
).reset_index(drop=True)

removed_train_duplicates = before_train - len(train_df)

print(
    f"\nRemoved duplicate training reviews: "
    f"{removed_train_duplicates}"
)


# --------------------------------------------------
# 2. Find reviews appearing in BOTH datasets
# --------------------------------------------------

test_reviews = set(test_df["text"])

overlap_mask = train_df["text"].isin(test_reviews)

overlap_count = overlap_mask.sum()

print(
    f"Reviews appearing in both train and test: "
    f"{overlap_count}"
)


# --------------------------------------------------
# 3. Remove overlapping reviews from TRAIN only
# --------------------------------------------------

train_df = train_df[~overlap_mask].reset_index(drop=True)


print(
    f"\nFinal training samples: "
    f"{len(train_df)}"
)

print(
    f"Final testing samples: "
    f"{len(test_df)}"
)


# --------------------------------------------------
# 4. Verify there is no leakage anymore
# --------------------------------------------------

remaining_overlap = set(train_df["text"]).intersection(
    set(test_df["text"])
)

print(
    f"\nRemaining train/test overlap: "
    f"{len(remaining_overlap)}"
)


if len(remaining_overlap) == 0:
    print("\n✅ DATA LEAKAGE SUCCESSFULLY REMOVED")
else:
    print("\n❌ Leakage still exists!")
    
# --------------------------------------------------
# 5. Save cleaned datasets
# --------------------------------------------------

DATA_DIR = "ml/data"

train_df.to_csv(
    f"{DATA_DIR}/train_clean.csv",
    index=False
)

test_df.to_csv(
    f"{DATA_DIR}/test.csv",
    index=False
)

print("\nDatasets saved successfully!")

print("Saved:")
print("→ ml/data/train_clean.csv")
print("→ ml/data/test.csv")