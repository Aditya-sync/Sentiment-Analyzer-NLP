from load_data import load_reviews


print("Loading training data...")
train_df = load_reviews("train")

print("Loading test data...")
test_df = load_reviews("test")


# Convert review text into sets
train_reviews = set(train_df["text"])
test_reviews = set(test_df["text"])


# Find exact duplicates between train and test
overlap = train_reviews.intersection(test_reviews)


print("\n===== DATA LEAKAGE CHECK =====")

print(f"Training reviews: {len(train_reviews)}")
print(f"Testing reviews:  {len(test_reviews)}")
print(f"Exact overlaps:   {len(overlap)}")


if len(overlap) == 0:
    print("\n✅ No exact review appears in both train and test.")
else:
    print("\n⚠️ Exact overlap detected!")

    print("\nExample overlapping review:")
    print(next(iter(overlap)))