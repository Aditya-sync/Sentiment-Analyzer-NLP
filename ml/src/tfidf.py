from src.preprocess import clean_text
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# --------------------------------------------------
# 1. Load cleaned datasets
# --------------------------------------------------

print("Loading cleaned datasets...")

train_df = pd.read_csv("ml/data/train_clean.csv")
test_df = pd.read_csv("ml/data/test.csv")

print(f"Training samples: {len(train_df)}")
print(f"Testing samples:  {len(test_df)}")


# --------------------------------------------------
# 2. Separate text and labels
# --------------------------------------------------

X_train = train_df["text"]
y_train = train_df["label"]

X_test = test_df["text"]
y_test = test_df["label"]


# --------------------------------------------------
# 3. Create TF-IDF vectorizer
# --------------------------------------------------

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    max_features=100000
)


# --------------------------------------------------
# 4. Learn vocabulary ONLY from training data
# --------------------------------------------------

print("\nFitting TF-IDF on training data...")

X_train_tfidf = vectorizer.fit_transform(X_train)


# --------------------------------------------------
# 5. Transform test data
# --------------------------------------------------

print("Transforming test data...")

X_test_tfidf = vectorizer.transform(X_test)


# --------------------------------------------------
# 6. Display results
# --------------------------------------------------

print("\n===== TF-IDF RESULTS =====")

print("Training matrix shape:")
print(X_train_tfidf.shape)

print("\nTesting matrix shape:")
print(X_test_tfidf.shape)

print("\nVocabulary size:")
print(len(vectorizer.vocabulary_))


# --------------------------------------------------
# 7. Inspect some vocabulary
# --------------------------------------------------

print("\n===== SAMPLE VOCABULARY =====")

vocabulary = vectorizer.get_feature_names_out()

print(vocabulary[:50])


# --------------------------------------------------
# 8. Inspect one review
# --------------------------------------------------

print("\n===== FIRST REVIEW =====")

print(X_train.iloc[0])

print("\nTF-IDF vector shape:")

print(X_train_tfidf[0].shape)