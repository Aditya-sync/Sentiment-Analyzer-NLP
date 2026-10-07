import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from ml.src.preprocess import clean_text


# ============================================================
# 1. Load cleaned datasets
# ============================================================

print("Loading datasets...")

train_df = pd.read_csv("ml/data/train_clean.csv")
test_df = pd.read_csv("ml/data/test.csv")

print(f"Training samples: {len(train_df)}")
print(f"Testing samples:  {len(test_df)}")


# ============================================================
# 2. Prepare text and labels
# ============================================================

X_train = train_df["text"]
y_train = train_df["label"]

X_test = test_df["text"]
y_test = test_df["label"]


# ============================================================
# 3. Build NLP + ML pipeline
# ============================================================

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            max_features=100000
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            C=2.0
        )
    )
])


# ============================================================
# 4. Clean the text
# ============================================================

print("\nCleaning text...")

X_train_clean = X_train.apply(clean_text)
X_test_clean = X_test.apply(clean_text)


# ============================================================
# 5. Train
# ============================================================

print("\nTraining model...")
print("This may take a little while...")

model.fit(X_train_clean, y_train)

print("✅ Training complete!")


# ============================================================
# 6. Make predictions
# ============================================================

print("\nMaking predictions...")

predictions = model.predict(X_test_clean)


# ============================================================
# 7. Accuracy
# ============================================================

accuracy = accuracy_score(y_test, predictions)

print("\n==============================")
print("       MODEL PERFORMANCE")
print("==============================")

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")


# ============================================================
# 8. Classification report
# ============================================================

print("\n===== CLASSIFICATION REPORT =====")

print(
    classification_report(
        y_test,
        predictions,
        target_names=["Negative", "Positive"]
    )
)


# ============================================================
# 9. Confusion matrix
# ============================================================

print("\n===== CONFUSION MATRIX =====")

cm = confusion_matrix(y_test, predictions)

print(cm)


# ============================================================
# 10. Save trained pipeline
# ============================================================

MODEL_PATH = "ml/models/sentiment_pipeline.pkl"

joblib.dump(model, MODEL_PATH)

print("\n==============================")
print("MODEL SAVED")
print("==============================")

print(f"Saved to: {MODEL_PATH}")