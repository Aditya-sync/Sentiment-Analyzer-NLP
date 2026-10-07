import joblib


# Load trained pipeline
MODEL_PATH = "ml/models/sentiment_pipeline.pkl"

model = joblib.load(MODEL_PATH)

print("✅ Model loaded successfully!\n")


# Test sentences
test_sentences = [
    "I absolutely loved this movie. The acting was fantastic!",
    
    "This was one of the worst movies I have ever watched.",
    
    "The movie was okay, nothing special.",
    
    "Amazing story, brilliant acting and beautiful cinematography.",
    
    "The plot was boring and the characters were annoying.",
    
    "I would definitely recommend this movie to everyone.",
    
    "I regret wasting two hours watching this terrible movie.",
]


# Make predictions
predictions = model.predict(test_sentences)

probabilities = model.predict_proba(test_sentences)


# Display results
for text, prediction, probability in zip(
    test_sentences,
    predictions,
    probabilities
):

    sentiment = "POSITIVE 😊" if prediction == 1 else "NEGATIVE 😞"

    confidence = max(probability) * 100

    print("=" * 70)

    print(f"Review: {text}")

    print(f"Prediction: {sentiment}")

    print(f"Confidence: {confidence:.2f}%")