import joblib
import os


class AmbiguityMLPredictor:
    """
    ML-based Question Ambiguity Predictor
    Uses TF-IDF + Logistic Regression
    """

    def __init__(self, model_path, vectorizer_path):
        """
        Load trained ML model and vectorizer

        Args:
            model_path (str): Path to trained model (.pkl)
            vectorizer_path (str): Path to TF-IDF vectorizer (.pkl)
        """

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")

        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

    def predict(self, question: str):
        """
        Predict whether a question is Clear or Ambiguous

        Args:
            question (str): Input question

        Returns:
            dict: Prediction result
        """

        if not question or not question.strip():
            return {
                "label": "Invalid",
                "is_ambiguous": False,
                "confidence": 0.0,
                "probabilities": {
                    "clear": 0.0,
                    "ambiguous": 0.0
                }
            }

        # Vectorize input
        X = self.vectorizer.transform([question])

        # Predict
        prediction = int(self.model.predict(X)[0])
        probabilities = self.model.predict_proba(X)[0]

        return {
            "label": "Ambiguous" if prediction == 1 else "Clear",
            "is_ambiguous": bool(prediction),
            "confidence": float(probabilities[prediction] * 100),
            "probabilities": {
                "clear": float(probabilities[0] * 100),
                "ambiguous": float(probabilities[1] * 100)
            }
        }

    def batch_predict(self, questions):
        """
        Predict ambiguity for multiple questions

        Args:
            questions (list[str]): List of questions

        Returns:
            list[dict]: List of predictions
        """
        return [self.predict(q) for q in questions]


# ------------------ TEST BLOCK ------------------
if __name__ == "__main__":
    predictor = AmbiguityMLPredictor(
        "ambiguity_model_quora.pkl",
        "tfidf_vectorizer_quora.pkl"
    )

    test_questions = [
        "What is machine learning?",
        "Book the ticket",
        "How does photosynthesis work?",
        "Send it",
        "What time does the store close?"
    ]

    print("\nML Model Predictions")
    print("=" * 60)

    for q in test_questions:
        result = predictor.predict(q)
        print(f"\nQuestion: {q}")
        print(f"Label: {result['label']}")
        print(f"Confidence: {result['confidence']:.2f}%")
        print(
            f"Clear: {result['probabilities']['clear']:.2f}% | "
            f"Ambiguous: {result['probabilities']['ambiguous']:.2f}%"
        )
