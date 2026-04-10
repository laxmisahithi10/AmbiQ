import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load dataset
data = pd.read_csv("dataset_quora.csv")
print("Enhanced Dataset loaded successfully!")
print(f"Total samples: {len(data)}")
print(f"\nClass distribution:\n{data['label'].value_counts()}")

# Preprocessing
X = data['question']
y = data['label'].map({'clear': 0, 'ambiguous': 1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=150, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression(random_state=42, max_iter=300)
model.fit(X_train_tfidf, y_train)

# Predictions
y_pred = model.predict(X_test_tfidf)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=['Clear', 'Ambiguous'])

print(f"\n{'='*50}")
print("MODEL EVALUATION RESULTS")
print(f"{'='*50}")
print(f"\nAccuracy: {accuracy:.2%}")
print(f"\nClassification Report:\n{class_report}")
print(f"\nConfusion Matrix:\n{conf_matrix}")

# Visualize Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Clear', 'Ambiguous'], 
            yticklabels=['Clear', 'Ambiguous'])
plt.title('Confusion Matrix - Ambiguity Detection Model\n(Enhanced Dataset - 307 Questions)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix_quora.png', dpi=300, bbox_inches='tight')
print("\nConfusion matrix saved as 'confusion_matrix_quora.png'")

# Feature importance (top words)
feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_[0]
top_clear_indices = np.argsort(coefficients)[:15]
top_ambiguous_indices = np.argsort(coefficients)[-15:]

print(f"\n{'='*50}")
print("TOP FEATURES")
print(f"{'='*50}")
print("\nTop words indicating CLEAR questions:")
for idx in top_clear_indices:
    print(f"  - {feature_names[idx]}: {coefficients[idx]:.3f}")

print("\nTop words indicating AMBIGUOUS questions:")
for idx in top_ambiguous_indices[::-1]:
    print(f"  - {feature_names[idx]}: {coefficients[idx]:.3f}")

# Save model and vectorizer
joblib.dump(model, 'ambiguity_model_quora.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer_quora.pkl')
print("\nModel saved as 'ambiguity_model_quora.pkl'")
print("Vectorizer saved as 'tfidf_vectorizer_quora.pkl'")

# Test with sample questions
print(f"\n{'='*50}")
print("SAMPLE PREDICTIONS")
print(f"{'='*50}")
test_questions = [
    "What is the weather today?",
    "Book it",
    "How do I install Python?",
    "Send that file",
    "What is machine learning?",
    "Fix the bug",
    "How does photosynthesis work?",
    "Delete this",
    "What are the benefits of exercise?",
    "Order now"
]

for question in test_questions:
    question_tfidf = vectorizer.transform([question])
    prediction = model.predict(question_tfidf)[0]
    probability = model.predict_proba(question_tfidf)[0]
    label = "Ambiguous" if prediction == 1 else "Clear"
    confidence = probability[prediction] * 100
    print(f"\nQuestion: '{question}'")
    print(f"Prediction: {label} (Confidence: {confidence:.1f}%)")
