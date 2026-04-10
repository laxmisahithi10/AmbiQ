from ml_predictor import AmbiguityMLPredictor

p = AmbiguityMLPredictor('ambiguity_model_quora.pkl', 'tfidf_vectorizer_quora.pkl')
r = p.predict('Book me a flight to Japan from India for 22nd March')

print(f"Label: {r['label']}")
print(f"Confidence: {r['confidence']:.2f}%")
print(f"Clear: {r['probabilities']['clear']:.2f}%")
print(f"Ambiguous: {r['probabilities']['ambiguous']:.2f}%")
