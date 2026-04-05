from ml_predictor import AmbiguityMLPredictor

p = AmbiguityMLPredictor('ambiguity_model_quora.pkl', 'tfidf_vectorizer_quora.pkl')

# Test various question types
test_cases = [
    # Clear booking questions (should be CLEAR)
    ("Book me a flight to Japan from India for 22nd March", "Clear"),
    ("Reserve a table for 4 people at 7 PM at Italian restaurant", "Clear"),
    ("Order 2 pizzas with extra cheese for delivery", "Clear"),
    ("Schedule a meeting tomorrow at 3 PM in room A", "Clear"),
    
    # Ambiguous booking questions (should be AMBIGUOUS)
    ("Book a flight", "Ambiguous"),
    ("Reserve a table", "Ambiguous"),
    ("Order pizza", "Ambiguous"),
    ("Schedule a meeting", "Ambiguous"),
    ("Book it", "Ambiguous"),
    ("Send that", "Ambiguous"),
    
    # Clear informational questions (should be CLEAR)
    ("What is machine learning?", "Clear"),
    ("How does photosynthesis work?", "Clear"),
    ("What is the capital of France?", "Clear"),
    ("How do I install Python?", "Clear"),
    
    # Edge cases
    ("Delete this", "Ambiguous"),
    ("Fix the bug", "Ambiguous"),
    ("Get it", "Ambiguous"),
]

print("=" * 80)
print("COMPREHENSIVE MODEL TEST")
print("=" * 80)

correct = 0
total = len(test_cases)

for question, expected in test_cases:
    result = p.predict(question)
    actual = result['label']
    is_correct = "PASS" if actual == expected else "FAIL"
    
    if actual == expected:
        correct += 1
    
    print(f"\n{is_correct} Question: '{question}'")
    print(f"  Expected: {expected} | Got: {actual}")
    print(f"  Clear: {result['probabilities']['clear']:.1f}% | Ambiguous: {result['probabilities']['ambiguous']:.1f}%")

print("\n" + "=" * 80)
print(f"ACCURACY: {correct}/{total} = {(correct/total)*100:.1f}%")
print("=" * 80)
