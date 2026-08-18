from classifier.predictor import Predictor

predictor = Predictor()

result = predictor.predict(
    "test_images/sample.png"
)

print("=" * 60)

print("Prediction")
print(result["prediction"])

print()

print("Confidence")
print(f"{result['confidence']:.2f}%")

print()

print("Top Matches")

for item in result["matches"]:
    print(item)