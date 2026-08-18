from models.feature_extractor import FeatureExtractor

extractor = FeatureExtractor()

embedding = extractor.extract_features("test_images/sample.png")

print("Embedding Shape:")
print(embedding.shape)

print("\nFirst 20 Values:")
print(embedding[:20])