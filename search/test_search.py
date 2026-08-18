import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from search.similarity_search import SimilaritySearch

searcher = SimilaritySearch()

results = searcher.search(
    "test_images/sample.png",
    top_k=5
)

print()

print("Top Matches")

print("=" * 50)

for result in results:

    print(result)