from collections import defaultdict
import time

import config
from search.similarity_search import SimilaritySearch


class Predictor:

    def __init__(self):

        self.search_engine = SimilaritySearch()

    ############################################################

    def predict(self, image_path):

        overall_start = time.perf_counter()

        results, search_timings = self.search_engine.search(
            image_path,
            top_k=config.TOP_K
        )

        ############################################################
        # Weighted Voting
        ############################################################

        vote_start = time.perf_counter()

        class_score = defaultdict(float)

        top_similarity = defaultdict(float)

        class_count = defaultdict(int)

        for rank, item in enumerate(results):

            cls = item["class"]
            sim = item["similarity"]

            class_count[cls] += 1

            if sim > top_similarity[cls]:
                top_similarity[cls] = sim

            # Rank Weight
            rank_weight = 1.0 / (rank + 1)

            # Similarity Weight
            similarity_weight = sim * sim

            class_score[cls] += rank_weight * similarity_weight

        ############################################################
        # Prototype Boost
        ############################################################

        prototype_bonus = 0.15

        final_scores = {}

        for cls in class_score:

            score = class_score[cls]

            score += prototype_bonus * top_similarity[cls]

            final_scores[cls] = score

        ############################################################

        prediction = max(
            final_scores,
            key=final_scores.get
        )

        total = sum(final_scores.values())

        confidence = (
            final_scores[prediction] /
            total
        ) * 100

        ############################################################

        vote_time = (
            time.perf_counter()
            - vote_start
        ) * 1000

        total_time = (
            time.perf_counter()
            - overall_start
        ) * 1000

        ############################################################

        if confidence < config.CONFIDENCE_THRESHOLD:

            print()

            print("Top Matches")

            for i, item in enumerate(results):

                print(
                    f"{i+1}. "
                    f"{item['class']:20s} "
                    f"{item['similarity']:.4f}"
                )

        ############################################################

        return {

            "prediction": prediction,

            "confidence": confidence,

            "matches": results,

            "timings": {

                "feature_extraction":
                    search_timings["feature_extraction"],

                "faiss":
                    search_timings["faiss"],

                "metadata":
                    search_timings["metadata"],

                "vote":
                    vote_time,

                "total":
                    total_time
            }
        }