import pickle
import time
from collections import defaultdict

import numpy as np

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    faiss = None
    HAS_FAISS = False

import config
from models.feature_extractor import FeatureExtractor


class SimilaritySearch:

    def __init__(self):

        self.extractor = FeatureExtractor()

        self.metadata = None
        self.index = None
        self.embeddings = None
        self.use_faiss = HAS_FAISS

        if self.use_faiss:
            print("Loading FAISS Index...")

            self.index = faiss.read_index(
                config.FAISS_INDEX
            )
        else:
            print("FAISS not available; loading numpy fallback embeddings...")
            self.embeddings = np.load(
                "database/reference_embeddings.npy"
            )

        with open(
            config.METADATA,
            "rb"
        ) as f:
            self.metadata = pickle.load(f)

    ##############################################################

    def search(
        self,
        image_path,
        top_k=None
    ):

        if top_k is None:
            top_k = config.TOP_K

        timings = {}

        ##############################################################
        # Feature Extraction
        ##############################################################

        start = time.perf_counter()

        embedding = self.extractor.extract_features(
            image_path
        )

        embedding = np.asarray(
            embedding,
            dtype=np.float32
        ).reshape(1, -1)

        timings["feature_extraction"] = (
            time.perf_counter() - start
        ) * 1000

        ##############################################################
        # Search more neighbours
        ##############################################################

        start = time.perf_counter()

        if self.use_faiss:
            similarities, indices = self.index.search(
                embedding,
                top_k * 5
            )
        else:
            ref_embeddings = self.embeddings
            query = embedding / (np.linalg.norm(embedding, axis=1, keepdims=True) + 1e-12)
            similarities = np.dot(query, ref_embeddings.T)
            indices = np.argsort(-similarities, axis=1)[:,: top_k * 5]
            similarities = np.take_along_axis(similarities, indices, axis=1)

        timings["faiss"] = (
            time.perf_counter() - start
        ) * 1000

        ##############################################################
        # Keep only BEST 3 per class
        ##############################################################

        start = time.perf_counter()

        grouped = defaultdict(list)

        for score, idx in zip(
            similarities[0],
            indices[0]
        ):

            item = self.metadata[idx].copy()

            item["similarity"] = float(score)

            grouped[item["class"]].append(item)

        filtered = []

        for cls in grouped:

            grouped[cls].sort(
                key=lambda x: x["similarity"],
                reverse=True
            )

            filtered.extend(
                grouped[cls][:3]
            )

        filtered.sort(
            key=lambda x: x["similarity"],
            reverse=True
        )

        filtered = filtered[:top_k]

        timings["metadata"] = (
            time.perf_counter() - start
        ) * 1000

        return filtered, timings