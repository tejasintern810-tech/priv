import os
import json
import pickle

import numpy as np
import torch
import torch.nn.functional as F

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    faiss = None
    HAS_FAISS = False

from models.feature_extractor import FeatureExtractor
from config import REFERENCE_FOLDER


class DatabaseBuilder:

    def __init__(self):

        print("Loading Feature Extractor...")

        self.extractor = FeatureExtractor()

        self.embeddings = []
        self.metadata = []
        self.class_statistics = {}

    def build_database(self):

        print("\nBuilding Reference Database...\n")

        for class_name in sorted(os.listdir(REFERENCE_FOLDER)):

            class_path = os.path.join(REFERENCE_FOLDER, class_name)

            if not os.path.isdir(class_path):
                continue

            print(f"Processing Class: {class_name}")

            self.class_statistics[class_name] = 0

            for image_name in sorted(os.listdir(class_path)):

                image_path = os.path.join(class_path, image_name)

                # Ignore folders
                if not os.path.isfile(image_path):
                    continue

                # Ignore non-image files (Thumbs.db etc.)
                if not image_name.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
                ):
                    continue

                try:

                    embedding = self.extractor.extract_features(image_path)

                    embedding = F.normalize(
                        embedding,
                        p=2,
                        dim=0
                    )

                    embedding = embedding.cpu().numpy().astype(np.float32)

                    self.embeddings.append(embedding)

                    self.metadata.append({
                        "class": class_name,
                        "filename": image_name,
                        "path": image_path
                    })

                    self.class_statistics[class_name] += 1

                except Exception as e:

                    print(f"Skipped {image_name}")
                    print(e)

        print("\nSaving Reference Database...\n")

        self.embeddings = np.array(self.embeddings).astype(np.float32)

        embedding_dimension = self.embeddings.shape[1]

        os.makedirs("database", exist_ok=True)

        np.save("database/reference_embeddings.npy", self.embeddings)

        if HAS_FAISS:
            print("Creating FAISS Index...")
            index = faiss.IndexFlatIP(embedding_dimension)
            index.add(self.embeddings)
            faiss.write_index(index, "database/reference.index")
        else:
            print("FAISS not available; saved numpy embeddings for fallback search.")

        with open("database/reference_metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)

        with open("database/class_statistics.json", "w") as f:
            json.dump(self.class_statistics, f, indent=4)

        print("\n===================================")
        print("Database Created Successfully!")
        print("===================================")
        print(f"Total Images : {len(self.metadata)}")
        print(f"Embedding Size : {embedding_dimension}")
        print("===================================")