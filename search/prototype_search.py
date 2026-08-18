import pickle

import numpy as np

import config


class PrototypeSearch:

    def __init__(self):

        with open(
            "database/prototype_database.pkl",
            "rb"
        ) as f:

            self.prototypes = pickle.load(f)

    def search(
        self,
        embedding,
        top_k=3
    ):

        scores = []

        for class_name, data in self.prototypes.items():

            prototype = data["embedding"]

            similarity = np.dot(
                embedding,
                prototype
            )

            scores.append(
                (
                    class_name,
                    float(similarity)
                )
            )

        scores.sort(
            key=lambda x: x[1],
            reverse=True
        )

        return scores[:top_k]