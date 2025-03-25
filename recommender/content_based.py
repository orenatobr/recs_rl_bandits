from typing import List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    def __init__(self, item_features: np.ndarray):
        """Initialize the content-based recommender with item features."""
        self.item_features = item_features

    def recommend(self, item_id: int, top_k: int = 5) -> List[int]:
        """Recommend top_k items similar to the given item_id."""
        similarities = cosine_similarity(
            self.item_features[item_id].reshape(1, -1), self.item_features
        ).flatten()
        similar_indices = np.argsort(similarities)[::-1][1 : top_k + 1]
        return similar_indices.tolist()
