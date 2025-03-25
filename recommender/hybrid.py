from typing import List

import numpy as np

from .collaborative_filtering import CollaborativeFilteringRecommender
from .content_based import ContentBasedRecommender


class HybridRecommender:
    def __init__(
        self,
        cb_model: ContentBasedRecommender,
        cf_model: CollaborativeFilteringRecommender,
        alpha: float = 0.5,
    ):
        """Combine content-based and collaborative filtering using weight alpha."""
        self.cb_model = cb_model
        self.cf_model = cf_model
        self.alpha = alpha

    def recommend(self, user_id: int, item_id: int, top_k: int = 5) -> List[int]:
        """Generate hybrid recommendations combining both models."""
        cb_scores = np.zeros(self.cb_model.item_features.shape[0])
        cb_scores[self.cb_model.recommend(item_id, top_k * 2)] = 1

        cf_scores = np.zeros(self.cf_model.user_item_matrix.shape[1])
        cf_scores[self.cf_model.recommend(user_id, top_k * 2)] = 1

        hybrid_scores = self.alpha * cb_scores + (1 - self.alpha) * cf_scores
        return np.argsort(hybrid_scores)[::-1][:top_k].tolist()
