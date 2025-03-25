from typing import List
import numpy as np

class CollaborativeFilteringRecommender:
    def __init__(self, user_item_matrix: np.ndarray):
        self.user_item_matrix = user_item_matrix

    def recommend(self, user_id: int, top_k: int = 5) -> List[int]:
        user_vector = self.user_item_matrix[user_id]
        scores = self.user_item_matrix.T @ user_vector
        recommended_indices = np.argsort(scores)[::-1]
        return recommended_indices[:top_k].tolist()
    