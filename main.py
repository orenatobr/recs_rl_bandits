import numpy as np
from recommender.content_based import ContentBasedRecommender
from recommender.collaborative_filtering import CollaborativeFilteringRecommender
from recommender.hybrid import HybridRecommender
from utils.metrics import precision_at_k

item_features = np.random.rand(10, 5)
user_item_matrix = np.random.randint(0, 2, (4, 10))

cb_model = ContentBasedRecommender(item_features)
cf_model = CollaborativeFilteringRecommender(user_item_matrix)
hybrid_model = HybridRecommender(cb_model, cf_model, alpha=0.7)

user_id = 1
item_id = 3
recommended_items = hybrid_model.recommend(user_id=user_id, item_id=item_id)

print("Recommended items:", recommended_items)

relevant_items = [2, 3, 7]  # exemplo
precision = precision_at_k(recommended_items, relevant_items)
print("Precision@k:", precision)
