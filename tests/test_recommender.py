import numpy as np

from recommender.collaborative_filtering import CollaborativeFilteringRecommender
from recommender.content_based import ContentBasedRecommender
from recommender.hybrid import HybridRecommender


def test_content_based():
    item_features = np.eye(5)
    model = ContentBasedRecommender(item_features)
    recs = model.recommend(0, top_k=2)
    assert isinstance(recs, list)
    assert len(recs) == 2
    assert 0 not in recs


def test_collaborative_filtering():
    matrix = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
    model = CollaborativeFilteringRecommender(matrix)
    recs = model.recommend(0, top_k=2)
    assert isinstance(recs, list)
    assert len(recs) == 2


def test_hybrid_recommender():
    features = np.eye(3)
    matrix = np.array([[1, 0, 1], [0, 1, 1]])
    cb_model = ContentBasedRecommender(features)
    cf_model = CollaborativeFilteringRecommender(matrix)
    hybrid = HybridRecommender(cb_model, cf_model, alpha=0.5)
    recs = hybrid.recommend(user_id=0, item_id=1, top_k=2)
    assert isinstance(recs, list)
    assert len(recs) == 2
