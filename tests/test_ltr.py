import numpy as np

from ltr.pointwise import PointwiseLTR


def test_pointwise_ltr():
    X = np.array([[0.2, 0.8], [0.4, 0.6], [0.5, 0.5], [0.9, 0.1]])
    y = np.array([1, 0, 1, 0])
    group = [4]  # 4 samples in uma lista (ndcg_score espera isso)
    model = PointwiseLTR()
    model.fit(X, y)
    pred = model.predict(X)
    score = model.evaluate(X, y, group)
    assert len(pred) == len(y)
    assert 0 <= score <= 1
