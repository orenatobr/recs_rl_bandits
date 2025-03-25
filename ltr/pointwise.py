from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ndcg_score


class PointwiseLTR:
    def __init__(self):
        self.model = LogisticRegression()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X, y_true, group):
        scores = self.predict(X)
        return ndcg_score([y_true], [scores])
