from typing import List

def precision_at_k(recommended: List[int], relevant: List[int], k: int = 5) -> float:
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & set(relevant))
    return hits / k
