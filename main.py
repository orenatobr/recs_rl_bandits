import numpy as np
from recommender.content_based import ContentBasedRecommender
from recommender.collaborative_filtering import CollaborativeFilteringRecommender
from recommender.hybrid import HybridRecommender
from utils.metrics import precision_at_k, cumulative_regret
from bandits.epsilon_greedy import EpsilonGreedyBandit
from bandits.ucb import UCBBandit
from bandits.thompson_sampling import ThompsonSamplingBandit

item_features = np.random.rand(10, 5)
user_item_matrix = np.random.randint(0, 2, (4, 10))

cb_model = ContentBasedRecommender(item_features)
cf_model = CollaborativeFilteringRecommender(user_item_matrix)
hybrid_model = HybridRecommender(cb_model, cf_model, alpha=0.7)

user_id = 1
item_id = 3
recommended_items = hybrid_model.recommend(user_id=user_id, item_id=item_id)

print("Recommended items:", recommended_items)

relevant_items = [2, 3, 7]
precision = precision_at_k(recommended_items, relevant_items)
print("Precision@k:", precision)

# Simulação de bandits
true_rewards = [0.1, 0.5, 0.8]
episodes = 100

for BanditClass in [EpsilonGreedyBandit, UCBBandit, ThompsonSamplingBandit]:
    bandit = BanditClass(n_arms=len(true_rewards)) if BanditClass != EpsilonGreedyBandit else BanditClass(n_arms=3, epsilon=0.1)
    rewards = []
    for _ in range(episodes):
        arm = bandit.select_arm()
        reward = 1 if np.random.rand() < true_rewards[arm] else 0
        bandit.update(arm, reward)
        rewards.append(reward)
    regret = cumulative_regret([max(true_rewards)] * episodes, rewards)
    print(f"{BanditClass.__name__} Regret:", regret)