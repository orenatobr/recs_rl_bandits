import logging

import numpy as np
import torch

from bandits.epsilon_greedy import EpsilonGreedyBandit
from bandits.thompson_sampling import ThompsonSamplingBandit
from bandits.ucb import UCBBandit
from ltr.listwise import ListwiseLTR
from ltr.pairwise import PairwiseLTR
from ltr.pointwise import PointwiseLTR
from recommender.collaborative_filtering import CollaborativeFilteringRecommender
from recommender.content_based import ContentBasedRecommender
from recommender.hybrid import HybridRecommender
from rl.policy_gradient import REINFORCEAgent
from utils.metrics import cumulative_regret, precision_at_k

logging.basicConfig(level=logging.INFO)

# Recommender systems
item_features = np.random.rand(10, 5)
user_item_matrix = np.random.randint(0, 2, (4, 10))

cb_model = ContentBasedRecommender(item_features)
cf_model = CollaborativeFilteringRecommender(user_item_matrix)
hybrid_model = HybridRecommender(cb_model, cf_model, alpha=0.7)

user_id = 1
item_id = 3
recommended_items = hybrid_model.recommend(user_id=user_id, item_id=item_id)

logging.info(f"Recommended items: {recommended_items}")

relevant_items = [2, 3, 7]
precision = precision_at_k(recommended_items, relevant_items)
logging.info(f"Precision@k: {precision:.2f}")

# Multi-Armed Bandits
true_rewards = [0.1, 0.5, 0.8]
episodes = 100

for BanditClass in [EpsilonGreedyBandit, UCBBandit, ThompsonSamplingBandit]:
    bandit = (
        BanditClass(n_arms=len(true_rewards))
        if BanditClass != EpsilonGreedyBandit
        else BanditClass(n_arms=3, epsilon=0.1)
    )
    rewards = []
    for _ in range(episodes):
        arm = bandit.select_arm()
        reward = 1 if np.random.rand() < true_rewards[arm] else 0
        bandit.update(arm, reward)
        rewards.append(reward)
    regret = cumulative_regret(
        [max(true_rewards)] * episodes, [float(r) for r in rewards]
    )
    logging.info(f"{BanditClass.__name__} Regret: {regret:.2f}")


# RL: Policy Gradient (REINFORCE)
class SimpleEnv:
    def __init__(self):
        self.state = 0
        self.n_states = 5
        self.n_actions = 2

    def reset(self):
        self.state = np.random.randint(0, self.n_states)
        return self.state

    def step(self, action):
        reward = 1 if (self.state == 2 and action == 1) else 0
        self.state = (self.state + 1) % self.n_states
        done = self.state == 0
        return self.state, reward, done


logging.info("Training REINFORCE agent")
rl_env = SimpleEnv()
agent = REINFORCEAgent(n_states=5, n_actions=2)

for episode in range(10):
    trajectory = []
    state = rl_env.reset()
    done = False
    while not done:
        action = agent.select_action(state)
        next_state, reward, done = rl_env.step(action)
        trajectory.append((state, action, reward))
        state = next_state
    agent.update(trajectory)
logging.info("Finished training REINFORCE")

# Learning to Rank: Pointwise
logging.info("Training PointwiseLTR model")
X = np.array([[0.2, 0.8], [0.4, 0.6], [0.5, 0.5], [0.9, 0.1]])
y = np.array([1, 0, 1, 0])
group = [4]  # all items in same group
ltr_pointwise = PointwiseLTR()
ltr_pointwise.fit(X, y)
pred = ltr_pointwise.predict(X)
score = ltr_pointwise.evaluate(X, y, group)
logging.info(f"PointwiseLTR NDCG: {score:.2f}")

# Learning to Rank: Pairwise
x_i = torch.tensor([[0.6, 0.4], [0.7, 0.3]])
x_j = torch.tensor([[0.2, 0.8], [0.1, 0.9]])
y_ij = torch.tensor([1, 1])
ltr_pairwise = PairwiseLTR(input_dim=2)
for _ in range(10):
    pairwise_loss = ltr_pairwise.train_step(x_i, x_j, y_ij)
logging.info(f"PairwiseLTR loss after training: {pairwise_loss:.4f}")

# Learning to Rank: Listwise
x_list = torch.tensor([[0.6, 0.4], [0.3, 0.7], [0.8, 0.2]])
y_list = torch.tensor([2.0, 1.0, 3.0])
ltr_listwise = ListwiseLTR(input_dim=2)
for _ in range(10):
    listwise_loss = ltr_listwise.train_step(x_list, y_list)
logging.info(f"ListwiseLTR loss after training: {listwise_loss:.4f}")
