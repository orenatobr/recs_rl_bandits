import numpy as np

from bandits.epsilon_greedy import EpsilonGreedyBandit
from bandits.thompson_sampling import ThompsonSamplingBandit
from bandits.ucb import UCBBandit


def test_epsilon_greedy_bandit():
    bandit = EpsilonGreedyBandit(n_arms=3, epsilon=0.1)
    for _ in range(100):
        arm = bandit.select_arm()
        reward = np.random.choice([0, 1])
        bandit.update(arm, reward)
    assert bandit.counts.sum() == 100


def test_ucb_bandit():
    bandit = UCBBandit(n_arms=3)
    for _ in range(100):
        arm = bandit.select_arm()
        reward = np.random.choice([0, 1])
        bandit.update(arm, reward)
    assert bandit.counts.sum() == 100


def test_thompson_sampling_bandit():
    bandit = ThompsonSamplingBandit(n_arms=3)
    for _ in range(100):
        arm = bandit.select_arm()
        reward = np.random.choice([0, 1])
        bandit.update(arm, reward)
    assert (
        bandit.successes.sum() + bandit.failures.sum() >= 6
    )  # At least 2 updates per arm
