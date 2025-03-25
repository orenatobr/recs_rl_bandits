import numpy as np


class EpsilonGreedyBandit:
    def __init__(self, n_arms: int, epsilon: float):
        """
        Epsilon-Greedy algorithm for the Multi-Armed Bandit problem.

        Args:
            n_arms (int): Number of available arms.
            epsilon (float): Probability of exploration (choosing a random arm).
        """
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def select_arm(self) -> int:
        """Select an arm to pull using epsilon-greedy strategy."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_arms)
        return int(np.argmax(self.values))

    def update(self, chosen_arm: int, reward: float):
        """Update estimated value of the chosen arm using incremental mean."""
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        self.values[chosen_arm] = ((n - 1) / n) * value + (1 / n) * reward
