import numpy as np


class ThompsonSamplingBandit:
    def __init__(self, n_arms: int):
        """
        Thompson Sampling algorithm for Multi-Armed Bandits.

        Args:
            n_arms (int): Number of available arms.
        """
        self.n_arms = n_arms
        self.successes = np.ones(n_arms)
        self.failures = np.ones(n_arms)

    def select_arm(self) -> int:
        """Select an arm based on beta distribution sampling."""
        samples = np.random.beta(self.successes, self.failures)
        return int(np.argmax(samples))

    def update(self, chosen_arm: int, reward: float):
        """Update beta distribution parameters based on the received reward."""
        if reward == 1:
            self.successes[chosen_arm] += 1
        else:
            self.failures[chosen_arm] += 1
