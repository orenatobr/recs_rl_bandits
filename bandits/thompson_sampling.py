import numpy as np

class ThompsonSamplingBandit:
    def __init__(self, n_arms: int):
        self.n_arms = n_arms
        self.successes = np.ones(n_arms)
        self.failures = np.ones(n_arms)

    def select_arm(self) -> int:
        samples = np.random.beta(self.successes, self.failures)
        return int(np.argmax(samples))

    def update(self, chosen_arm: int, reward: float):
        if reward == 1:
            self.successes[chosen_arm] += 1
        else:
            self.failures[chosen_arm] += 1
            