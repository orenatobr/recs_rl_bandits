import numpy as np

class UCBBandit:
    def __init__(self, n_arms: int):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.total_counts = 0

    def select_arm(self) -> int:
        self.total_counts += 1
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm

        ucb_values = self.values + np.sqrt((2 * np.log(self.total_counts)) / self.counts)
        return int(np.argmax(ucb_values))

    def update(self, chosen_arm: int, reward: float):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        self.values[chosen_arm] = ((n - 1) / n) * value + (1 / n) * reward
