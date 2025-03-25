import numpy as np


class REINFORCEAgent:
    def __init__(self, n_states: int, n_actions: int, alpha=0.01, gamma=0.99):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.policy = np.ones((n_states, n_actions)) / n_actions

    def select_action(self, state: int) -> int:
        return int(np.random.choice(self.n_actions, p=self.policy[state]))

    def update(self, trajectory):
        for t, (state, action, reward) in enumerate(trajectory):
            Gt = sum(r * (self.gamma**i) for i, (_, _, r) in enumerate(trajectory[t:]))
            self.policy[state][action] += (
                self.alpha * Gt * (1 - self.policy[state][action])
            )
            for a in range(self.n_actions):
                if a != action:
                    self.policy[state][a] -= self.alpha * Gt * self.policy[state][a]
