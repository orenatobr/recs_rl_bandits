import random

import numpy as np


class QLearningAgent:
    def __init__(
        self, n_states: int, n_actions: int, alpha=0.1, gamma=0.99, epsilon=0.1
    ):
        """
        Q-Learning Agent for discrete environments.

        Args:
            n_states (int): Number of discrete states
            n_actions (int): Number of possible actions
            alpha (float): Learning rate
            gamma (float): Discount factor
            epsilon (float): Exploration rate
        """
        self.q_table = np.zeros((n_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions

    def select_action(self, state: int) -> int:
        """Choose action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        return int(np.argmax(self.q_table[state]))

    def update(self, state: int, action: int, reward: float, next_state: int):
        """Update Q-values using the Bellman equation."""
        best_next = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error
