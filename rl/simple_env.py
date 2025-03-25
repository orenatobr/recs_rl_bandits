import numpy as np


class SimpleEnv:
    def __init__(self):
        """Toy environment with 5 states and 2 actions."""
        self.state = 0
        self.n_states = 5
        self.n_actions = 2

    def reset(self) -> int:
        self.state = np.random.randint(0, self.n_states)
        return self.state

    def step(self, action: int):
        """Perform action and return next state, reward, done."""
        reward = 1 if (self.state == 2 and action == 1) else 0
        self.state = (self.state + 1) % self.n_states
        done = self.state == 0
        return self.state, reward, done
