import numpy as np

from rl.policy_gradient import REINFORCEAgent


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


def test_reinforce_training():
    env = SimpleEnv()
    agent = REINFORCEAgent(n_states=5, n_actions=2)

    for _ in range(10):
        trajectory = []
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            trajectory.append((state, action, reward))
            state = next_state
        agent.update(trajectory)

    assert agent.policy.shape == (5, 2)
    assert np.allclose(agent.policy.sum(axis=1), 1.0)
