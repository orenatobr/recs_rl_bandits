from rl.q_learning import QLearningAgent
from rl.simple_env import SimpleEnv


def test_q_learning_train():
    env = SimpleEnv()
    agent = QLearningAgent(n_states=5, n_actions=2)

    for _ in range(50):
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state

    assert agent.q_table.shape == (5, 2)
    assert (agent.q_table >= 0).all()
