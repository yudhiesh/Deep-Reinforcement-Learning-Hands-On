import gym
import collections
from tensorboardX import SummaryWriter

ENV_NAME = "FrozenLake-v0"
# ENV_NAME = "FrozenLake8x8-v0"      # uncomment for larger version
GAMMA = 0.9
TEST_EPISODES = 20


class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state = self.env.reset()
        self.rewards = collections.defaultdict(float)
        self.transits = collections.defaultdict(collections.Counter)
        self.values = collections.defaultdict(float)

    def play_n_random_steps(self, count):
        for _ in range(count):
            action = self.env.action_space.sample()
            new_state, reward, is_done, _ = self.env.step(action)
            self.rewards[(self.state, action, new_state)] = reward
            self.transits[(self.state, action)][new_state] += 1
            self.state = self.env.reset() if is_done else new_state

    def calc_action_value(self, state, action):
        target_counts = self.transits[(state, action)]
        total = sum(target_counts.values())
        action_value = 0.0
        for tgt_state, count in target_counts.items():
            reward = self.rewards[(state, action, tgt_state)]
            val = reward + GAMMA * self.values[tgt_state]
            action_value += (count / total) * val
        return action_value

    def select_action(self, state):
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            action_value = self.calc_action_value(state, action)
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action

    def play_episode(self, env):
        total_reward = 0.0
        state = env.reset()
        while True:
            action = self.select_action(state)
            new_state, reward, is_done, _ = env.step(action)
            self.rewards[(state, action, new_state)] = reward
            self.transits[(state, action)][new_state] += 1
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward

    def value_iteration(self):
        for state in range(self.env.observation_space.n):
            state_values = [
                self.calc_action_value(state, action)
                for action in range(self.env.action_space.n)
            ]
            self.values[state] = max(state_values)


if __name__ == "__main__":
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment="-v-iteration")

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        agent.play_n_random_steps(100)
        agent.value_iteration()

        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES
        writer.add_scalar("reward", reward, iter_no)
        if reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (best_reward, reward))
            best_reward = reward
        if reward > 0.80:
            print("Solved in %d iterations!" % iter_no)
            break
    writer.close()

# Transitions: defaultdict(<class 'collections.Counter'>, {(0, 2): Counter({0: 37, 4: 37, 1: 24}), (0, 1): Counter({1: 70, 0: 61, 4: 56}), (0, 3): Counter({0: 73, 1: 32}), (1, 1): Counter({5: 18, 0: 16, 2: 14}), (0, 0): Counter({0: 788, 4: 373}), (4, 3): Counter({5: 15, 4: 13, 0: 12}), (4, 2): Counter({8: 14, 5: 12, 0: 10}), (2, 2): Counter({3: 33, 2: 27, 6: 25}), (3, 1): Counter({2: 5, 3: 4, 7: 2}), (2, 3): Counter({1: 8, 2: 5, 3: 4}), (1, 0): Counter({5: 12, 0: 10, 1: 6}), (1, 2): Counter({5: 21, 1: 19, 2: 18}), (4, 1): Counter({4: 25, 8: 23, 5: 22}), (8, 0): Counter({4: 95, 8: 92, 12: 86}), (8, 2): Counter({9: 33, 4: 19, 12: 10}), (8, 1): Counter({8: 20, 9: 13, 12: 10}), (9, 2): Counter({13: 14, 5: 12, 10: 6}), (10, 0): Counter({14: 9, 9: 9, 6: 6}), (14, 0): Counter({10: 3, 13: 1}), (13, 0): Counter({12: 2, 9: 1}), (4, 0): Counter({4: 256, 0: 250, 8: 225}), (1, 3): Counter({1: 28, 0: 20, 2: 17}), (8, 3): Counter({8: 6, 4: 4, 9: 3}), (2, 1): Counter({3: 9, 1: 7, 6: 5}), (3, 3): Counter({3: 54, 2: 34}), (2, 0)^C5, 3): Counter()})

# Rewards : defaultdict(<class 'float'>, {(0, 3, 0): 0.0, (0, 2, 1): 0.0, (1, 0, 0): 0.0, (0, 2, 4): 0.0, (4, 2, 5): 0.0, (0, 1, 1): 0.0, (1, 1, 0): 0.0, (4, 3, 5): 0.0, (0, 0, 0): 0.0, (0, 2, 0): 0.0, (0, 1, 0): 0.0, (4, 1, 5): 0.0, (0, 0, 4): 0.0, (0, 1, 4): 0.0, (0, 3, 1): 0.0, (1, 0, 1): 0.0, (1, 2, 5): 0.0, (1, 3, 0): 0.0, (4, 0, 8): 0.0, (8, 3, 9): 0.0, (9, 2, 10): 0.0, (10, 3, 11): 0.0, (1, 1, 2): 0.0, (2, 2, 2): 0.0, (2, 1, 6): 0.0, (6, 2, 7): 0.0, (4, 0, 4): 0.0, (4, 1, 4): 0.0, (4, 3, 4): 0.0, (4, 3, 0): 0.0, (4, 0, 0): 0.0, (8, 1, 9): 0.0, (9, 1, 10): 0.0, (10, 1, 14): 0.0, (14, 0, 14): 0.0, (14, 0, 10): 0.0, (10, 1, 11): 0.0, (1, 1, 5): 0.0, (4, 1, 8): 0.0, (8, 1, 8): 0.0, (8, 2, 9): 0.0, (9, 0, 8): 0.0, (8, 2, 4): 0.0, (1, 0, 5): 0.0, (8, 3, 8): 0.0, (8, 2, 12): 0.0, (8, 0, 12): 0.0, (8, 0, 4): 0.0, (8, 0, 8): 0.0, (8, 1, 12): 0.0, (1, 2, 2): 0.0, (2, 1, 1): 0.0, (1, 3, 2): 0.0, (2, 2, 3): 0.0, (3, 2, 7): 0.0, (4, 2, 0): 0.0, (1, 3, 1): 0.0, (4, 2, 8): 0.0, (9, 2, 13): 0.0, (13, 0, 13): 0.0, (13, 1, 13): 0.0, (13, 3, 14): 0.0, (14, 2, 10): 0.0, (2, 3, 2): 0.0, (2, 2, 6): 0.0, (6, 1, 10): 0.0, (10, 3, 6): 0.0, (6, 2, 10): 0.0, (6, 0, 5): 0.0, (2, 3, 1): 0.0, (9, 3, 8): 0.0, (2, 0, 1): 0.0, (2, 1, 3): 0.0, (3, 3, 2): 0.0, (6, 0, 2): 0.0, (10, 2, 14): 0.0, (14, 2, 14): 0.0, (14, 1, 14): 0.0, (14, 2, 15): 1.0, (9, 0, 13): 0.0, (13, 1, 14): 0.0, (1, 2, 1): 0.0, (2, 0, 6): 0.0, (6, 1, 5): 0.0, (8, 3, 4): 0.0, (6, 0, 10): 0.0, (9, 0, 5): 0.0, (3, 1, 7): 0.0, (2, 0, 2): 0.0, (6, 2, 2): 0.0, (9, 2, 5): 0.0, (9, 1, 13): 0.0, (13, 3, 9): 0.0, (10, 2, 6): 0.0, (6, 1, 7): 0.0, (9, 1, 8): 0.0, (13, 1, 12): 0.0, (10, 2, 11): 0.0, (13, 0, 9): 0.0, (13, 0, 12): 0.0, (13, 3, 12): 0.0, (6, 3, 2): 0.0, (6, 3, 7): 0.0, (3, 1, 2): 0.0, (9, 3, 10): 0.0, (14, 3, 15): 1.0, (3, 3, 3): 0.0, (14, 3, 10): 0.0, (14, 1, 13): 0.0, (14, 3, 13): 0.0, (2, 3, 3): 0.0, (3, 0, 3): 0.0, (6, 3, 5): 0.0, (3, 2, 3): 0.0, (3, 1, 3): 0.0, (3, 0, 2): 0.0, (3, 0, 7): 0.0, (10, 1, 9): 0.0, (13, 2, 14): 0.0, (10, 0, 14): 0.0, (10, 0, 9): 0.0, (10, 0, 6): 0.0, (13, 2, 9): 0.0, (13, 2, 13): 0.0, (10, 3, 9): 0.0, (9, 3, 5): 0.0})
