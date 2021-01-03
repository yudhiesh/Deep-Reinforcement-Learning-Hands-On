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

    def select_action(self, state):
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            action_value = self.values[(state, action)]
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
            for action in range(self.env.action_space.n):
                action_value = 0.0
                target_counts = self.transits[(state, action)]
                total = sum(target_counts.values())
                for tgt_state, count in target_counts.items():
                    key = (state, action, tgt_state)
                    reward = self.rewards[key]
                    best_action = self.select_action(tgt_state)
                    val = reward + GAMMA * self.values[(tgt_state, best_action)]
                    action_value += (count / total) * val
                self.values[(state, action)] = action_value


if __name__ == "__main__":
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment="-q-iteration")

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

# Q Values: defaultdict(<class 'float'>, {(4, 0): 0.082228093737905, (4, 1): 0.06166155306374471, (4, 2): 0.057134185819578284, (4, 3): 0.04826337148421127, (0, 0): 0.06210615439095045, (0, 1): 0.059390452626235254, (0, 2): 0.05949131297999493, (0, 3): 0.05375780535523347, (1, 0): 0.0340020006032031, (1, 1): 0.039785065403588046, (1, 2): 0.034523072417260436, (1, 3): 0.054917755467398535, (5, 0): 0.0, (5, 1): 0.0, (5, 2): 0.0, (5, 3): 0.0, (2, 0): 0.06515394633560975, (2, 1): 0.0620510523056874, (2, 2): 0.06340591316501232, (2, 3): 0.051894044610334325, (6, 0): 0.08078873468067137, (6, 1): 0.056272640677751715, (6, 2): 0.09643959362329536, (6, 3): 0.014659637925512194, (3, 0): 0.034777251609808935, (3, 1): 0.03661273623228893, (3, 2): 0.030115974788499162, (3, 3): 0.05000831931893495, (8, 0): 0.06399765326898463, (8, 1): 0.09876363466604021, (8, 2): 0.08765924312480672, (8, 3): 0.13033288096972456, (10, 0): 0.2581827465725477, (10, 1): 0.2378904081389128, (10, 2): 0.18354108393131224, (10, 3): 0.09498300288249212, (7, 0): 0.0, (7, 1): 0.0, (7, 2): 0.0, (7, 3): 0.0, (12, 0): 0.0, (12, 1): 0.0, (12, 2): 0.0, (12, 3): 0.0, (9, 0): 0.13421912121209234, (9, 1): 0.22017041598501175, (9, 2): 0.17872908720653746, (9, 3): 0.12812221974510857, (11, 0): 0.0, (11, 1): 0.0, (11, 2): 0.0, (11, 3): 0.0, (13, 0): 0.17390224064678708, (13, 1): 0.2842770578043694, (13, 2): 0.3446614742888415, (13, 3): 0.2844052960904202, (14, 0): 0.36400179916571423, (14, 1): 0.4898331847206363, (14, 2): 0.5907767241289444, (14, 3): 0.4770972772768786, (15, 0): 0.0, (15, 1): 0.0, (15, 2): 0.0, (15, 3): 0.0})
