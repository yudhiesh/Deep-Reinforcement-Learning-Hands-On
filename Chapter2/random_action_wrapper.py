import gym
from typing import TypeVar
import random

# The wrapper classes enable the user to override the base Env class with specific methods
# You have access to the ObservationWrapper, ActionWrapper and the RewardWrapper

Action = TypeVar("Action")


class RandomActionWrapper(gym.ActionWrapper):
    def __init__(self, env, epsilon=0.1):
        super(RandomActionWrapper, self).__init__(env)
        self.epsilon = epsilon

    # For this example let us override the value of the actions with a random one for fun
    def action(self, action: Action) -> Action:
        if random.random() < self.epsilon:
            print("Random!")
            return self.env.action_space.sample()
        return action


if __name__ == "__main__":
    env = RandomActionWrapper(gym.make("CartPole-v0"))
    obs = env.reset()
    total_reward = 0.0

    while True:
        obs, reward, done, _ = env.step(0)
        total_reward += reward
        if done:
            break

    print(f"Reward : {total_reward:.2f}")
