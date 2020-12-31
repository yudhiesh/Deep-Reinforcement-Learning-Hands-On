import random
from typing import List


class Environment:
    def __init__(self):
        self.steps_left = 10

    # Returns the current environments observations to the agent
    def get_observations(self) -> List[float]:
        return [0.0, 0.0, 0.0, 0.0]

    # This method allows the agent to query the actions that the agent can take in the environment
    # This example only allows the agent to make actions of 0 or 1
    def get_actions(self) -> List[int]:
        return [0, 1]

    # This method is used to check whether the agent has any moves left and if not we can use the value to then end the run
    def is_done(self) -> bool:
        return self.steps_left == 0

    # The action function is responsible for making sure that the agent is able to receive rewards as well as to end the game when the steps is 0
    def action(self, action: int) -> float:
        if self.is_done():
            raise Exception("Game is over")
        self.steps_left -= 1
        return random.random()


class Agent:
    def __init__(self):
        self.total_reward = 0.0

    def step(self, env: Environment):
        # This example the agent is lazy and does not take into account the current_obs
        current_obs = env.get_observations()
        actions = env.get_actions()
        reward = env.action(random.choice(actions))
        self.total_reward += reward


if __name__ == "__main__":
    env = Environment()
    agent = Agent()

    while not env.is_done():
        agent.step(env)
        print(f"Steps left : {env.steps_left}")
        print(f"Total reward : {agent.total_reward}")

    print(f"Total reward got {agent.total_reward:4f}")
