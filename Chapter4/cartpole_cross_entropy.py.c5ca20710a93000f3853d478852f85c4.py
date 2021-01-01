import gym
from collections import namedtuple
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim

HIDDEN_SIZE = 128
BATCH_SIZE = 32
PERCENTILE = 70


class Net(nn.Module):
    # It takes a single observation from the environment as an input vector and outputs a number for every action we can perform
    # The output from the NN is a probability distribution over actions
    # A straightforward way to implement this would be to include softmax nonlineriality after the last layer
    # But we do not do that here as we want to increase the numerical stability of the training process
    # Rather than calculating softmax which uses exponentiation and then calculating cross-entropy loss which uses log of probabiliities we can use the nn.CrossEntropyLoss
    # This combines the two in a single more numerically stable expression
    # The downside of this method is that we need to apply softmax every time we need to get probabiliities from our NN's output
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x):
        return self.net(x)


Episode = namedtuple("Episode", field_names=["reward", "steps"])
EpisodeStep = namedtuple("EpisodeStep", field_names=["observation", "action"])


def iterate_batches(env, net, batch_size):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset()
    sm = nn.Softmax(dim=1)
    while True:
        # Our NN expects a batch of data items so we have to convert the obs from a vector of 4 numbers to a tensor of size 1 X 4
        obs_v = torch.FloatTensor([obs])
        # As there are no nonlinerialities used as the output of the NN we output raw action scores which need to be feed to the softmax function
        act_probs_v = sm(net(obs_v))
        # Both the NN and the softmax return a tensor that tracks the gradients so we need to unpack these by accessing the data field
        # Then we convert them to a numpy array
        # This array will have a two-dimensional structure as the input with the batch dimension on axis 0
        # So we need to get the first batch element to obtain a one-dimensional vector of action probabiliities
        act_probs = act_probs_v.data.numpy()[0]
        # Now we have the probability distribution of the actions we can use it to obtain the actual action for the current step by sampling this distribution
        action = np.random.choice(len(act_probs), p=act_probs)
        # Then we pass in this action to the env to get the next observation, reward for this step and whether the episode is done
        next_obs, reward, is_done, _ = env.step(action)
        # The reward is added to the total reward
        episode_reward += reward
        # The list of episode_steps is extended with the pair of (observation, action)
        # NOTE: We save the obs that was used to choose the action and not the observation that was returned as a result of the action
        step = EpisodeStep(observation=obs, action=action)
        episode_steps.append(step)
        print(f"EpisodeStep: {episode_steps}")
        if is_done:
            e = Episode(reward=episode_reward, steps=episode_steps)
            batch.append(e)
            print(f"Batch: {batch}")
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []
        # Then we assign an observation from the environment to the current observation variable
        obs = next_obs


# From the given batch of the episodes it calculates the boundary reward which is then used to filter the elite episodes from the rest
def filter_batch(batch, percentile):
    print(f"Batch in filter_batch")
    print(f"BATCH: {batch}")
    print(f"PERCENTILE: {percentile}")
    rewards = list(map(lambda s: s.reward, batch))
    print(f"Rewards: {rewards}")
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))

    train_obs = []
    train_act = []
    for reward, steps in batch:
        if reward < reward_bound:
            continue
        train_obs.extend(map(lambda step: step.observation, steps))
        train_act.extend(map(lambda step: step.action, steps))

        train_obs_v = torch.FloatTensor(train_obs)
        train_act_v = torch.LongTensor(train_act)
    return train_obs_v, train_act_v, reward_bound, reward_mean


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    env = gym.wrappers.Monitor(env, directory="mon", force=True)
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)
    writer = SummaryWriter(comment="-cartpole")

    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
        obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)
        optimizer.zero_grad()
        action_scores_v = net(obs_v)
        loss_v = objective(action_scores_v, acts_v)
        loss_v.backward()
        optimizer.step()
        print(
            "%d: loss=%.3f, reward_mean=%.1f, rw_bound=%.1f"
            % (iter_no, loss_v.item(), reward_m, reward_b)
        )
        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_bound", reward_b, iter_no)
        writer.add_scalar("reward_mean", reward_m, iter_no)
        # The CartPole environment is considered to be solved when the mean reward is greater than 195 for 100 episodes
        if reward_m > 199:
            print("Solved!")
            break
    writer.close()
