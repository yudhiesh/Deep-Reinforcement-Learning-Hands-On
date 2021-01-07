from lib import wrappers
from lib import dqn_model

import argparse
import time
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter


DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 19.0

# Gamma use for the Bellman Approximation
GAMMA = 0.99
# Batch size sampled from the replay buffer
BATCH_SIZE = 32
# Max capacity of the buffer
REPLAY_SIZE = 10000
# Learning rate for the Adam optimizera
LEARNING_RATE = 1e-4
# How frequently we sync the model weights from the training model to the target model
# Which is used to get the values of the next state in the Bellman approximation
SYNC_TARGET_FRAMES = 1000
# The count of frames we wait for before starting training to populate the replay buffer
REPLAY_START_SIZE = 10000

# Epsilon starts at 1.0 which causes all actions to be random
# Then during the first 150,000 frames, epsilon is linearly decayed to 0.01 which corresponds to the random action taken in 1% of the steps
EPSILON_DECAY_LAST_FRAME = 150000
EPSILON_START = 1.0
EPSILON_FINAL = 0.01


# The replay buffer which is used to keep the transitions obtained from the environment
# The transitions contain the tuple of the observation, action, reward, done and the next state
Experience = collections.namedtuple(
    "Experience", field_names=["state", "action", "reward", "done", "new_state"]
)


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    # Each time we do a step in the environment we push the transition into the buffer
    def append(self, experience):
        self.buffer.append(experience)

    # For training we randomly sample the batch of transitions from the replay buffer
    # This allows us to break the correlation between subsequent steps in the environment
    # Here we create a list of random indices and then repack the sampled entries into NumPy arrays for more convenient loss calculations
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, done, next_states = zip(
            *[self.buffer[idx] for idx in indices]
        )
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(done, dtype=np.uint8),
            np.array(next_states),
        )


class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = env.reset()
        self.total_reward = 0.0

    # The main function of the agent is to perform a step in environment and store its result in the buffer
    # To do this we need to select the action first

    # Disable gradient calculations which is useful for inference
    # It reduces memory usage for computations that would otherwise have requires_grad=True
    # The result of the computations will have requires_grad=False even if the inputs specify this to be true
    # torch.no_grad() impacts the autograd engine and decativates it but you will not be able to do gradient computation for backpropagation
    @torch.no_grad()
    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None

        # Here is the implementation of how we take random actions intially
        # When the value of epsilon is high we take random actions
        # When the value of epsilon is low we take actions based on the NN model
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            # act_v is a float and we round it down to an int
            # this int is the action that the agent should take
            action = int(act_v.item())

        # now the action has been picked either randomly or from the q_vals
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        # we pass the action to the environment to get the next observation and reward, store the data ion the experience buffer and then handle the end of the episode
        # this returns the total cumulative reward if we are at the end of the episode
        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward

    def calc_loss(batch, net, tgt_net, device="cpu"):
        # The batch is passed in as a tuple of arrays
        # net is the NN we are training on
        # tgt_net is the target NN which is periodically synced with the training NN
        states, actions, rewards, dones, next_states = batch
        # net is used to calculate values for the next states and this calculation shouldn't affect gradients

        # here we wrap the NumPy arrays with batch data in PyTorch tensors and copy them to GPU in the CUDA device was specified in arguments
        states_v = torch.tensor(np.array(states, copy=False)).to(device)
        next_states_v = torch.tensor(np.array(next_states, copy=False)).to(device)
        actions_v = torch.tensor(actions).to(device)
        rewards_v = torch.tensor(rewards).to(device)
        done_mask = torch.BoolTensor(dones).to(device)

        # The output of the model is a matrix with the row being the batch and the column being the action
        # actions_v is a vector of the actions taken
        # We want to select the actions taken based on the index in the output of the model
        # actions_v is turned into a 2D matrix in order to index the output of the model
        # these actions are then squeezed into a vector that is the state_action_values
        state_action_values = (
            net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
        )
        # the result of gather() applied to tensors is a differentiable oeration that will keep all the gradients with respect to the final loss value
        with torch.no_grad():
            # We apply the target network to the next state observations and calculate the max Q-value along the action dimension which is 1
            # max() returns the max values as well as the indices
            # here we take only the first value
            next_state_values = tgt_net(next_states_v).max(1)[0]
            # if the transition in the batch is from the last step in the episode, then our value of the action doesn't have a discounted reward of the next state
            # as there is no next state from which to gather the reward
            next_state_values[done_mask] = 0.0
            # we detach the value from its computation graph to prevent gradients from flowing into the NN used to calculate Q approximation for the next states
            # without this the backpropagation for the loss will start to affect both predictions for the current state and the next state
            # However we do not want to touch predictions for the next state, as they are used in the Bellman equation to calculate reference Q-values
            # here it basically return the tensor without connection to its calculation history
            next_state_values = next_state_values.detach()
        # calculate the Bellman approximation value
        expected_state_action_values = next_state_values * GAMMA + rewards_v

        # calculate the MSE loss for the state_action_values compared to the expected_state_action_values

        return nn.MSELoss()(state_action_values, expected_state_action_values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda", default=False, action="store_true", help="Enable cuda"
    )
    parser.add_argument(
        "--env",
        default=DEFAULT_ENV_NAME,
        help="Name of the environment, default=" + DEFAULT_ENV_NAME,
    )
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    env = wrappers.make_env(args.env)

    # Neural Network
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    # Target Network
    tgt_net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    writer = SummaryWriter(comment="-" + args.env)
    print(net)

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_m_reward = None

    while True:
        frame_idx += 1
        # We count the number of iterations completed and decrease epsilon according to our schedule.
        # Epsilon will drop linearly  during the given number of frames and then be kept at the same level of EPSILON_FINAL
        epsilon = max(
            EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME
        )

        reward = agent.play_step(net, epsilon, device=device)
        if reward is not None:
            total_rewards.append(reward)
            # Speed as a count of frames preocessed per second
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            # Mean reward for the last 100 episodes
            m_reward = np.mean(total_rewards[-100:])
            print(
                "%d: done %d games, reward %.3f, "
                "eps %.2f, speed %.2f f/s"
                % (frame_idx, len(total_rewards), m_reward, epsilon, speed)
            )
            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", m_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)
            if best_m_reward is None or best_m_reward < m_reward:
                torch.save(net.state_dict(), args.env + "-best_%.0f.dat" % m_reward)
                if best_m_reward is not None:
                    print(
                        "Best reward updated %.3f -> %.3f" % (best_m_reward, m_reward)
                    )
                best_m_reward = m_reward
            # If the reward exceeds MEAN_REWARD_BOUND which is 19 which means we win 19 out of 21 games then stop the training
            if m_reward > MEAN_REWARD_BOUND:
                print("Solved in %d frames!" % frame_idx)
                break
        # Check whether the buffer is large enough for training
        # It will wait for the data to be at least 10k transitions
        if len(buffer) < REPLAY_START_SIZE:
            continue

        # Here we sync the parameters from the main NN to the target network
        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())

        # Zero gradients
        optimizer.zero_grad()
        # Sample data batches from the experience replay buffer
        batch = buffer.sample(BATCH_SIZE)
        # calculate the loss
        loss_t = calc_loss(batch, net, tgt_net, device=device)
        loss_t.backward()
        # perform optimization to minimize the loss
        optimizer.step()
    writer.close()
