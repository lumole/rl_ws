import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_nums):
        super().__init__()
        self.fl1 = nn.Linear(state_dim, 128)
        self.fl_mu = nn.Linear(128, action_nums)
        self.fl_sigma = nn.Linear(128, action_nums)
        self.relu = nn.ReLU()
        # self.q_network = nn.Sequential(
        #     nn.Linear(state_dim, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, action_nums*2),
        #     # nn.ReLU(),
        #     # nn.Linear(64, action_nums)
        # )

    def forward(self, x):
        x = self.fl1(x)
        x = self.relu(x)
        mu = self.fl_mu(x)
        sigma = self.fl_sigma(x)
        sigma = F.softplus(sigma)+1e-5
        return mu, sigma #使用tuple有好处，就是以后可以直接提取，因为并不会像之前一样需要输出整个参与运算，在REINFORCE中的loss是一个比较特殊的loss。

class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fl1 = nn.Linear(state_dim, 128)
        self.fl2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fl1(x)
        x = self.relu(x)
        x = self.fl2(x)
        return x

class ReplayBuffer:
    def __init__(self, max_size=10000, batch_size=64, gamma = 0.99):
        self.buffer = deque(maxlen=max_size)
        self.tmp_buffer = []
        self.batch_size = batch_size
        self.gamma = gamma

    def push(self, state, action, reward, nextstate, done):
        # CartPole action is an integer, no need for np.array([action])
        done = float(done)
        self.tmp_buffer.append((state, action, reward, nextstate, done))
        if done:
            states, actions, rewards, nextstates, dones = zip(*self.tmp_buffer)

            returns = [0]*len(rewards)
            ret_tmp = 0
            for i in range(len(rewards))[::-1] :
                ret_tmp = rewards[i] + self.gamma * ret_tmp
                returns[i] = ret_tmp

            states = torch.from_numpy(np.stack(states)).float()
            actions = torch.from_numpy(np.stack(actions)).float()
            returns = torch.from_numpy(np.stack(returns)).float()
            nextstates = torch.from_numpy(np.stack(nextstates)).float()
            dones = torch.from_numpy(np.stack(dones)).float()
            self.buffer.append((states, actions, returns, nextstates, dones))
            self.tmp_buffer = []

    def get(self):
        '''
            获取到了是batch_size个元组，每一个元组里都有一个轨迹的(states, actions, rewards, nextstates, dones)，
            并且每条轨迹的states的维度还各不相同，一般是(T, 3)
        '''
        minibatch = random.sample(self.buffer, self.batch_size)
        return minibatch

    @property
    def buffer_size(self):
        return len(self.buffer)

class Agent:
    def __init__(self, state_dim, action_size):
        self.action_size = action_size # Store action size
        self.gamma = 0.99
        self.lr = 2e-4
        self.p = PolicyNetwork(state_dim, action_size)
        self.v = ValueNetwork(state_dim)
        self.rb = ReplayBuffer(gamma = self.gamma)
        self.p_optimizer = optim.Adam(self.p.parameters(), lr=self.lr)
        self.v_optimizer = optim.Adam(self.v.parameters(), lr=self.lr)
        self.v_criterion = torch.nn.MSELoss()

    def get_action(self, state: torch.Tensor):
        with torch.no_grad():
            state = torch.from_numpy(state)
            state = state.unsqueeze(0).float()
            mu, sigma =  self.p(state)
            action_dist = torch.distributions.Normal(mu, sigma)
            action = action_dist.sample().squeeze(0).numpy()

            return action


    def update(self):
        samples = self.rb.get()
        total_loss = 0
        v_total_loss = 0
        for each_traj in samples:
            states, actions, returns, nextstates, dones = each_traj
            mus, sigmas = self.p(states)
            log_probs = torch.distributions.Normal(mus, sigmas).log_prob(actions)
            log_probs = torch.flatten(log_probs)

            returns = torch.flatten(returns)
            # returns_mean = torch.mean(returns)
            # returns_std = torch.std(returns)
            # returns = (returns-returns_mean)/(returns_std+1e-8)

            state_value = self.v(states)
            advantage_value = returns - state_value.detach().flatten()
            loss_policy = -(log_probs*advantage_value).mean()
            total_loss += loss_policy

            v_loss = self.v_criterion(state_value, returns.unsqueeze(1))
            v_total_loss += v_loss
        total_loss = total_loss/len(samples)
        v_total_loss = v_total_loss/len(samples)

        self.p_optimizer.zero_grad()
        total_loss.backward()
        self.p_optimizer.step()

        self.v_optimizer.zero_grad()
        v_total_loss.backward()
        self.v_optimizer.step()


def train():
    # Use the global environment created in __main__
    global env, agent
    writer = SummaryWriter(log_dir = 'runs/pendulum'+time.strftime("%Y%m%d_%H%M%S"))
    obs, info = env.reset()
    global_step = 0 # Track total steps for epsilon decay and target update

    for episode in range(1, 100000 + 1):
        done = False
        total_reward = 0
        # Reset environment for the new episode
        obs, info = env.reset()

        while not done:
            state = obs # Current state

            # Agent chooses action using epsilon-greedy
            action = agent.get_action(state)

            # Take the action in the environment
            obs, reward, terminated, truncated, info = env.step(action)
            nextstate = obs # Next state
            total_reward += reward

            # Check if the episode is truly done (terminated or truncated)
            is_done = terminated or truncated

            # Store the transition in the replay buffer
            # Note: action is an integer here, no need to wrap in np.array
            agent.rb.push(state, action, reward, nextstate, is_done) 

            # If buffer is full enough, perform a training step

            # print(agent.rb.buffer_size,agent.rb.batch_size)
            # Increment global step
            global_step += 1

            # End episode if done
            if is_done:
                done = True
        if agent.rb.buffer_size >= agent.rb.batch_size:
            agent.update()  # Pass global step to update
        writer.add_scalar("Reward_episode", total_reward, episode)
        if episode % 100 == 0:
            # Print current epsilon along with reward
            print(f'Episode: {episode}, Total reward: {total_reward}, global_step: {global_step}')

        # Optional: Save model periodically
        if episode % 2000 == 0:
            torch.save(agent.p.state_dict(), f'model_pendulum_{episode}.pth')
    writer.close()

def play():
    # Create a new environment instance with render_mode='human' for visualization
    myenv = gym.make('Pendulum-v1', render_mode='human')
    obs, info = myenv.reset()
    done = False
    total_reward = 0

    while not done:
        state = obs

        action = agent.get_action(state)

        obs, reward, terminated, truncated, info = myenv.step(action)
        total_reward += reward

        # Add a small delay to see the rendering
        # time.sleep(1) # Use a smaller sleep for smoother playback

        # Check if episode ended
        if terminated or truncated:
            done = True

    print(f'Total reward during play: {total_reward}')
    myenv.close() # Close the rendering window

if __name__ == '__main__':
    # Initialize the environment and agent
    env = gym.make('Pendulum-v1')
    state_dim = env.observation_space.shape[0] 
    action_size = env.action_space.shape[0]  

    agent = Agent(state_dim, action_size)

    # Start training
    print("Starting training...")
    train()
    print("Training finished.")

    # Close the training environment
    env.close()

    # Start playing with the trained agent
    agent.p.load_state_dict(torch.load('model_pendulum_52000.pth'))
    print("\nStarting playback...")
    play()
    print("Playback finished.")