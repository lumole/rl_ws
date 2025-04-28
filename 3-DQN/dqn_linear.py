import gym_snakegame
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import time
from collections import deque


class QNetwork(nn.Module):
    def __init__(self, input_channel, width, action_nums):
        super().__init__()
        self.flatten_size = input_channel*width**2
        self.q_network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_nums)
        )
    def forward(self, x):
        q_values = self.q_network(x)
        return q_values
     
class ReplayBuffer:
    def __init__(self):
        self.buffer = deque(maxlen=10000)
        self.batch_size = 32
    def push(self, state, action, reward, nextstate, done):
        self.buffer.append((state, action, reward, nextstate, done))
    def get(self):
        minibatch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, nextstates, dones = zip(*minibatch)
        states = torch.from_numpy(np.stack(states)).float()
        actions = torch.from_numpy(np.stack(actions)).int()
        rewards = torch.from_numpy(np.stack(rewards)).float()
        nextstates = torch.from_numpy(np.stack(nextstates)).float()
        dones = torch.from_numpy(np.stack(dones)).float()

        return (states, actions, rewards, nextstates, dones)
    
    @property
    def buffer_size(self):
        return len(self.buffer)

class Agent:
    def __init__(self, channel, board_size, action_size):
        self.q = QNetwork(channel, board_size, action_size)
        self.eps = 0.1
        self.rb = ReplayBuffer()
        self.gamma = 0.99
        self.criterion = nn.MSELoss()
        self.lr = 1e-3
        self.optimizer = optim.Adam(self.q.parameters(), lr=self.lr)

    
    def get_random_action(self):
        return random.choice([0,1,2,3])
    
    def get_best_action(self, state:torch.Tensor):
        with torch.no_grad():
            # state = state.unsqueeze(0)
            q_values = self.q(state)
            return q_values.argmax(dim=1).item()

    def eps_greedy_choose_action(self, state):
        # return self.q.get_random_action()

        if random.random() < self.eps:
            return self.get_random_action()
        else:
            return self.get_best_action(state)
    
    def update(self):
        states, actions, rewards, nextstates, dones = self.rb.get()
        targets = []
        with torch.no_grad():
            ns = nextstates
            q_values = self.q(ns)
            max_q_values = q_values.max(dim = 1)[0]
            targets = rewards + self.gamma*max_q_values*(1-dones)
        
        self.q.zero_grad()

        outputs = self.q(states)
        related_outputs = outputs[torch.arange(states.shape[0]), actions.flatten()]
        # print(related_outputs, targets, related_outputs.shape, targets.shape)
        loss = self.criterion(related_outputs, targets)

        loss.backward()
        self.optimizer.step()


def train():
    obs, info = env.reset()
    for episode in range(1, 50000+1):
        done = False
        while not done:
            state = obs
            with torch.no_grad():
                # print(torch.from_numpy(state))
                action = agent.eps_greedy_choose_action(torch.from_numpy(state).float())
            
            obs, reward, terminated, truncated, info = env.step(action)
            nextstate = obs
            agent.rb.push(state, action, reward, nextstate, float(terminated))
            
            if agent.rb.buffer_size >= agent.rb.batch_size:
                agent.update()
            
            if terminated or truncated:
                obs, info = env.reset()
                done = True
        print('episod: ', episode)
        if episode%10000 == 0:
            torch.save(agent.q.state_dict(), 'model_{}.pth'.format(episode))


def play():
    board_size = 10
    myenv = gym.make(
        "gym_snakegame/SnakeGame-v0", board_size=board_size, n_channel=1, n_target=1, render_mode='human'
    )
    obs, info = myenv.reset()
    done = False
    while not done:
        state = obs

        with torch.no_grad():
            # print(torch.from_numpy(state))
            action = agent.q(torch.from_numpy(state).float()).argmax(dim=1)
        # action = myenv.action_space.sample()
        obs, reward, terminated, truncated, info = myenv.step(action)
        
        time.sleep(1)
        if terminated or truncated:
            done = True

if __name__ == '__main__':
    board_size = 10
    env = gym.make(
        "gym_snakegame/SnakeGame-v0", board_size=board_size, n_channel=1, n_target=1
    )
    agent = Agent(1, board_size, 4)
    
    train()
    
    env.close()
    # torch.save(agent.q.state_dict(), 'model.pth')
    # loaded_q_network = QNetwork(input_channel=1, width=board_size, action_nums=4)
    # # 加载之前保存的 state_dict
    # loaded_q_network.load_state_dict(torch.load(MODEL_SAVE_PATH))
    play()