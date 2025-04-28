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
from torch.utils.tensorboard import SummaryWriter

class QNetwork(nn.Module):
    def __init__(self, input_channel, width, action_nums):
        super().__init__()
        self.flatten_size = input_channel*width**2
        self.fc1 = nn.Conv2d(input_channel, 32, kernel_size=3, padding=1)
        self.fc2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fl1 = nn.Linear(64*width*width, 128)
        self.fl2 = nn.Linear(128, action_nums)
        # self.q_network = nn.Sequential(
        #     nn.Conv2d(input_channel, 32, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Flatten(),
        #     nn.Linear(64*width*width, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, action_nums),
        # )
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.flatten(1)
        x = F.relu(self.fl1(x))
        x = self.fl2(x)
        return x
        # q_values = self.q_network(x)
        # return q_values
     
class ReplayBuffer:
    def __init__(self):
        self.buffer = deque(maxlen=100000)
        self.batch_size = 64
    def push(self, state, action, reward, nextstate, done):
        self.buffer.append((state, action, reward, nextstate, done))
    def get(self):
        minibatch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, nextstates, dones = zip(*minibatch)
        states = torch.from_numpy(np.stack(states)).float()
        actions = torch.from_numpy(np.stack(actions)).long()
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
        self.eps_start = 1
        self.eps_end = 0.01
        self.eps = self.eps_start
        self.eps_decacy_step = (self.eps_start-self.eps_end)/1e6
        self.rb = ReplayBuffer()
        self.gamma = 0.99
        self.criterion = nn.MSELoss()
        self.lr = 1e-3
        self.optimizer = optim.Adam(self.q.parameters(), lr=self.lr)
        self.target_q = QNetwork(channel, board_size, action_size)
        self.target_q.load_state_dict(self.q.state_dict())
        self.target_q_update_timestep = 2000

    def get_random_action(self):
        return random.choice([0,1,2,3])
    
    def get_best_action(self, state:torch.Tensor):
        with torch.no_grad():
            state = state.unsqueeze(0)
            state = state.float()
            q_values = self.q(state)
            return q_values.argmax(dim=1).item()

    def eps_greedy_choose_action(self, state):
        # return self.q.get_random_action()
        self.eps = max(self.eps_end, self.eps-self.eps_decacy_step)
        if random.random() < self.eps:
            return self.get_random_action()
        else:
            return self.get_best_action(state)
    
    def update(self, global_step):
        states, actions, rewards, nextstates, dones = self.rb.get()
        targets = []
        with torch.no_grad():
            ns = nextstates
            q_values = self.target_q(ns)
            max_q_values = q_values.max(dim = 1)[0]
            targets = rewards + self.gamma*max_q_values*(1-dones)
        
        self.q.zero_grad()

        outputs = self.q(states)
        related_outputs = outputs[torch.arange(states.shape[0]), actions.flatten()]
        # print(related_outputs, targets, related_outputs.shape, targets.shape)
        loss = self.criterion(related_outputs, targets)

        loss.backward()
        self.optimizer.step()

        if global_step%self.target_q_update_timestep == 0:
            self.target_q.load_state_dict(self.q.state_dict())


def train():
    obs, info = env.reset()
    writer = SummaryWriter(log_dir='runs/'+'snake_'+time.strftime("%Y%m%d_%H%M%S"))
    global_step = 0
    for episode in range(1, 50000+1):
        done = False
        total_reward = 0
        while not done:
            global_step += 1
            state = obs
            with torch.no_grad():
                # print(torch.from_numpy(state))
                state_input = torch.from_numpy(state)
                action = agent.eps_greedy_choose_action(state_input)
            
            obs, reward, terminated, truncated, info = env.step(action)
            nextstate = obs
            agent.rb.push(state, action, reward, nextstate, float(terminated))
            total_reward+=reward
            if agent.rb.buffer_size >= agent.rb.batch_size:
                agent.update(global_step)
            
            if terminated or truncated:
                obs, info = env.reset()
                done = True
        writer.add_scalar('snake/reward', total_reward, episode)
        if episode%500 == 0:
            torch.save(agent.q.state_dict(), '3-DQN/models/snake_model_{}.pth'.format(episode))
        if episode%100==0:
            print(f'Episode: {episode}, Reward: {total_reward}, global_step: {global_step}, Epsilon: {agent.eps:.4f}')
    writer.close()

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
            state_input = torch.from_numpy(state)
            action = agent.get_best_action(state_input)
            print(state_input, action)
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
    # agent.q.load_state_dict(torch.load('model_30000.pth'))
    play()