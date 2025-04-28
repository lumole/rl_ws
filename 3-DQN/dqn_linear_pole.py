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
    def __init__(self, state_dim, action_nums):
        super().__init__()
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_nums),
            # nn.ReLU(),
            # nn.Linear(64, action_nums)
        )

    def forward(self, x):
        q_values = self.q_network(x)
        return q_values

class ReplayBuffer:
    def __init__(self, max_size=1000, batch_size=64):
        self.buffer = deque(maxlen=max_size)
        self.batch_size = batch_size

    def push(self, state, action, reward, nextstate, done):
        # CartPole action is an integer, no need for np.array([action])
        self.buffer.append((state, action, reward, nextstate, done))

    def get(self):
        minibatch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, nextstates, dones = zip(*minibatch)

        # Convert to tensors
        states = torch.from_numpy(np.stack(states)).float()
        # Actions should be LongTensor for indexing
        actions = torch.from_numpy(np.stack(actions)).long()
        rewards = torch.from_numpy(np.stack(rewards)).float()
        nextstates = torch.from_numpy(np.stack(nextstates)).float()
        # Dones should be float for multiplication
        dones = torch.from_numpy(np.stack(dones)).float()

        return (states, actions, rewards, nextstates, dones)

    @property
    def buffer_size(self):
        return len(self.buffer)

class Agent:
    def __init__(self, state_dim, action_size):
        self.action_size = action_size # Store action size
        self.q = QNetwork(state_dim, action_size)
        self.target_q = QNetwork(state_dim, action_size) # Target network
        self.target_q.load_state_dict(self.q.state_dict()) # Initialize target network

        # Epsilon decay parameters
        # self.eps_start = 1.0
        # self.eps_end = 0.01
        # self.eps_decay_steps = 10000 # Number of steps for epsilon to decay
        self.eps_start = 1
        self.eps_end = 0.01
        self.eps_step = (self.eps_start-self.eps_end)/1e5
        self.eps = self.eps_start

        self.rb = ReplayBuffer()
        self.gamma = 0.99
        self.criterion = nn.MSELoss()
        self.lr = 1e-3
        self.optimizer = optim.Adam(self.q.parameters(), lr=self.lr)
        self.target_update_frequency = 500 # Update target network every N steps

    def get_random_action(self):
        # Corrected: random action within the valid range [0, action_size-1]
        return random.randrange(self.action_size)

    def get_best_action(self, state: torch.Tensor):
        with torch.no_grad():
            state = state.unsqueeze(0).float()
            q_values = self.q(state)
            return q_values.argmax(dim=1).item() # Get action index from batch


    def eps_greedy_choose_action(self, state):
        self.eps = max(self.eps_end, self.eps-self.eps_step)
        if random.random() < self.eps:
            return self.get_random_action()
        else:
            return self.get_best_action(state)

    def update(self, global_step):
        samples = self.rb.get()

        states, actions, rewards, nextstates, dones = samples

        # Calculate target Q values using the target network
        with torch.no_grad():
            max_next_q_values = self.target_q(nextstates).max(dim=1)[0]
            targets = rewards + self.gamma * max_next_q_values * (1 - dones)

        outputs = self.q(states)
        related_outputs = outputs.gather(1, actions.unsqueeze(1)).squeeze(1) # Use gather for indexing


        # Calculate the loss (MSE between predicted Q and target Q)
        loss = self.criterion(related_outputs, targets.detach()) # Detach targets to prevent backprop through target network

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        # Optional: Clip gradients to prevent exploding gradients
        # torch.nn.utils.clip_grad_norm_(self.q.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update target network periodically
        if global_step % self.target_update_frequency == 0:
            self.target_q.load_state_dict(self.q.state_dict())
            # print("Target network updated") # Optional: print confirmation

def train():
    # Use the global environment created in __main__
    global env, agent
    writer = SummaryWriter(log_dir = 'runs/cartpole'+time.strftime("%Y%m%d_%H%M%S"))
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
            action = agent.eps_greedy_choose_action(torch.from_numpy(state).float())

            # Take the action in the environment
            obs, reward, terminated, truncated, info = env.step(action)

            nextstate = obs # Next state
            total_reward += reward

            # Check if the episode is truly done (terminated or truncated)
            is_done = terminated or truncated

            # Store the transition in the replay buffer
            # Note: action is an integer here, no need to wrap in np.array
            agent.rb.push(state, action, reward, nextstate, float(terminated)) # Use 'terminated' for done flag in replay buffer for Bellman update

            # If buffer is full enough, perform a training step
            if agent.rb.buffer_size >= agent.rb.batch_size:
                 agent.update(global_step) # Pass global step to update

            # Increment global step
            global_step += 1

            # End episode if done
            if is_done:
                done = True
        writer.add_scalar("Reward_episode", total_reward, episode)
        if episode % 100 == 0:
            # Print current epsilon along with reward
            print(f'Episode: {episode}, Total reward: {total_reward}, Epsilon: {agent.eps:.4f}, global_step: {global_step}')

        # Optional: Save model periodically
        if episode % 500 == 0:
            torch.save(agent.q.state_dict(), f'model_cartpole_{episode}.pth')
    writer.close()

def play():
    # Create a new environment instance with render_mode='human' for visualization
    myenv = gym.make('CartPole-v1', render_mode='human')
    obs, info = myenv.reset()
    done = False
    total_reward = 0

    while not done:
        state = obs

        action = agent.get_best_action(torch.from_numpy(state).float())

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
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]  # CartPole状态空间维度为4
    action_size = env.action_space.n  # CartPole动作空间维度为2 (0 or 1)

    agent = Agent(state_dim, action_size)

    # Start training
    print("Starting training...")
    train()
    print("Training finished.")

    # Close the training environment
    env.close()

    # Start playing with the trained agent
    # agent.q.load_state_dict(torch.load('model_cartpole_3000.pth'))
    print("\nStarting playback...")
    play()
    print("Playback finished.")