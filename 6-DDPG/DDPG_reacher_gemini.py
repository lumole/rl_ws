import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
import copy
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# --- Default Hyperparameters (can be overridden by train_agent arguments) ---
DEFAULT_LR_ACTOR = 1e-4
DEFAULT_LR_CRITIC = 3e-4
DEFAULT_GAMMA = 0.99
DEFAULT_TAU = 0.005
DEFAULT_BUFFER_SIZE = int(1e6)
DEFAULT_BATCH_SIZE = 256
DEFAULT_NOISE_STD = 0.1
DEFAULT_HIDDEN_DIM = 256

# --- Default Directories ---
DEFAULT_MODELS_DIR = "saved_models"
DEFAULT_LOG_DIR = "runs"


# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size, device): # Pass device for tensor creation
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)).to(device),
            torch.FloatTensor(np.array(actions)).to(device),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(device),
            torch.FloatTensor(np.array(next_states)).to(device),
            torch.FloatTensor(np.array(dones)).unsqueeze(1).to(device),
        )

    def __len__(self):
        return len(self.buffer)

# --- Actor Network ---
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=DEFAULT_HIDDEN_DIM):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.layer_1(state))
        x = F.relu(self.layer_2(x))
        x = torch.tanh(self.layer_3(x)) * self.max_action
        return x

# --- Critic Network ---
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=DEFAULT_HIDDEN_DIM):
        super(Critic, self).__init__()
        self.layer_1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q = F.relu(self.layer_1(sa))
        q = F.relu(self.layer_2(q))
        q = self.layer_3(q)
        return q

# --- DDPG Agent ---
class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action,
                 lr_actor=DEFAULT_LR_ACTOR, lr_critic=DEFAULT_LR_CRITIC,
                 gamma=DEFAULT_GAMMA, tau=DEFAULT_TAU,
                 buffer_size=DEFAULT_BUFFER_SIZE, batch_size=DEFAULT_BATCH_SIZE,
                 noise_std=DEFAULT_NOISE_STD, hidden_dim=DEFAULT_HIDDEN_DIM,
                 device=torch.device("cpu")):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.tau = tau
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.noise_std = noise_std
        self.hidden_dim = hidden_dim # Using one hidden_dim for simplicity
        self.device = device

        self.main_actor = Actor(state_dim, action_dim, max_action, hidden_dim=self.hidden_dim).to(self.device)
        self.main_critic = Critic(state_dim, action_dim, hidden_dim=self.hidden_dim).to(self.device)
        self.target_actor = copy.deepcopy(self.main_actor).to(self.device)
        self.target_critic = copy.deepcopy(self.main_critic).to(self.device)

        for p in self.target_actor.parameters(): p.requires_grad = False
        for p in self.target_critic.parameters(): p.requires_grad = False

        self.main_actor_optimizer = optim.Adam(self.main_actor.parameters(), lr=self.lr_actor)
        self.main_critic_optimizer = optim.Adam(self.main_critic.parameters(), lr=self.lr_critic)
        self.v_criterion = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(self.buffer_size)

    def select_action(self, state, add_noise=True):
        state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        self.main_actor.eval()
        with torch.no_grad():
            action = self.main_actor(state_tensor).cpu().data.numpy().flatten()
        self.main_actor.train()

        if add_noise:
            noise = np.random.normal(0, self.noise_std * self.max_action, size=self.action_dim)
            action = (action + noise).clip(-self.max_action, self.max_action)
        return action

    def update_networks(self):
        if len(self.replay_buffer) < self.batch_size:
            return None, None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size, self.device)

        with torch.no_grad():
            target_actions = self.target_actor(next_states)
            target_Q = self.target_critic(next_states, target_actions)
            y = rewards + self.gamma * (1 - dones) * target_Q
        current_Q = self.main_critic(states, actions)
        critic_loss = self.v_criterion(current_Q, y)

        self.main_critic_optimizer.zero_grad()
        critic_loss.backward()
        self.main_critic_optimizer.step()

        actor_actions = self.main_actor(states)
        actor_loss = -self.main_critic(states, actor_actions).mean()

        self.main_actor_optimizer.zero_grad()
        actor_loss.backward()
        self.main_actor_optimizer.step()

        for target_param, main_param in zip(self.target_actor.parameters(), self.main_actor.parameters()):
            target_param.data.copy_(self.tau * main_param.data + (1.0 - self.tau) * target_param.data)
        for target_param, main_param in zip(self.target_critic.parameters(), self.main_critic.parameters()):
            target_param.data.copy_(self.tau * main_param.data + (1.0 - self.tau) * target_param.data)

        return critic_loss.item(), actor_loss.item()

    def save_models(self, directory, prefix="ddpg"):
        os.makedirs(directory, exist_ok=True)
        actor_path = os.path.join(directory, f"{prefix}_actor.pth")
        critic_path = os.path.join(directory, f"{prefix}_critic.pth")
        torch.save(self.main_actor.state_dict(), actor_path)
        torch.save(self.main_critic.state_dict(), critic_path)
        print(f"Models saved to {actor_path} and {critic_path}")

    def load_models(self, actor_path, critic_path):
        self.main_actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        self.main_critic.load_state_dict(torch.load(critic_path, map_location=self.device))
        self.target_actor = copy.deepcopy(self.main_actor)
        self.target_critic = copy.deepcopy(self.main_critic)
        print(f"Models loaded from {actor_path} and {critic_path}")

# --- Train Function ---
def train_agent(
    env_name="Reacher-v4",
    num_episodes=1000,
    max_steps_per_episode=200,
    lr_actor_hp=DEFAULT_LR_ACTOR,
    lr_critic_hp=DEFAULT_LR_CRITIC,
    gamma_hp=DEFAULT_GAMMA,
    tau_hp=DEFAULT_TAU,
    buffer_size_hp=DEFAULT_BUFFER_SIZE,
    batch_size_hp=DEFAULT_BATCH_SIZE,
    noise_std_hp=DEFAULT_NOISE_STD,
    hidden_dim_hp=DEFAULT_HIDDEN_DIM,
    models_dir=DEFAULT_MODELS_DIR,
    log_dir=DEFAULT_LOG_DIR,
    save_interval=100,
    min_buffer_factor=5, # Factor to multiply batch_size for min samples before training
    device_str="auto"
):
    if device_str == "auto":
        current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        current_device = torch.device(device_str)

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    print(f"\n--- Starting Training ---")
    print(f"Environment: {env_name}, State Dim: {state_dim}, Action Dim: {action_dim}, Max Action: {max_action}")
    print(f"Device: {current_device}")
    print(f"Hyperparameters: LR_Actor={lr_actor_hp}, LR_Critic={lr_critic_hp}, BatchSize={batch_size_hp}, HiddenDim={hidden_dim_hp}")

    agent = DDPGAgent(
        state_dim, action_dim, max_action,
        lr_actor=lr_actor_hp, lr_critic=lr_critic_hp, gamma=gamma_hp, tau=tau_hp,
        buffer_size=buffer_size_hp, batch_size=batch_size_hp,
        noise_std=noise_std_hp, hidden_dim=hidden_dim_hp, device=current_device
    )

    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_subdir = f"{env_name}_DDPG_{current_time}"
    writer = SummaryWriter(os.path.join(log_dir, log_subdir))
    print(f"TensorBoard logs will be saved to: {os.path.join(log_dir, log_subdir)}")

    total_steps = 0
    episode_rewards_history = []
    min_buffer_needed = batch_size_hp * min_buffer_factor
    print(f"Training updates will start after {min_buffer_needed} samples in replay buffer.")

    last_saved_actor_path = ""

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        for t in range(max_steps_per_episode):
            action = agent.select_action(state, add_noise=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.replay_buffer.add(state, action, reward, next_state, float(done))
            state = next_state
            episode_reward += reward
            total_steps += 1

            if len(agent.replay_buffer) >= min_buffer_needed:
                critic_loss_val, actor_loss_val = agent.update_networks()
                if critic_loss_val is not None and actor_loss_val is not None:
                    writer.add_scalar('Loss/Critic_Loss', critic_loss_val, total_steps)
                    writer.add_scalar('Loss/Actor_Loss', actor_loss_val, total_steps)
            
            if done:
                break
        
        episode_rewards_history.append(episode_reward)
        avg_reward = np.mean(episode_rewards_history[-100:])
        
        writer.add_scalar('Reward/Episode_Reward', episode_reward, episode + 1)
        writer.add_scalar('Reward/Avg_Reward_Last_100', avg_reward, episode + 1)
        
        print(f"Episode: {episode+1}/{num_episodes}, Steps: {t+1}, Total Steps: {total_steps}, Reward: {episode_reward:.2f}, Avg Reward (100ep): {avg_reward:.2f}")

        if (episode + 1) % save_interval == 0:
            model_prefix = f"{env_name}_ddpg_episode_{episode+1}"
            agent.save_models(directory=models_dir, prefix=model_prefix)
            last_saved_actor_path = os.path.join(models_dir, f"{model_prefix}_actor.pth")

    # Save the final model if it wasn't saved in the last interval step
    if num_episodes % save_interval != 0:
        model_prefix = f"{env_name}_ddpg_episode_{num_episodes}"
        agent.save_models(directory=models_dir, prefix=model_prefix)
        last_saved_actor_path = os.path.join(models_dir, f"{model_prefix}_actor.pth")
    elif not last_saved_actor_path: # Handle case where num_episodes < save_interval
        model_prefix = f"{env_name}_ddpg_episode_{num_episodes}"
        agent.save_models(directory=models_dir, prefix=model_prefix)
        last_saved_actor_path = os.path.join(models_dir, f"{model_prefix}_actor.pth")


    env.close()
    writer.close()
    print("--- Training finished. ---")
    return last_saved_actor_path

# --- Play Function ---
def play_trained_agent(env_id, actor_model_path, num_episodes=5, hidden_dim_hp=DEFAULT_HIDDEN_DIM, device_str="auto"):
    print(f"\n--- Playing Trained Agent ---")
    print(f"Environment: {env_id}, Model: {actor_model_path}")

    if device_str == "auto":
        current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        current_device = torch.device(device_str)
    print(f"Using device: {current_device}")

    try:
        env_to_play = gym.make(env_id, render_mode="human")
    except Exception as e:
        print(f"Error creating environment {env_id} with render_mode='human': {e}")
        print("Try playing without rendering, or ensure display is configured.")
        try:
            env_to_play = gym.make(env_id) # Fallback without rendering
            print("Fallback: Playing without rendering.")
        except Exception as e2:
            print(f"Error creating environment {env_id} even without rendering: {e2}")
            return


    state_dim_play = env_to_play.observation_space.shape[0]
    action_dim_play = env_to_play.action_space.shape[0]
    max_action_play = float(env_to_play.action_space.high[0])

    actor_net = Actor(state_dim_play, action_dim_play, max_action_play, hidden_dim=hidden_dim_hp).to(current_device)
    
    if not os.path.exists(actor_model_path):
        print(f"Error: Actor model path not found: {actor_model_path}")
        env_to_play.close()
        return
        
    try:
        actor_net.load_state_dict(torch.load(actor_model_path, map_location=current_device))
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        env_to_play.close()
        return

    actor_net.eval()

    for episode in range(num_episodes):
        state, _ = env_to_play.reset()
        ep_reward = 0
        terminated, truncated = False, False
        max_play_steps = env_to_play.spec.max_episode_steps if env_to_play.spec else 200 # Use env spec if available

        for step_count in range(max_play_steps):
            state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(current_device)
            with torch.no_grad():
                action = actor_net(state_tensor).cpu().data.numpy().flatten()
            
            next_state, reward, terminated, truncated, _ = env_to_play.step(action)
            state = next_state
            ep_reward += reward
            if terminated or truncated:
                break
        print(f"Play Episode {episode + 1}/{num_episodes}: Reward = {ep_reward:.2f}")
    env_to_play.close()
    print("--- Playback finished. ---")


# --- Main Execution ---
if __name__ == "__main__":
    ENV_ID_MAIN = "Reacher-v4" # "Pendulum-v1" is another common one
    NUM_TRAIN_EPISODES = 200  # Increase for real training (e.g., 1000+)
    MAX_TRAIN_STEPS = 200    # Max steps per episode during training

    # You can override default hyperparameters here by passing them to train_agent
    # For example: lr_actor_hp=5e-5
    final_actor_model_path = train_agent(
        env_name=ENV_ID_MAIN,
        num_episodes=NUM_TRAIN_EPISODES,
        max_steps_per_episode=MAX_TRAIN_STEPS,
        # Example of overriding a few defaults:
        # batch_size_hp=128,
        # hidden_dim_hp=128,
        save_interval=50 # Save more frequently for shorter test runs
    )

    if final_actor_model_path and os.path.exists(final_actor_model_path):
        play_trained_agent(
            env_id=ENV_ID_MAIN,
            actor_model_path=final_actor_model_path,
            num_episodes=5,
            hidden_dim_hp=DEFAULT_HIDDEN_DIM # Ensure this matches the hidden_dim used during training
                                           # Or better, save/load hidden_dim with the model config
        )
    else:
        print(f"No valid model path returned from training, or model not found. Skipping playback.")