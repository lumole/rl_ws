import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
# from collections import deque # Not strictly needed in this revised version
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

# --- Networks (Your original networks are fine for CartPole) ---
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_nums):
        super().__init__()
        self.fl1 = nn.Linear(state_dim, 128)
        self.fl2 = nn.Linear(128, action_nums)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fl1(x)
        x = self.relu(x)
        x = self.fl2(x) # Outputs logits
        return x

class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fl1 = nn.Linear(state_dim, 128)
        self.fl2 = nn.Linear(128, 1) # Outputs a single value
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fl1(x)
        x = self.relu(x)
        x = self.fl2(x)
        return x

# --- Agent ---
class Agent:
    def __init__(self, state_dim, action_size,
                 lr_actor=3e-4, lr_critic=1e-3,
                 gamma=0.99, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5,
                 grad_clip_norm=0.5):

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.grad_clip_norm = grad_clip_norm

        self.policy_net = PolicyNetwork(state_dim, action_size)
        self.value_net = ValueNetwork(state_dim)

        self.actor_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.value_net.parameters(), lr=lr_critic)

    def get_action_value(self, state_np: np.ndarray):
        state_tensor = torch.from_numpy(state_np).float().unsqueeze(0)
        with torch.no_grad():
            logits = self.policy_net(state_tensor)
            value = self.value_net(state_tensor)

        action_dist = Categorical(logits=logits)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        # Ensure value is squeezed to a 0-dim tensor if input is single state,
        # or 1-dim if multiple states were hypothetically passed (though not in this function)
        return action.item(), log_prob, value.squeeze()


    def _calculate_gae(self, rewards_tensor: torch.Tensor, values_tensor: torch.Tensor,
                       dones_tensor: torch.Tensor, last_value_tensor: torch.Tensor):
        advantages_gae = torch.zeros_like(rewards_tensor)
        gae_accumulator = 0.0

        for t in reversed(range(len(rewards_tensor))):
            if t == len(rewards_tensor) - 1:
                v_s_t_plus_1 = last_value_tensor
            else:
                v_s_t_plus_1 = values_tensor[t + 1]

            td_error = rewards_tensor[t] + self.gamma * v_s_t_plus_1 * (1 - dones_tensor[t]) - values_tensor[t]
            gae_accumulator = td_error + self.gamma * self.gae_lambda * (1 - dones_tensor[t]) * gae_accumulator
            advantages_gae[t] = gae_accumulator

        returns_n_step = advantages_gae + values_tensor
        return advantages_gae, returns_n_step

    def update(self, trajectory_batch, last_next_state_value_tensor: torch.Tensor):
        states_np = np.array([t[0] for t in trajectory_batch])
        actions_np = np.array([t[1] for t in trajectory_batch])
        rewards_np = np.array([t[2] for t in trajectory_batch])
        dones_np = np.array([t[4] for t in trajectory_batch])
        values_rollout = torch.stack([t[6] for t in trajectory_batch]).detach() # V(s_0)...V(s_{N-1}) from rollout

        rewards_tensor = torch.from_numpy(rewards_np).float()
        dones_tensor = torch.from_numpy(dones_np.astype(np.uint8)).float()
        states_tensor = torch.from_numpy(states_np).float()
        actions_tensor = torch.from_numpy(actions_np).long()

        advantages, returns_gae = self._calculate_gae(
            rewards_tensor, values_rollout, dones_tensor, last_next_state_value_tensor.squeeze() # ensure last_next_state_value is scalar like
        )

        # --- FIX 1: Advantage Normalization ---
        # Use unbiased=False for std if numel can be 1, or handle N=1 case.
        # If advantages.numel() is 1, its mean is itself, and std (unbiased=False) is 0.
        # (X - X.mean()) / (0 + 1e-8) would be 0 / 1e-8 = 0. This is fine.
        # The problem was advantages.std() with default unbiased=True for N=1 returning NaN.
        if advantages.numel() > 0: # Check if advantages is not empty
            adv_mean = advantages.mean()
            # Use unbiased=False if you expect frequent single-element batches for std calculation
            # or handle the single element case explicitly to avoid NaN from unbiased=True.
            if advantages.numel() > 1:
                adv_std = advantages.std(unbiased=True)
            else:
                adv_std = torch.tensor(0.0, device=advantages.device) # Population std of 1 element is 0
            advantages_norm = (advantages - adv_mean) / (adv_std + 1e-8)
        else: # Should not happen if trajectory_batch is not empty
            advantages_norm = torch.zeros_like(advantages)


        logits_new = self.policy_net(states_tensor)
        action_dist_new = Categorical(logits=logits_new)
        current_log_probs = action_dist_new.log_prob(actions_tensor)
        entropy = action_dist_new.entropy().mean()

        policy_loss = -(current_log_probs * advantages_norm.detach()).mean() - self.entropy_coef * entropy

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        if self.grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip_norm)
        self.actor_optimizer.step()

        # --- FIX 2: Value Loss Shapes ---
        # Ensure current_values_pred is 1D [N] to match returns_gae [N]
        current_values_pred = self.value_net(states_tensor).view(-1) # Use view(-1)
        value_loss = F.mse_loss(current_values_pred, returns_gae.detach()) * self.value_loss_coef

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        if self.grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(self.value_net.parameters(), self.grad_clip_norm)
        self.critic_optimizer.step()

        return policy_loss.item(), value_loss.item(), entropy.item()


# --- Training Loop (largely the same, ensure trajectory_batch is not empty before update) ---
def train(env_name='CartPole-v1', num_episodes=2000, n_steps_per_update=20, log_interval=100, model_save_interval=5000):
    # global env, agent # Not needed if initialized within train

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = Agent(state_dim, action_size)

    writer = SummaryWriter(log_dir=f'runs/{env_name}_A2C_GAE_fixed_{time.strftime("%Y%m%d_%H%M%S")}')
    total_steps_collected = 0

    for episode in range(1, num_episodes + 1):
        current_state_np, info = env.reset()
        episode_reward = 0
        trajectory_batch = []
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, log_prob_tensor, value_tensor = agent.get_action_value(current_state_np)
            next_state_np, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            total_steps_collected += 1
            is_episode_done = terminated or truncated
            trajectory_batch.append((current_state_np, action, reward, next_state_np, is_episode_done, log_prob_tensor, value_tensor))
            current_state_np = next_state_np

            if len(trajectory_batch) >= n_steps_per_update or is_episode_done:
                if not trajectory_batch: # Should not happen if loop runs once
                    continue

                last_next_state_value = torch.tensor([0.0], device=log_prob_tensor.device) # Ensure device consistency
                if not is_episode_done:
                    with torch.no_grad():
                         _, _, last_next_state_value_scalar = agent.get_action_value(current_state_np)
                         last_next_state_value = last_next_state_value_scalar.unsqueeze(0) # Make it [1] like other value tensors if needed, or ensure it's scalar like

                # Ensure last_next_state_value is a scalar tensor for _calculate_gae's expectation
                # The _calculate_gae expects last_value_tensor to be a scalar tensor for broadcasting if needed.
                # agent.get_action_value returns a scalar tensor for value already.
                
                p_loss, v_loss, entropy_val = agent.update(trajectory_batch, last_next_state_value) # Pass scalar tensor
                
                writer.add_scalar('Loss/Policy', p_loss, total_steps_collected)
                writer.add_scalar('Loss/Value', v_loss, total_steps_collected)
                writer.add_scalar('Metrics/Entropy', entropy_val, total_steps_collected)
                trajectory_batch = []

        writer.add_scalar("Reward/EpisodeTotal", episode_reward, episode)
        if episode % log_interval == 0:
            print(f'Episode: {episode}, Total Reward: {episode_reward:.2f}, Total Steps: {total_steps_collected}')

        if episode % model_save_interval == 0:
            torch.save(agent.policy_net.state_dict(), f'a2c_policy_{env_name}_e{episode}.pth')
            torch.save(agent.value_net.state_dict(), f'a2c_value_{env_name}_e{episode}.pth')
            print(f"Models saved at episode {episode}")
            
    writer.close()
    env.close()
    print("Training finished.")

# --- Playback Function (Unchanged) ---
def play(env_name='CartPole-v1', policy_path=None, value_path=None, num_episodes=5):
    if policy_path is None or value_path is None:
        print("Policy or value path not provided. Cannot play.")
        return

    play_env = gym.make(env_name, render_mode='human')
    state_dim = play_env.observation_space.shape[0]
    action_size = play_env.action_space.n
    
    playback_agent = Agent(state_dim, action_size) 
    
    try:
        playback_agent.policy_net.load_state_dict(torch.load(policy_path))
        playback_agent.value_net.load_state_dict(torch.load(value_path)) 
        playback_agent.policy_net.eval()
        playback_agent.value_net.eval()
        print("Models loaded successfully for playback.")
    except FileNotFoundError:
        print(f"Error: Model file not found at {policy_path} or {value_path}")
        play_env.close()
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        play_env.close()
        return

    for episode in range(num_episodes):
        obs, info = play_env.reset()
        done = False
        truncated = False
        total_reward = 0
        while not (done or truncated):
            action, _, _ = playback_agent.get_action_value(obs)
            obs, reward, done, truncated, info = play_env.step(action)
            total_reward += reward
            # time.sleep(0.02) # Slow down rendering if needed
        print(f'Playback Episode {episode + 1}: Total Reward: {total_reward}')
    
    play_env.close()

if __name__ == '__main__':
    ENV_NAME = 'CartPole-v1'
    print(f"Starting training for {ENV_NAME}...")
    train(env_name=ENV_NAME, num_episodes=100000, n_steps_per_update=20, log_interval=50, model_save_interval=5000)
    
    print("\nStarting playback with trained models...")
    # Example:
    # policy_file = f'a2c_policy_{ENV_NAME}_e1000.pth' # Adjust episode number as needed
    # value_file = f'a2c_value_{ENV_NAME}_e1000.pth'
    # if os.path.exists(policy_file) and os.path.exists(value_file):
    #    play(env_name=ENV_NAME, policy_path=policy_file, value_path=value_file, num_episodes=3)
    # else:
    #    print(f"Model files for playback not found (e.g., {policy_file}). Run training first.")
    print("Playback example commented out. Uncomment and adjust paths after training.")