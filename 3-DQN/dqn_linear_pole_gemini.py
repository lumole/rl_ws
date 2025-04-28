import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import typing as T # Import typing module for type hints
import dataclasses # Import dataclasses for configuration

# Use dataclass for cleaner hyperparameter management
@dataclasses.dataclass
class DQNConfig:
    env_id: str = 'CartPole-v1'
    seed: int = 42
    total_timesteps: int = 1000000 # Max training steps, might finish earlier based on episodes
    num_episodes: int = 100000    # Max training episodes
    buffer_size: int = 10000      # Replay buffer size
    batch_size: int = 128         # Training batch size
    gamma: float = 0.99           # Discount factor
    learning_rate: float = 1e-3   # Learning rate for Adam optimizer
    target_update_frequency: int = 500 # How often to update the target network
    train_frequency: int = 1      # How often to train (update) the Q network (in steps)
    epsilon_start: float = 1.0    # Starting epsilon for epsilon-greedy
    epsilon_end: float = 0.01     # Ending epsilon for epsilon-greedy
    epsilon_decay_steps: int = 50000 # Steps over which epsilon decays
    # Removed epsilon_step calculation here, will calculate dynamically
    log_frequency: int = 100      # Log training progress every N episodes
    save_frequency: int = 500     # Save model every N episodes
    render_play: bool = True      # Whether to render during play
    # Device will be determined at runtime
    device: T.Optional[torch.device] = None

class QNetwork(nn.Module):
    """
    DQN's Q-Value Network.
    """
    def __init__(self, state_dim: int, action_nums: int):
        super().__init__()
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_nums) # Output layer directly predicts Q-values for each action
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        Args:
            x: Input tensor representing state(s).
        Returns:
            Tensor of Q-values for each action in the given state(s).
        """
        q_values = self.q_network(x)
        return q_values

class ReplayBuffer:
    """
    Experience Replay Buffer.
    Stores transitions (state, action, reward, next_state, done) and allows sampling.
    """
    def __init__(self, max_size: int, batch_size: int, device: torch.device):
        self.buffer = deque(maxlen=max_size)
        self.batch_size = batch_size
        self.device = device

    def push(self, state: np.ndarray, action: int, reward: float, nextstate: np.ndarray, done: bool):
        """
        Adds a transition to the buffer.
        Args:
            state: Current state (numpy array).
            action: Action taken (integer).
            reward: Reward received (float).
            nextstate: Next state (numpy array).
            done: Whether the episode terminated (boolean). Note: truncated is handled separately in train loop.
        """
        # Store as numpy arrays, conversion to tensor happens during sampling
        self.buffer.append((state, action, reward, nextstate, done))

    def sample(self) -> T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples a random batch of transitions from the buffer.
        Returns:
            A tuple of tensors: (states, actions, rewards, nextstates, dones) on the specified device.
        """
        minibatch = random.sample(self.buffer, self.batch_size)

        # Unpack and convert to numpy arrays first (if not already)
        # Then convert to tensors and move to device
        states = torch.from_numpy(np.stack([s for s,a,r,ns,d in minibatch])).float().to(self.device)
        # Actions need to be LongTensor for gather
        actions = torch.from_numpy(np.stack([a for s,a,r,ns,d in minibatch])).long().to(self.device)
        rewards = torch.from_numpy(np.stack([r for s,a,r,ns,d in minibatch])).float().to(self.device)
        nextstates = torch.from_numpy(np.stack([ns for s,a,r,ns,d in minibatch])).float().to(self.device)
        # Dones need to be float for multiplication in the target calculation
        dones = torch.from_numpy(np.stack([float(d) for s,a,r,ns,d in minibatch])).float().to(self.device) # Convert bool to float (1.0 or 0.0)

        return (states, actions, rewards, nextstates, dones)

    @property
    def size(self) -> int:
        """Returns the current number of transitions in the buffer."""
        return len(self.buffer)

    def can_sample(self) -> bool:
        """Checks if the buffer has enough samples for a batch."""
        return self.size >= self.batch_size

class DQNAgent:
    """
    DQN Agent that interacts with the environment and learns.
    """
    def __init__(self, state_dim: int, action_size: int, config: DQNConfig, device: torch.device):
        self.action_size = action_size
        self.config = config
        self.device = device

        # Q-Network and Target Q-Network
        self.q_network = QNetwork(state_dim, action_size).to(device)
        self.target_q_network = QNetwork(state_dim, action_size).to(device)
        self.target_q_network.load_state_dict(self.q_network.state_dict()) # Initialize target network weights

        # Optimizer and Loss Function
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)
        self.criterion = nn.MSELoss()

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(config.buffer_size, config.batch_size, device)

        # Epsilon for epsilon-greedy policy
        self.epsilon = self.config.epsilon_start
        self.epsilon_decay_rate = (self.config.epsilon_start - self.config.epsilon_end) / self.config.epsilon_decay_steps

    def get_random_action(self) -> int:
        """Returns a random action."""
        return random.randrange(self.action_size)

    def get_greedy_action(self, state: np.ndarray) -> int:
        """
        Returns the best action based on the current Q-network.
        Args:
            state: Current state (numpy array).
        Returns:
            The chosen action (integer).
        """
        # Convert state to tensor, add batch dimension, move to device
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        # Get Q-values from the Q-network
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            
        # Return the action with the highest Q-value
        return q_values.argmax(dim=1).item()

    def choose_action(self, state: np.ndarray, global_step: int) -> int:
        """
        Chooses an action using an epsilon-greedy policy.
        Also updates epsilon.
        Args:
            state: Current state (numpy array).
            global_step: Current total training steps.
        Returns:
            The chosen action (integer).
        """
        # Decay epsilon linearly
        self.epsilon = max(self.config.epsilon_end, self.epsilon - self.epsilon_decay_rate)

        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            return self.get_random_action()
        else:
            return self.get_greedy_action(state)

    def learn(self, global_step: int) -> T.Optional[float]:
        """
        Performs a single learning step (updates Q-network).
        Args:
            global_step: Current total training steps.
        Returns:
            The loss value if an update was performed, None otherwise.
        """
        # Only learn if the buffer has enough samples
        if not self.replay_buffer.can_sample():
            return None

        # Sample a batch from the replay buffer
        states, actions, rewards, nextstates, dones = self.replay_buffer.sample()

        # Calculate target Q values using the target network
        # max_next_q_values: max Q-value for the next state using the target network
        with torch.no_grad():
            # Target Q for the next state: max Q(s', a')
            max_next_q_values = self.target_q_network(nextstates).max(dim=1)[0]
            # Target Q value: r + gamma * max_a' Q_target(s', a') * (1 - done)
            # (1 - done) ensures target is just reward for terminal states
            targets = rewards + self.config.gamma * max_next_q_values * (1 - dones)

        # Get the Q-values for the actions actually taken from the Q-network
        # outputs: Q-values for all actions in the sampled states
        outputs = self.q_network(states)
        # predicted_q_values: Q-values for the *specific* actions taken in the sampled states
        # Use gather to select Q-values corresponding to the 'actions' tensor
        # unsqueeze(1) and squeeze(1) are for correct tensor shape for gather
        predicted_q_values = outputs.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Calculate the loss (MSE between predicted Q and target Q)
        loss = self.criterion(predicted_q_values, targets) # targets does not need detach here as it's already built from no_grad tensors

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        # Optional: Clip gradients to prevent exploding gradients
        # torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update target network periodically
        if global_step % self.config.target_update_frequency == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())
            # print(f"Target network updated at global step {global_step}") # Optional print

        return loss.item() # Return the loss value

def set_seed(seed: int):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train(config: DQNConfig):
    """
    Trains the DQN agent.
    Args:
        config: The DQNConfig object containing hyperparameters.
    """
    # Determine device
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {config.device}")

    # Set seeds for reproducibility
    set_seed(config.seed)

    # Create environment
    # Use a dummy env to get state/action space info without rendering during training
    dummy_env = gym.make(config.env_id)
    state_dim = dummy_env.observation_space.shape[0]
    action_size = dummy_env.action_space.n
    dummy_env.close() # Close the dummy environment

    # Create the actual training environment
    env = gym.make(config.env_id)
    # env = gym.make(config.env_id, render_mode='rgb_array') # Uncomment if you want to record videos

    # Initialize agent
    agent = DQNAgent(state_dim, action_size, config, config.device)

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=f'runs/{config.env_id}_DQN_{time.strftime("%Y%m%d_%H%M%S")}')

    global_step = 0
    episode_count = 0
    start_time = time.time()

    print("Starting training...")

    # Training loop
    for episode in range(1, config.num_episodes + 1):
        obs, info = env.reset(seed=config.seed + episode) # Reset env with changing seed for variation
        done = False
        total_reward = 0
        episode_start_time = time.time()

        while not done and global_step < config.total_timesteps:
            # Choose action using epsilon-greedy policy
            # The state is numpy array from env
            action = agent.choose_action(obs, global_step)

            # Take a step in the environment
            next_obs, reward, terminated, truncated, info = env.step(action)

            # Store the transition in the replay buffer
            # Use 'terminated' for the 'done' flag in the buffer, as truncation doesn't mean end of MDP sequence
            agent.replay_buffer.push(obs, action, reward, next_obs, terminated)

            # Update the agent if buffer is ready and every train_frequency steps
            if agent.replay_buffer.can_sample() and global_step % config.train_frequency == 0:
                loss = agent.learn(global_step)
                if loss is not None:
                    writer.add_scalar("train/loss", loss, global_step)


            # Update current state and episode status
            obs = next_obs
            total_reward += reward
            done = terminated or truncated # Episode ends if terminated or truncated

            global_step += 1
            writer.add_scalar("train/epsilon", agent.epsilon, global_step)


        episode_count += 1
        episode_duration = time.time() - episode_start_time

        # Log episode metrics
        writer.add_scalar("episode/reward", total_reward, episode_count)
        writer.add_scalar("episode/duration_sec", episode_duration, episode_count)
        writer.add_scalar("episode/length_steps", global_step - (global_step - (total_reward)), episode_count) # rough steps for this episode
        writer.add_scalar("episode/epsilon_end", agent.epsilon, episode_count)


        # Print training progress
        if episode_count % config.log_frequency == 0:
             # Calculate average reward over last N episodes (optional)
             # last_n_rewards = ... # Needs storing rewards
             print(f"Episode: {episode_count}/{config.num_episodes}, Global Steps: {global_step}/{config.total_timesteps}, "
                   f"Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.4f}, Time: {time.time()-start_time:.2f}s")

        # Save model periodically
        if episode_count % config.save_frequency == 0:
            model_path = f'models/{config.env_id}_dqn_{episode_count}.pth'
            torch.save(agent.q_network.state_dict(), model_path)
            print(f"Model saved to {model_path}")

        # Stop training if total timesteps reached
        if global_step >= config.total_timesteps:
            print(f"Total timesteps {config.total_timesteps} reached. Stopping training.")
            break

    print("Training finished.")
    env.close()
    writer.close()
    
    # Save final model
    final_model_path = f'models/{config.env_id}_dqn_final.pth'
    torch.save(agent.q_network.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")


def play(config: DQNConfig, model_path: T.Optional[str] = None):
    """
    Runs the trained agent in the environment to visualize performance.
    Args:
        config: The DQNConfig object.
        model_path: Optional path to a saved model file. If None, uses the agent's current weights.
    """
    print("\nStarting playback...")
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Ensure device is set

    # Create environment with rendering enabled
    render_mode = 'human' if config.render_play else None
    env = gym.make(config.env_id, render_mode=render_mode)
    state_dim = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Create a new agent instance for playback (only need Q-network)
    # Initialize with dummy config values for non-training related parameters
    play_config = DQNConfig(env_id=config.env_id, seed=config.seed, total_timesteps=1, num_episodes=1,
                            buffer_size=1, batch_size=1, gamma=config.gamma, learning_rate=config.learning_rate,
                            target_update_frequency=1, train_frequency=1,
                            epsilon_start=0, epsilon_end=0, epsilon_decay_steps=1, # Epsilon doesn't matter in play
                            log_frequency=1, save_frequency=1, render_play=config.render_play, device=config.device)

    agent = DQNAgent(state_dim, action_size, play_config, config.device)

    # Load trained model weights if path is provided
    if model_path:
        try:
            agent.q_network.load_state_dict(torch.load(model_path, map_location=config.device))
            print(f"Loaded model from {model_path}")
        except FileNotFoundError:
            print(f"Model file not found at {model_path}. Using current agent weights.")
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}. Using current agent weights.")


    # Set network to evaluation mode (important for dropout/batchnorm if used, not strictly necessary here but good practice)
    agent.q_network.eval()

    obs, info = env.reset(seed=config.seed + 1000) # Use a different seed for play
    done = False
    total_reward = 0

    while not done:
        # Choose action greedily (no epsilon-greedy during play)
        # State is numpy array from env
        action = agent.get_greedy_action(obs)

        # Take a step in the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Episode ends if terminated or truncated
        done = terminated or truncated

        # Optional: Add a small delay to see the rendering clearly
        # time.sleep(0.01)

    print(f'Playback finished. Total reward: {total_reward}')
    env.close()


if __name__ == '__main__':
    # Create a configuration object
    config = DQNConfig()

    # Create models directory if it doesn't exist
    import os
    os.makedirs('models', exist_ok=True)

    # Start training
    train(config)

    # Start playing with the trained agent (optional: load a specific model)
    # You can specify a model path here, e.g., 'models/CartPole-v1_dqn_final.pth'
    play(config, model_path=f'models/{config.env_id}_dqn_final.pth')