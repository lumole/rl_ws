import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque, namedtuple
import matplotlib.pyplot as plt # 用于绘图
import os # 用于文件路径操作
from torch.utils.tensorboard import SummaryWriter # <<< TensorBoard导入

# 超参数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BUFFER_SIZE = int(1e5)      # 经验回放缓冲区大小
BATCH_SIZE = 128            # 小批量大小
GAMMA = 0.99                # 折扣因子
TAU = 1e-3                  # 目标网络软更新参数
LR_ACTOR = 1e-4             # Actor学习率
LR_CRITIC = 1e-3            # Critic学习率
WEIGHT_DECAY_CRITIC = 0.0001 # Critic优化器的L2权重衰减
NOISE_SIGMA = 0.1           # 高斯噪声的标准差 (用于动作探索)
MAX_TIMESTEPS_PER_EPISODE = 200 # 对于Pendulum，最大时间步通常是200
NUM_EPISODES = 300          # 训练的总回合数

# 模型保存相关常量
MODEL_DIR = "./ddpg_pendulum_models"
BEST_MODEL_FILENAME_PREFIX = "ddpg_pendulum_best"
FINAL_MODEL_FILENAME_PREFIX = "ddpg_pendulum_final"
PERIODIC_SAVE_FILENAME_PREFIX = "ddpg_pendulum_episode"

# TensorBoard日志目录
LOG_DIR = "runs/pendulum_ddpg_experiment" # <<< TensorBoard日志目录

# 经验元组
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    """固定大小的缓冲区，用于存储经验元组"""
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        """保存一个经验元组"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """从内存中随机采样一批经验元组"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """返回当前内存中的经验数量"""
        return len(self.memory)

class Actor(nn.Module):
    """Actor (策略) 网络"""
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.max_action = max_action # 动作的最大值，用于缩放输出

        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x)) * self.max_action
        return action

class Critic(nn.Module):
    """Critic (Q值) 网络"""
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q = F.relu(self.fc1(sa))
        q = F.relu(self.fc2(q))
        q = self.fc3(q)
        return q

class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.device = DEVICE
        self.gamma = GAMMA
        self.tau = TAU
        self.max_action = max_action

        self.main_actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.main_critic = Critic(state_dim, action_dim).to(self.device)
        self.target_actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.target_critic = Critic(state_dim, action_dim).to(self.device)

        self.target_actor.load_state_dict(self.main_actor.state_dict())
        self.target_critic.load_state_dict(self.main_critic.state_dict())
        for param in self.target_actor.parameters():
            param.requires_grad = False
        for param in self.target_critic.parameters():
            param.requires_grad = False

        self.main_actor_optimizer = optim.Adam(self.main_actor.parameters(), lr=LR_ACTOR)
        self.main_critic_optimizer = optim.Adam(self.main_critic.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY_CRITIC)
        self.v_criterion = nn.MSELoss()
        self.memory = ReplayBuffer(BUFFER_SIZE)
        self.noise_sigma_current = NOISE_SIGMA
        self.learn_step_counter = 0 # <<< 用于TensorBoard记录学习步骤

    def select_action(self, state, add_noise=True):
        state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        self.main_actor.eval()
        with torch.no_grad():
            action = self.main_actor(state_tensor).cpu().data.numpy().flatten()
        self.main_actor.train()
        if add_noise:
            noise = np.random.normal(0, self.noise_sigma_current * self.max_action, size=action.shape)
            action = action + noise
        return np.clip(action, -self.max_action, self.max_action)

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return None, None # <<< 如果不学习，则返回None

        self.learn_step_counter += 1 # <<< 增加学习步骤计数器

        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        states = torch.FloatTensor(np.array(batch.state)).to(self.device)
        actions = torch.FloatTensor(np.array(batch.action)).to(self.device)
        if actions.ndim == 1:
             actions = actions.unsqueeze(1)
        rewards = torch.FloatTensor(np.array(batch.reward)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        dones = torch.FloatTensor(np.array(batch.done).astype(np.uint8)).unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            target_q_values = self.target_critic(next_states, next_actions)
            y = rewards + self.gamma * (1 - dones) * target_q_values

        current_q_values = self.main_critic(states, actions)
        critic_loss = self.v_criterion(current_q_values, y)

        self.main_critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.main_critic.parameters(), 1.0)
        self.main_critic_optimizer.step()

        actor_actions = self.main_actor(states)
        actor_loss = -self.main_critic(states, actor_actions).mean()

        self.main_actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.main_actor.parameters(), 1.0)
        self.main_actor_optimizer.step()

        self.soft_update_target_networks()

        return critic_loss.item(), actor_loss.item() # <<< 返回损失值

    def soft_update_target_networks(self):
        for target_param, main_param in zip(self.target_actor.parameters(), self.main_actor.parameters()):
            target_param.data.copy_(self.tau * main_param.data + (1.0 - self.tau) * target_param.data)
        for target_param, main_param in zip(self.target_critic.parameters(), self.main_critic.parameters()):
            target_param.data.copy_(self.tau * main_param.data + (1.0 - self.tau) * target_param.data)

    def save_networks(self, directory, filename_prefix):
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.main_actor.state_dict(), f"{directory}/{filename_prefix}_actor.pth")
        torch.save(self.main_critic.state_dict(), f"{directory}/{filename_prefix}_critic.pth")
        print(f"Saved: {directory}/{filename_prefix}_actor.pth and _critic.pth")

    def load_networks(self, actor_path, critic_path=None):
        try:
            self.main_actor.load_state_dict(torch.load(actor_path, map_location=self.device))
            self.target_actor.load_state_dict(self.main_actor.state_dict())
            print(f"Loaded actor network from {actor_path}")
            if critic_path and os.path.exists(critic_path):
                self.main_critic.load_state_dict(torch.load(critic_path, map_location=self.device))
                self.target_critic.load_state_dict(self.main_critic.state_dict())
                print(f"Loaded critic network from {critic_path}")
            elif critic_path:
                 print(f"Critic model at {critic_path} not found, not loaded.")
        except Exception as e:
            print(f"Error loading networks: {e}")


def train_agent(num_episodes_train=NUM_EPISODES):
    env = gym.make("Pendulum-v1")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    print(f"Device: {DEVICE}")
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}, Max action: {max_action}")

    agent = DDPGAgent(state_dim, action_dim, max_action)

    # <<< 初始化TensorBoard Writer
    writer = SummaryWriter(log_dir=LOG_DIR)
    # <<< 确保日志目录存在
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    # <<< 为避免覆盖，可以每次运行时创建唯一的子目录
    # import time
    # current_time = time.strftime("%Y%m%d-%H%M%S")
    # writer = SummaryWriter(log_dir=os.path.join(LOG_DIR, current_time))


    episode_rewards = []
    avg_rewards_list = []
    best_avg_reward = -float('inf') # 用于保存最佳模型

    print(f"Starting training for {num_episodes_train} episodes...")
    print(f"TensorBoard logs will be saved to: {writer.log_dir}") # <<< 打印日志路径

    total_steps = 0 # <<< 用于记录总步数，作为损失的x轴

    for episode in range(1, num_episodes_train + 1):
        state, info = env.reset()
        episode_reward = 0
        episode_steps = 0 # <<< 当前回合的步数

        # 可选：实现噪声衰减
        # agent.noise_sigma_current = NOISE_SIGMA * (1 - episode / num_episodes_train)
        # writer.add_scalar('Parameters/NoiseSigma', agent.noise_sigma_current, episode) # <<< 记录噪声

        for t in range(MAX_TIMESTEPS_PER_EPISODE): # 使用 Pendulum 的最大步数
            action = agent.select_action(state, add_noise=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.store_transition(state, action, reward, next_state, done)

            critic_loss, actor_loss = agent.learn() # <<< 获取损失
            if critic_loss is not None and actor_loss is not None:
                writer.add_scalar('Loss/Critic_Loss', critic_loss, agent.learn_step_counter) # <<< 按学习步骤记录
                writer.add_scalar('Loss/Actor_Loss', actor_loss, agent.learn_step_counter)   # <<< 按学习步骤记录

            state = next_state
            episode_reward += reward
            total_steps += 1
            episode_steps +=1
            if done:
                break

        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards[-100:])
        avg_rewards_list.append(avg_reward)

        # <<< TensorBoard 记录 (每个回合)
        writer.add_scalar('Reward/Episode_Reward', episode_reward, episode)
        writer.add_scalar('Reward/Average_Reward_100_Episodes', avg_reward, episode)
        writer.add_scalar('Performance/Episode_Duration_Steps', episode_steps, episode)


        print(f"Episode {episode}/{num_episodes_train} | Reward: {episode_reward:.2f} | Avg Reward (last 100): {avg_reward:.2f} | Noise: {agent.noise_sigma_current:.3f} | Steps: {episode_steps}")

        # 保存最佳模型 (例如，在训练了至少10%的回合后开始)
        if episode > max(10, num_episodes_train // 10) and avg_reward > best_avg_reward :
            best_avg_reward = avg_reward
            print(f"New best average reward: {best_avg_reward:.2f}. Saving best model...")
            agent.save_networks(directory=MODEL_DIR, filename_prefix=BEST_MODEL_FILENAME_PREFIX)

        # 可选：定期保存模型
        # if episode % 50 == 0:
        #     agent.save_networks(directory=MODEL_DIR, filename_prefix=f"{PERIODIC_SAVE_FILENAME_PREFIX}_{episode}")

    print("Training finished. Saving final model...")
    agent.save_networks(directory=MODEL_DIR, filename_prefix=FINAL_MODEL_FILENAME_PREFIX)
    env.close()
    writer.close() # <<< 关闭TensorBoard Writer

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.title("Episode Reward Over Time")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")

    plt.subplot(1, 2, 2)
    plt.plot(avg_rewards_list)
    plt.title("Average Reward (Last 100 Episodes) Over Time")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")

    plt.tight_layout()
    plt.savefig(os.path.join(writer.log_dir, "rewards_plot.png")) # <<< 将图表也保存到日志目录
    plt.show()
    # 返回最终模型的actor路径，方便后续直接播放
    return f"{MODEL_DIR}/{FINAL_MODEL_FILENAME_PREFIX}_actor.pth"


def play_trained_model(env_name="Pendulum-v1", actor_model_path=None, num_play_episodes=5):
    if actor_model_path is None:
        print("Error: No actor model path provided for playing.")
        return

    print(f"\nStarting to play with model: {actor_model_path}")
    play_env = gym.make(env_name, render_mode="human") # 开启渲染
    state_dim = play_env.observation_space.shape[0]
    action_dim = play_env.action_space.shape[0]
    max_action = float(play_env.action_space.high[0])

    play_agent = DDPGAgent(state_dim, action_dim, max_action) # 创建新的agent实例

    try:
        # 仅加载actor模型用于播放，因为critic不直接参与动作选择
        play_agent.main_actor.load_state_dict(torch.load(actor_model_path, map_location=DEVICE))
        play_agent.main_actor.eval() # 设置为评估模式
        print(f"Successfully loaded actor model from {actor_model_path}")
    except FileNotFoundError:
        print(f"Error: Actor model file not found at {actor_model_path}. Cannot play.")
        play_env.close()
        return
    except Exception as e:
        print(f"Error loading actor model: {e}. Cannot play.")
        play_env.close()
        return

    episode_rewards_play = []
    for episode in range(1, num_play_episodes + 1):
        state, info = play_env.reset()
        current_episode_reward = 0
        done = False
        truncated = False

        for t in range(MAX_TIMESTEPS_PER_EPISODE): # 使用环境的最大步数
            action = play_agent.select_action(state, add_noise=False) # 播放时不加噪声
            next_state, reward, terminated, truncated, info = play_env.step(action)
            done = terminated or truncated
            state = next_state
            current_episode_reward += reward
            if done:
                break

        episode_rewards_play.append(current_episode_reward)
        print(f"Play Episode: {episode}, Reward: {current_episode_reward:.2f}")

    avg_play_reward = np.mean(episode_rewards_play)
    print(f"\nAverage reward over {num_play_episodes} play episodes: {avg_play_reward:.2f}")
    play_env.close()


if __name__ == '__main__':
    # --- 模式选择 ---
    MODE = "train_then_play"  # 仅训练并保存模型
    # MODE = "train_then_play" # 先训练，然后播放刚训练好的模型
    # MODE = "play"   # 仅播放预先训练好的模型

    # --- 指定要播放的模型路径 (如果MODE = "play") ---
    # 如果MODE = "play"，请确保这里的路径指向一个已存在的actor模型文件
    ACTOR_MODEL_TO_PLAY = f"{MODEL_DIR}/{BEST_MODEL_FILENAME_PREFIX}_actor.pth"
    # 或: ACTOR_MODEL_TO_PLAY = f"{MODEL_DIR}/{FINAL_MODEL_FILENAME_PREFIX}_actor.pth"
    # 或: ACTOR_MODEL_TO_PLAY = f"{MODEL_DIR}/{PERIODIC_SAVE_FILENAME_PREFIX}_300_actor.pth" # 假设保存了第300回合的模型
    # ACTOR_MODEL_TO_PLAY = f"{MODEL_DIR}/{FINAL_MODEL_FILENAME_PREFIX}_actor.pth" # 默认播放最终模型


    if MODE == "train":
        print("Mode: Train")
        train_agent(num_episodes_train=NUM_EPISODES)
    elif MODE == "train_then_play":
        print("Mode: Train then Play")
        trained_actor_model_path = train_agent(num_episodes_train=NUM_EPISODES)
        if os.path.exists(trained_actor_model_path):
            play_trained_model(actor_model_path=trained_actor_model_path, num_play_episodes=5)
        else:
            print(f"Trained model path not found: {trained_actor_model_path}. Cannot play.")
    elif MODE == "play":
        print("Mode: Play")
        if not os.path.exists(ACTOR_MODEL_TO_PLAY):
            print(f"Model file for playing ({ACTOR_MODEL_TO_PLAY}) not found. Please train a model first or check the path.")
        else:
            play_trained_model(actor_model_path=ACTOR_MODEL_TO_PLAY, num_play_episodes=10) # 播放更多回合
    else:
        print("Invalid MODE selected. Choose 'train', 'play', or 'train_then_play'.")