import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal

# --- 超参数 ---
CONFIG = {
    "seed": 42,
    "env_id": "Pendulum-v1",
    "max_episodes": 3000,       # 增加训练回合数
    "max_steps_per_episode": 200,
    "actor_lr": 3e-4,           # 演员学习率
    "critic_lr": 1e-3,          # 评论家学习率 (通常可以稍高或需要独立调整)
    "gamma": 0.99,              # 折扣因子
    "gae_lambda": 0.95,         # GAE lambda 参数
    "entropy_coeff": 0.005,     # 熵奖励系数 (可适当调小)
    "hidden_size": 256,         # 网络隐藏层大小
    "adam_eps": 1e-5,           # Adam优化器epsilon
    "grad_clip_norm": 0.5,      # 梯度裁剪范数
    "reward_scale_factor": 8.0, # 用于 (reward + factor) / factor 的奖励缩放
    "print_interval": 20,
    "save_interval": 500,
    "log_dir_prefix": "runs/pendulum_A2C_optimized_"
}

# 设置随机种子
SEED = CONFIG["seed"]
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")

# --- 工具函数：权重初始化 ---
def layer_init_orthogonal(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# --- 演员网络 ---
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super().__init__()
        self.fc1 = layer_init_orthogonal(nn.Linear(state_dim, hidden_size))
        self.fc_mu = layer_init_orthogonal(nn.Linear(hidden_size, action_dim), std=0.01) # 输出层用较小std
        self.fc_sigma = layer_init_orthogonal(nn.Linear(hidden_size, action_dim), std=0.01)

    def forward(self, x):
        x = torch.tanh(self.fc1(x)) # 使用 tanh 作为激活函数
        mu = self.fc_mu(x)
        sigma = F.softplus(self.fc_sigma(x)) + 1e-5
        return mu, sigma

# --- 评论家网络 ---
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_size):
        super().__init__()
        self.fc1 = layer_init_orthogonal(nn.Linear(state_dim, hidden_size))
        self.fc_value = layer_init_orthogonal(nn.Linear(hidden_size, 1), std=1.0)

    def forward(self, x):
        x = torch.tanh(self.fc1(x)) # 使用 tanh 作为激活函数
        value = self.fc_value(x)
        return value

# --- 智能体 ---
class Agent:
    def __init__(self, state_dim, action_dim, action_space_low, action_space_high, config):
        self.config = config
        self.gamma = config["gamma"]
        self.gae_lambda = config["gae_lambda"]
        self.entropy_coeff = config["entropy_coeff"]
        self.grad_clip_norm = config["grad_clip_norm"]

        self.action_space_low = torch.tensor(action_space_low, dtype=torch.float32, device=DEVICE)
        self.action_space_high = torch.tensor(action_space_high, dtype=torch.float32, device=DEVICE)
        self.max_angular_velocity = 8.0 # Pendulum 特有的角速度最大值，用于状态标准化

        self.policy_net = PolicyNetwork(state_dim, action_dim, config["hidden_size"]).to(DEVICE)
        self.value_net = ValueNetwork(state_dim, config["hidden_size"]).to(DEVICE)

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=config["actor_lr"], eps=config["adam_eps"])
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=config["critic_lr"], eps=config["adam_eps"])

        self.memory_states = []
        self.memory_actions = []
        self.memory_rewards = []
        self.memory_log_probs = []
        self.memory_dones = []
        self.memory_next_states = [] # GAE 需要下一个状态

    def _normalize_state(self, state):
        # state: [cos(theta), sin(theta), angular_velocity]
        # cos, sin 已经在 [-1, 1]
        # 将 angular_velocity 标准化到大致 [-1, 1]
        state_normalized = state.copy()
        state_normalized[2] = state_normalized[2] / self.max_angular_velocity
        return state_normalized

    def get_action(self, state: np.ndarray, deterministic=False):
        normalized_state = self._normalize_state(state)
        state_tensor = torch.from_numpy(normalized_state).float().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            mu, sigma = self.policy_net(state_tensor)
            action_dist = Normal(mu, sigma)
            if deterministic:
                action_raw = mu
            else:
                action_raw = action_dist.sample()
            log_prob = action_dist.log_prob(action_raw).sum(dim=-1)

        action_np_raw = action_raw.squeeze(0).cpu().numpy()
        action_clipped = np.clip(action_np_raw, self.action_space_low.cpu().numpy(), self.action_space_high.cpu().numpy())

        return action_clipped, log_prob # 返回 log_prob 以便存储

    def store_transition(self, state, action, reward, next_state, done, log_prob):
        self.memory_states.append(torch.from_numpy(self._normalize_state(state)).float().to(DEVICE))
        self.memory_actions.append(torch.tensor(action, dtype=torch.float32).to(DEVICE)) # 存储裁剪后的动作
        self.memory_rewards.append(torch.tensor([reward], dtype=torch.float32).to(DEVICE))
        self.memory_next_states.append(torch.from_numpy(self._normalize_state(next_state)).float().to(DEVICE))
        self.memory_dones.append(torch.tensor([float(done)], dtype=torch.float32).to(DEVICE))
        self.memory_log_probs.append(log_prob) # log_prob 是 tensor

    def clear_memory(self):
        self.memory_states.clear()
        self.memory_actions.clear()
        self.memory_rewards.clear()
        self.memory_log_probs.clear()
        self.memory_dones.clear()
        self.memory_next_states.clear()

    def update(self):
        if not self.memory_rewards:
            return 0.0, 0.0, 0.0

        states_tensor = torch.stack(self.memory_states).to(DEVICE)
        # actions_tensor = torch.stack(self.memory_actions).to(DEVICE) # GAE中不直接用它计算新log_prob
        rewards_tensor = torch.cat(self.memory_rewards).to(DEVICE).squeeze(-1)
        next_states_tensor = torch.stack(self.memory_next_states).to(DEVICE)
        dones_tensor = torch.cat(self.memory_dones).to(DEVICE).squeeze(-1)
        log_probs_old_tensor = torch.cat(self.memory_log_probs).to(DEVICE)

        # --- 计算GAE优势和价值目标 ---
        with torch.no_grad():
            values = self.value_net(states_tensor).squeeze(-1)
            next_values = self.value_net(next_states_tensor).squeeze(-1)

            advantages_gae = torch.zeros_like(rewards_tensor).to(DEVICE)
            last_gae_lam = 0
            for t in reversed(range(len(rewards_tensor))):
                if t == len(rewards_tensor) - 1: # 如果是最后一步
                    # dones_tensor[t] 如果为1 (True), 表示已终止, next_value应为0
                    # 如果为0 (False), 表示未终止 (通常是max_steps截断), 使用next_value
                    delta = rewards_tensor[t] + self.gamma * next_values[t] * (1.0 - dones_tensor[t]) - values[t]
                else:
                    delta = rewards_tensor[t] + self.gamma * values[t+1] * (1.0 - dones_tensor[t]) - values[t]
                    # 注意：这里我们用的是 memory_dones[t] 来判断 s[t] 之后是否是终止状态
                    # 如果 dones_tensor[t] is True, 那么 V(s_{t+1}) 应该是 0.
                    # 对于 GAE, 我们通常这样计算 delta:
                    # delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
                    # A_t = delta_t + gamma * lambda * A_{t+1} * (1 - done_t)
                    # dones_tensor[t] 对应 transition (s_t, a_t, r_t, s_{t+1}, done_t) 中的 done_t
                # 正确的delta计算:
                # (1.0 - dones_tensor[t]) 确保如果当前 transition 导致终止，则不考虑 V(s_next)
                delta = rewards_tensor[t] + self.gamma * next_values[t] * (1.0 - dones_tensor[t]) - values[t]
                advantages_gae[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * last_gae_lam * (1.0 - dones_tensor[t])

            value_targets = advantages_gae + values # GAE论文中，价值函数的目标是 GAE优势 + 当前价值估计
            # 或者使用蒙特卡洛回报作为目标，但GAE论文建议前者
            # returns_monte_carlo = []
            # discounted_sum = 0
            # for reward, done_val in zip(reversed(rewards_tensor.tolist()), reversed(dones_tensor.tolist())):
            #     if done_val: # if episode was done at this step, Gt for this step is just reward
            #         discounted_sum = 0 # Reset if this was a true terminal state within an episode,
            #                            # or use V(s_next_bootstrap) if truncated.
            #                            # For GAE, this calculation is implicitly handled.
            #     discounted_sum = reward + self.gamma * discounted_sum
            #     returns_monte_carlo.insert(0, discounted_sum)
            # value_targets = torch.tensor(returns_monte_carlo, dtype=torch.float32, device=DEVICE)


        # --- 评论家网络更新 ---
        # 注意：这里 value_net(states_tensor) 会重新计算一遍 values，
        # 这对于PyTorch的计算图是必要的，因为values在no_grad()块中计算。
        current_values_for_loss = self.value_net(states_tensor).squeeze(-1)
        value_loss = F.mse_loss(current_values_for_loss, value_targets.detach()) # 目标不参与梯度计算

        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.grad_clip_norm)
        self.value_optimizer.step()

        # --- 演员网络更新 ---
        # 标准化GAE优势 (重要!)
        advantages_normalized = (advantages_gae - advantages_gae.mean()) / (advantages_gae.std() + 1e-8)

        # 重新评估当前策略在新权重下的log_probs和熵
        # (或者使用旧的log_probs，如果这是纯A2C而不是PPO的话，但通常会用旧的)
        # log_probs_old_tensor 是从采样时获得的
        # policy_loss = -(log_probs_old_tensor * advantages_normalized.detach()).mean()
        # 如果要重新评估（通常在PPO中做，A2C用旧的）:
        mu_new, sigma_new = self.policy_net(states_tensor)
        action_dist_new = Normal(mu_new, sigma_new)
        # 需要用当时采取的动作来计算新的log_prob，所以要从memory_actions中获取
        log_probs_new = action_dist_new.log_prob(torch.stack(self.memory_actions)).sum(dim=-1)


        # A2C 使用旧的log_probs (从采样时获得)
        policy_loss = -(log_probs_old_tensor * advantages_normalized.detach()).mean()

        # 熵计算 (基于当前策略)
        # mu_entropy, sigma_entropy = self.policy_net(states_tensor) # 已经计算为 mu_new, sigma_new
        entropy_dist = Normal(mu_new, sigma_new) # 使用当前策略计算熵
        entropy = entropy_dist.entropy().mean()

        actor_total_loss = policy_loss - self.entropy_coeff * entropy

        self.policy_optimizer.zero_grad()
        actor_total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip_norm)
        self.policy_optimizer.step()

        avg_entropy_val = entropy.item()
        self.clear_memory()

        return actor_total_loss.item() - (-self.entropy_coeff * avg_entropy_val), value_loss.item(), avg_entropy_val


# --- 训练函数 ---
def train():
    env = gym.make(CONFIG["env_id"])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_low = env.action_space.low
    action_high = env.action_space.high

    agent = Agent(state_dim, action_dim, action_low, action_high, CONFIG)

    log_dir = f'{CONFIG["log_dir_prefix"]}{time.strftime("%Y%m%d_%H%M%S")}'
    writer = SummaryWriter(log_dir=log_dir)
    print(f"Tensorboard 日志将保存到: {log_dir}")
    print(f"超参数配置: {CONFIG}")


    for episode in range(1, CONFIG["max_episodes"] + 1):
        obs, info = env.reset(seed=SEED + episode)
        episode_reward = 0
        policy_loss_val = 0.0
        value_loss_val = 0.0
        entropy_val = 0.0
        steps_in_episode = 0

        for step in range(CONFIG["max_steps_per_episode"]):
            action, log_prob = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            # 奖励缩放
            scaled_reward = (reward + CONFIG["reward_scale_factor"]) / CONFIG["reward_scale_factor"] \
                            if CONFIG["reward_scale_factor"] is not None else reward
            # scaled_reward = reward / CONFIG["reward_scale_factor"] # simpler scaling

            agent.store_transition(obs, action, scaled_reward, next_obs, terminated or truncated, log_prob)

            obs = next_obs
            episode_reward += reward # 记录原始奖励用于评估
            steps_in_episode += 1

            if terminated or truncated:
                break

        policy_loss_val, value_loss_val, entropy_val = agent.update()

        writer.add_scalar("奖励/每回合原始奖励", episode_reward, episode)
        writer.add_scalar("损失/策略损失(不含熵)", policy_loss_val, episode) # policy_loss_val 已减去熵效应
        writer.add_scalar("损失/价值损失", value_loss_val, episode)
        writer.add_scalar("统计/熵", entropy_val, episode)
        writer.add_scalar("统计/每回合步数", steps_in_episode, episode)

        if episode % CONFIG["print_interval"] == 0:
            print(f'回合: {episode}, 总原始奖励: {episode_reward:.2f}, '
                  f'策略损失: {policy_loss_val:.3f}, 价值损失: {value_loss_val:.3f}, '
                  f'熵: {entropy_val:.3f}, 步数: {steps_in_episode}')

        if episode % CONFIG["save_interval"] == 0:
            torch.save(agent.policy_net.state_dict(), f'{log_dir}/pendulum_A2C_policy_{episode}.pth')
            torch.save(agent.value_net.state_dict(), f'{log_dir}/pendulum_A2C_value_{episode}.pth')
            print(f"已在回合 {episode} 保存模型于 {log_dir}")

    writer.close()
    env.close()
    print("训练完成。")
    return agent, log_dir

# --- 回放函数 ---
def play(trained_agent_policy_path, trained_agent_value_path, config, eval_episodes=5):
    env = gym.make(config["env_id"], render_mode='human')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_low = env.action_space.low
    action_high = env.action_space.high

    # 创建一个新的agent实例来加载模型，或直接在主训练agent上加载
    agent_to_play = Agent(state_dim, action_dim, action_low, action_high, config)
    agent_to_play.policy_net.load_state_dict(torch.load(trained_agent_policy_path, map_location=DEVICE))
    agent_to_play.value_net.load_state_dict(torch.load(trained_agent_value_path, map_location=DEVICE))
    agent_to_play.policy_net.eval() # 设置为评估模式
    agent_to_play.value_net.eval()  # 设置为评估模式

    print("\n使用训练好的智能体开始回放...")
    for ep in range(eval_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        while not done and steps < config["max_steps_per_episode"]:
            action, _ = agent_to_play.get_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            # env.render() # gym新版不需要手动调用render，在make时指定即可
            time.sleep(0.01)
            done = terminated or truncated
            steps += 1
        print(f'回放回合 {ep + 1}: 总奖励: {total_reward:.2f}, 步数: {steps}')
    env.close()


if __name__ == '__main__':
    trained_agent, final_log_dir = train()

    # --- 自动回放最新训练的模型 ---
    # 假设我们总是想回放该次训练的最后一个保存点之前的模型
    # (或者你可以手动指定路径)
    latest_policy_model_path = f'{final_log_dir}/pendulum_A2C_policy_{CONFIG["max_episodes"] // CONFIG["save_interval"] * CONFIG["save_interval"]}.pth'
    latest_value_model_path = f'{final_log_dir}/pendulum_A2C_value_{CONFIG["max_episodes"] // CONFIG["save_interval"] * CONFIG["save_interval"]}.pth'
    
    # 检查文件是否存在
    import os
    if os.path.exists(latest_policy_model_path) and os.path.exists(latest_value_model_path):
         print(f"\n将回放模型: {latest_policy_model_path}")
         play(latest_policy_model_path, latest_value_model_path, CONFIG)
    else:
         print(f"\n未找到指定的模型文件进行回放: {latest_policy_model_path} 或 {latest_value_model_path}")
         print("你可以尝试手动指定已保存的模型路径进行回放。")

    print("程序结束。")