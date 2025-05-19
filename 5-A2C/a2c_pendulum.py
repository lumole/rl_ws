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
from torch.distributions import Normal

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, action_limit_tensor): # action_limit_tensor 用于缩放tanh输出
        super().__init__()
        self.action_limit = action_limit_tensor
        self.fl1 = nn.Linear(state_dim, 256) # 可以尝试256
        self.fl1_1 = nn.Linear(256, 256) # 可以尝试256

        self.relu = nn.ReLU()
        
        # 输出原始均值，后续通过tanh和action_limit缩放
        self.mean_head_linear = nn.Linear(256, action_dim) 
        self.log_std_head = nn.Linear(256, action_dim)

    def forward(self, x):
        x = self.fl1(x)
        x = self.relu(x)
        x = self.fl1_1(x)
        x = self.relu(x)
        raw_mean = self.mean_head_linear(x)
        # 使用tanh将均值压缩到[-1, 1]，然后乘以action_limit进行缩放
        mean = torch.tanh(raw_mean) * self.action_limit.to(x.device) 
        
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fl1 = nn.Linear(state_dim, 256) # 可以尝试256
        self.fl1_1 = nn.Linear(256, 256) # 可以尝试256
        self.fl2 = nn.Linear(256, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fl1(x)
        x = self.relu(x)
        x = self.fl1_1(x)
        x = self.relu(x)
        x = self.fl2(x)
        return x

class Rollout:
    def __init__(self, max_size=20000, gamma=0.99): # gamma 实际上在Agent的GAE计算中使用
        self.buffer = deque(maxlen=max_size)
        # self.gamma = gamma # GAE的gamma在Agent中定义，这里可以移除

    def push(self, state, raw_action_numpy, reward, nextstate, done): # 存储原始（未裁剪）动作
        done = float(done)
        self.buffer.append((state, raw_action_numpy, reward, nextstate, done))

    def get(self):
        states, raw_actions, rewards, nextstates, dones = zip(*self.buffer)
        states = torch.from_numpy(np.stack(states)).float()
        # raw_actions 是从策略网络直接采样得到的，可能未裁剪
        raw_actions = torch.from_numpy(np.stack(raw_actions)).float()
        if raw_actions.ndim == 1:
            raw_actions = raw_actions.unsqueeze(1)
            
        rewards = torch.from_numpy(np.array(rewards)).float().unsqueeze(1)
        nextstates = torch.from_numpy(np.stack(nextstates)).float()
        dones = torch.tensor(dones).float().unsqueeze(1)
        return states, raw_actions, rewards, nextstates, dones

    def clear(self):
        self.buffer.clear()

    @property
    def buffer_size(self):
        return len(self.buffer)

class Agent:
    def __init__(self, state_dim, action_dim, env_action_low_np, env_action_high_np, action_limit_value, gae_lambda=0.95): # 添加 gae_lambda
        self.action_dim = action_dim
        # 用于裁剪动作到环境范围
        self.env_action_low_np = env_action_low_np
        self.env_action_high_np = env_action_high_np
        
        # 策略网络将使用这个tensor来缩放tanh输出
        self.action_limit_tensor = torch.tensor([action_limit_value], dtype=torch.float32)

        self.gamma = 0.99
        self.gae_lambda = gae_lambda # GAE lambda 参数
        self.lr_policy = 1e-4 
        self.lr_value = 1e-4  
        self.entropy_coeff = 0.01 

        self.p = PolicyNetwork(state_dim, action_dim, self.action_limit_tensor)
        self.v = ValueNetwork(state_dim)
        self.rb = Rollout(max_size=200 * 10) # gamma从Rollout移除，因为它在Agent中处理
        self.p_optimizer = optim.Adam(self.p.parameters(), lr=self.lr_policy)
        self.v_optimizer = optim.Adam(self.v.parameters(), lr=self.lr_value)
        self.v_criterion = torch.nn.MSELoss()

    def get_action(self, state: np.ndarray, deterministic=False):
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).unsqueeze(0).float()
            mean, log_std = self.p(state_tensor) 
            std = torch.exp(log_std)
            
            dist = Normal(mean, std)
            if deterministic:
                action_tensor = mean 
            else:
                action_tensor = dist.sample() 
            
            return action_tensor 

    def update(self):
        if self.rb.buffer_size == 0:
            return None, None 

        states, actions_raw, rewards, nextstates, dones = self.rb.get()
        
        # --- 价值函数相关计算 ---
        state_values = self.v(states)
        next_state_values = self.v(nextstates) # 不需要 no_grad，因为计算GAE时会用到，但其梯度不用于value_loss的目标

        # --- GAE 计算 ---
        advantages_gae = torch.zeros_like(rewards)
        # td_targets 用于价值函数学习
        td_targets = torch.zeros_like(rewards)

        gae = 0
        # 从后往前计算 GAE 和 TD Targets
        # 注意：这里的 dones[i] 指的是 s_i -> s_{i+1} 转换是否是终止状态
        # 如果 s_{i+1} 是终止状态 (dones[i] == 1)，则 V(s_{i+1}) = 0
        for i in reversed(range(rewards.size(0))):
            # td_error (delta_t)
            # next_state_values[i] 是 V(s_{i+1})
            # state_values[i] 是 V(s_i)
            delta = rewards[i] + self.gamma * next_state_values[i] * (1 - dones[i]) - state_values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae 
            advantages_gae[i] = gae
            # TD target: R_t + gamma * V(s_{t+1}) * (1 - done_t)
            td_targets[i] = rewards[i] + self.gamma * next_state_values[i] * (1 - dones[i])
        
        # 为了策略损失，我们将GAE优势视为常数，所以 detach
        advantages_gae_detached = advantages_gae.detach()
        # 价值函数的目标 td_targets 也要 detach，因为 next_state_values 不应通过它反向传播到价值网络自身的目标计算中
        td_targets_detached = td_targets.detach()

        # 归一化 GAE 优势
        advantages_normalized = (advantages_gae_detached - advantages_gae_detached.mean()) / (advantages_gae_detached.std() + 1e-8)
        
        # --- 策略损失 ---
        mean, log_std = self.p(states) 
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        
        log_probs = dist.log_prob(actions_raw).sum(axis=-1, keepdim=True)
        
        policy_loss = -(log_probs * advantages_normalized).mean() - self.entropy_coeff * dist.entropy().mean()
        
        # --- 价值损失 ---
        # V(s) 应该逼近 td_targets_detached
        value_loss = self.v_criterion(state_values, td_targets_detached)

        self.p_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.p.parameters(), max_norm=0.5) 
        self.p_optimizer.step()

        self.v_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.v.parameters(), max_norm=0.5) 
        self.v_optimizer.step()
        
        return policy_loss.item(), value_loss.item()


def train():
    global env, agent 
    writer = SummaryWriter(log_dir='runs/pendulum_gae/' + time.strftime("%Y%m%d_%H%M%S")) # 新的log目录
    
    max_episode_steps = 200 
    global_step_counter = 0
    update_every_n_steps = 200 # 或者 agent.rb.max_size

    for episode in range(1, 50000 + 1): 
        obs, info = env.reset() 
        episode_reward = 0
        
        for step in range(max_episode_steps):
            state = obs 
            
            action_raw_tensor = agent.get_action(state)
            action_raw_numpy = action_raw_tensor.detach().cpu().numpy().flatten() 

            action_clipped_for_env = np.clip(action_raw_numpy, agent.env_action_low_np, agent.env_action_high_np)
            
            obs, reward, terminated, truncated, info = env.step(action_clipped_for_env)
            next_state = obs 
            done = terminated or truncated
            
            agent.rb.push(state, action_raw_numpy, reward, next_state, done)
            
            episode_reward += reward 
            global_step_counter += 1

            # 通常在收集了足够的数据后（例如一个完整的 episode 或固定的步数）进行更新
            if global_step_counter % update_every_n_steps == 0 and agent.rb.buffer_size > 0 : # 或者 if done and agent.rb.buffer_size > 0:
                update_results = agent.update()
                if update_results: 
                    policy_l, value_l = update_results
                    # 可以选择在每个更新步骤而不是每个episode记录损失
                    writer.add_scalar("Loss/Policy_Loss_steps", policy_l, global_step_counter)
                    writer.add_scalar("Loss/Value_Loss_steps", value_l, global_step_counter)
                agent.rb.clear() # 清空缓冲区以便收集下一批数据

            if done:
                break
        
        # 如果更新不是基于 episode 结束，那么可能需要在 episode 结束后检查是否需要更新剩余数据
        # 但通常 A2C (或 A3C 的单线程版本) 会在收集固定数量的经验后更新，或者在 episode 结束后更新
        # 如果是按 episode 更新，则把 update 和 clear 移到这里
        # if agent.rb.buffer_size > 0: # 如果选择在 episode 结束后更新
        #     update_results = agent.update()
        #     if update_results:
        #         policy_l, value_l = update_results
        #         writer.add_scalar("Loss/Policy_Loss_episode", policy_l, episode)
        #         writer.add_scalar("Loss/Value_Loss_episode", value_l, episode)
        #     agent.rb.clear()

        writer.add_scalar("Reward/Episode_Reward", episode_reward, episode)
        
        if episode % 50 == 0:
            print(f'Episode: {episode}, Total reward: {episode_reward:.2f}, Global Steps: {global_step_counter}')

        if episode % 500 == 0:
            torch.save(agent.p.state_dict(), f'model_pendulum_gae_policy_{episode}.pth')
            torch.save(agent.v.state_dict(), f'model_pendulum_gae_value_{episode}.pth')
            print(f"Models saved at episode {episode}")

    writer.close()

def play():
    play_env = gym.make('Pendulum-v1', render_mode='human')
    # 如果训练时用了NormalizeObservation, 理论上评估也需要。
    # 你需要保存训练时 NormalizeObservation 的均值和方差，并在评估时加载它们。
    # 为简单起见，这里我们不使用，但最佳实践是使用的。
    # play_env = gym.wrappers.NormalizeObservation(play_env)
    # # 假设你保存了 obs_rms (running mean/std object)
    # # play_env.obs_rms = torch.load('obs_rms.pth') # 示例加载方式
    # play_env.training = False # 设置为评估模式，不更新均值和方差

    obs, info = play_env.reset()
    done = False
    total_reward = 0
    
    # 加载模型 (确保路径正确，并选择一个训练好的模型)
    model_episode = 2000 # 示例，选择一个表现好的模型
    try:
        agent.p.load_state_dict(torch.load(f'model_pendulum_gae_policy_{model_episode}.pth'))
        agent.v.load_state_dict(torch.load(f'model_pendulum_gae_value_{model_episode}.pth'))
        print(f"Loaded models from episode {model_episode}")
    except FileNotFoundError:
        print(f"Could not find models for episode {model_episode}. Using initial models for playback.")

    agent.p.eval() 
    agent.v.eval()

    for _ in range(200): 
        state_for_agent = obs 

        action_raw_tensor = agent.get_action(state_for_agent, deterministic=True)
        action_raw_numpy = action_raw_tensor.detach().cpu().numpy().flatten()
        action_clipped_for_env = np.clip(action_raw_numpy, agent.env_action_low_np, agent.env_action_high_np)
        
        obs, reward, terminated, truncated, info = play_env.step(action_clipped_for_env)
        total_reward += reward
        play_env.render()
        time.sleep(0.02) 
        if terminated or truncated:
            break
            
    print(f'Total reward during play: {total_reward}')
    play_env.close()

if __name__ == '__main__':
    raw_env = gym.make('Pendulum-v1')
    
    # 对观测进行归一化
    env = gym.wrappers.NormalizeObservation(raw_env)
    # 还可以考虑 NormalizeReward，但 NormalizeObservation 更为关键
    # env = gym.wrappers.NormalizeReward(env, gamma=0.99) # gamma用于回报的衰减计算

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    
    env_action_low_np = raw_env.action_space.low 
    env_action_high_np = raw_env.action_space.high
    action_limit_value = raw_env.action_space.high[0] 

    # 实例化 Agent 时传入 gae_lambda
    agent = Agent(state_dim, action_dim, env_action_low_np, env_action_high_np, action_limit_value, gae_lambda=0.95)

    print("Starting training for Pendulum-v1 (with GAE)...")
    print("\nStarting playback with trained agent...")
    play() 
    print("Playback finished.")
    # try:
    #     train()
    # except KeyboardInterrupt:
    #     print("Training interrupted by user.")
    # finally:
    #     print("Training finished or interrupted.")
    #     # 在训练结束后，如果你想在play函数中使用 NormalizeObservation 的统计数据，可以在这里保存它们
    #     # 例如: if isinstance(env, gym.wrappers.NormalizeObservation):
    #     #           torch.save(env.obs_rms, 'obs_rms.pth')
    #     env.close() 

    #     print("\nStarting playback with trained agent...")
    #     play() 
    #     print("Playback finished.")