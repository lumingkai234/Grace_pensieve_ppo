import torch
import torch.nn as nn
import torch.optim as optim
from ppo_config import config
from collections import deque
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from grace_pensieve_ppo_env import GraceEnv
from grace_new import read_video_into_frames, GraceInterface, AEModel
import os
import random
from torch.distributions import Categorical

# 新增状态归一化层
class Normalizer:
    def __init__(self, num_features):
        self.n = 0
        self.mean = np.zeros(num_features)
        self.std = np.zeros(num_features)
        
    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = np.zeros_like(x)
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.std = self.std + (x - old_mean) * (x - self.mean)

    def normalize(self, x):
        x = np.array(x)
        return (x - self.mean) / (self.std + 1e-8)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.LayerNorm(512),  # 添加层归一化
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )
        # 使用正交初始化
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=0.01)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, state):
        logits = self.net(state)
        probs = F.softmax(logits, dim=-1)
        return probs

class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        # 使用正交初始化
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, state):
        return self.net(state).squeeze(-1)

class PPO:
    def __init__(self, state_dim, action_dim=6, 
                 actor_lr=3e-4, critic_lr=1e-3,  # 调整学习率
                 gamma=0.99, eps_clip=0.2, 
                 K_epochs=10, entropy_coef=0.01,
                 max_grad_norm=0.5):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim).to(self.device)
        self.optimizer = optim.Adam([
            {'params': self.actor.parameters(), 'lr': actor_lr},
            {'params': self.critic.parameters(), 'lr': critic_lr}
        ], eps=1e-5)
        
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.normalizer = Normalizer(state_dim)  # 状态归一化
        
        # 调整探索参数
        self.epsilon = 1.0
        self.epsilon_decay = 0.998
        self.epsilon_min = 0.05
        self.lamda_values = [4096, 2048, 6144, 8192, 12288, 16384]

    def get_action(self, state, explore=True):
        # 状态归一化
        self.normalizer.update(state)
        norm_state = self.normalizer.normalize(state)
        state_tensor = torch.FloatTensor(norm_state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            probs = self.actor(state_tensor)
            dist = Categorical(probs)
            value = self.critic(state_tensor)
        
        if explore and random.random() < self.epsilon:
            action = dist.sample().item()
        else:
            action = torch.argmax(probs).item()
        
        log_prob = dist.log_prob(torch.tensor(action).to(self.device))
        entropy = dist.entropy()
        
        # 打印概率分布和选择结果
        print(f"Action probabilities: {probs.cpu().numpy()}")
        print(f"Selected action: {action}, Log probability: {log_prob.item()}, Value: {value.item()}, Entropy: {entropy.item()}")
        
        return action, log_prob.item(), value.item(), entropy.item()

    def update(self, buffer, batch_size=256):
        states = torch.FloatTensor(buffer.states).to(self.device)
        actions = torch.LongTensor(buffer.actions).to(self.device)
        old_log_probs = torch.FloatTensor(buffer.log_probs).to(self.device)
        rewards = torch.FloatTensor(buffer.rewards).to(self.device)
        dones = torch.FloatTensor(buffer.dones).to(self.device)
        
        # 计算GAE
        with torch.no_grad():
            values = self.critic(states)
            next_value = self.critic(states[-1].unsqueeze(0)).item()
        
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * 0.95 * gae  # lambda=0.95
            advantages.insert(0, gae)
            next_value = values[t].item()
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # 策略更新
        for _ in range(self.K_epochs):
            indices = torch.randperm(len(states))[:batch_size]
            batch_states = states[indices]
            batch_actions = actions[indices]
            batch_old_log_probs = old_log_probs[indices]
            batch_advantages = advantages[indices]
            
            # 计算新策略的概率
            probs = self.actor(batch_states)
            dist = Categorical(probs)
            log_probs = dist.log_prob(batch_actions)
            entropy = dist.entropy().mean()
            
            # 计算ratio
            ratios = torch.exp(log_probs - batch_old_log_probs)
            
            # 策略损失
            surr1 = ratios * batch_advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 价值函数损失
            values_pred = self.critic(batch_states)
            value_loss = F.mse_loss(values_pred, rewards[indices] + self.gamma * (1 - dones[indices]) * next_value)
            
            # 总损失
            loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy
            print(loss)
            
            # 梯度裁剪
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()


    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        print(f"Updated epsilon: {self.epsilon}")

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.priorities = []
    
    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.dones[:]
        del self.priorities[:]

    def add(self, state, action, log_prob, reward, done, priority):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.priorities.append(priority)

    def sample(self, batch_size):
        priorities = np.array(self.priorities)
        priorities = priorities / priorities.sum()
        indices = np.random.choice(len(self.states), batch_size, p=priorities)
        return (
            np.array(self.states)[indices],
            np.array(self.actions)[indices],
            np.array(self.log_probs)[indices],
            np.array(self.rewards)[indices],
            np.array(self.dones)[indices]
        )
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

if __name__ == "__main__":
    state_dim = 13  # 状态维度保持为 13（5 个过去的带宽 + 空间复杂度 + 时间复杂度 + 5 个过去的丢包率 + 缓冲区大小）
    buffer = RolloutBuffer()
    trace_file = "/home/lmk/Grace_project/cooked_traces/trace_866_http---www.amazon.com"
    ppo = PPO(state_dim)
    env = GraceEnv(config, trace_file)
    
    with open("INDEX.txt", "r") as f:
        video_files = [line.strip() for line in f.readlines()]
    
    num_epochs = 20
    max_frames_per_video = 120  # 每个视频处理的最大帧数
    max_videos_per_epoch = 3  # 每个 epoch 处理的视频数量
    
    # 添加训练进度监控
    with tqdm(total=num_epochs, desc="Training Progress") as pbar:
        for epoch in range(num_epochs):
            state, _, _ = env.reset()
            rewards = []
            video_count = 0  # 计数器，记录处理的视频数量
            for video in video_files:
                if video_count >= max_videos_per_epoch:
                    break  # 如果处理的视频数量达到最大值，则跳出循环
                frames = read_video_into_frames(video, nframes=max_frames_per_video)
                for frame in frames:
                    state, space_complexity, time_complexity, current_bandwidth, loss_rate, buffer_size = env.step(frame)
                    action_idx, log_prob, value, entropy = ppo.get_action(state, explore=True)
                    model_id = ppo.lamda_values[action_idx]  # 使用 lamda_values 中的值作为 model_id
                    reward = env.reward(model_id, frame, loss_rate)  # 使用当前带宽和丢包率计算奖励
                    rewards.append(reward)
                    done = False
                    
                    priority = abs(reward) + 1e-6
                    buffer.add(state, action_idx, log_prob, reward, done, priority)
                    
                    if done:
                        state, _, _ = env.reset()
                
                video_count += 1  # 增加处理的视频数量
            
            # 奖励归一化
            rewards = np.array(rewards)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            
            # 更新 PPO
            ppo.update(buffer, batch_size=120*3)
            buffer.clear()
            
            # 更新 epsilon
            ppo.decay_epsilon()
            pbar.update(1)
            pbar.set_postfix({"Epsilon": ppo.epsilon, "Avg Reward": np.mean(rewards)})
    
    ppo.save("./ppo_result/ppo_model_19_(test).pth")