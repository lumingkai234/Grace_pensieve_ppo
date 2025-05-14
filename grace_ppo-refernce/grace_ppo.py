import torch
import torch.nn as nn
import torch.optim as optim
from ppo_config import config
from collections import deque
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm  # 导入进度条库
from grace_ppo_env_new import GraceEnv  # 导入 grace_env 类
from grace_new import read_video_into_frames, GraceInterface, AEModel  # 导入读取视频帧的函数和相关类
import os  # 导入 os 模块
import random

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),  # 增加神经元数量
            nn.ReLU(),
            nn.Linear(256, 256),  # 增加神经元数量
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        self.init_weights()

    def init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.01, 0.01)  # 使用更小的均匀分布初始化权重
                nn.init.constant_(m.bias, 0)

    def forward(self, state):
        logits = self.net(state)
        probs = F.softmax(logits, dim=-1)
        return probs

class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),  # 增加神经元数量
            nn.ReLU(),
            nn.Linear(256, 256),  # 增加神经元数量
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, state):
        return self.net(state).squeeze(-1)

class PPO:
    def __init__(self, state_dim, action_dim=6,
                 actor_lr=1e-5, critic_lr=1e-5,
                 gamma=0.99, eps_clip=0.2, K_epochs=15, epsilon=0.99, entropy_coef=0.01, epsilon_decay=0.995, epsilon_min=0.1):  # 增加 epsilon 和 epsilon_decay
        # 策略网络和价值网络
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        
        # 优化器
        self.optimizer = optim.Adam([
            {'params': self.actor.parameters(), 'lr': actor_lr},
            {'params': self.critic.parameters(), 'lr': critic_lr}
        ])
        
        # 超参数
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.epsilon = epsilon
        self.entropy_coef = entropy_coef
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # 动作对应的 λ 值
        self.lamda_values = [4096, 2048, 6144, 8192, 12288, 16384]

    def get_action(self, state, explore=True):
        state = torch.FloatTensor([state]).unsqueeze(0)
        with torch.no_grad():
            probs = self.actor(state)
            print("Action probabilities:", probs)
            value = self.critic(state)
        
        if explore and random.random() < self.epsilon:
            action = random.randint(0, len(self.lamda_values) - 1)
        else:
            action = torch.multinomial(probs, 1).item()
        
        log_prob = torch.log(probs.squeeze(0)[action])
        lamda = self.lamda_values[action]
        print(action, log_prob.item(), value.item(), lamda)
        return action, log_prob.item(), value.item(), lamda

    def update(self, buffer, batch_size=64):
        for _ in range(self.K_epochs):
            states, actions, old_log_probs, rewards, dones = buffer.sample(batch_size)
            states = torch.FloatTensor(states).unsqueeze(-1)
            actions = torch.LongTensor(actions)
            old_log_probs = torch.FloatTensor(old_log_probs)
            
            returns = []
            discounted_reward = 0
            for reward, done in zip(reversed(rewards), reversed(dones)):
                if done:
                    discounted_reward = 0
                discounted_reward = reward + self.gamma * discounted_reward
                returns.insert(0, discounted_reward)
            returns = torch.FloatTensor(returns)
            
            with torch.no_grad():
                values = self.critic(states)
            advantages = returns - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            probs = self.actor(states)
            log_probs = torch.log(probs.gather(1, actions.unsqueeze(1)).squeeze(1))
            
            ratios = torch.exp(log_probs - old_log_probs)
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
            actor_loss -= self.entropy_coef * entropy
            
            values = self.critic(states)
            critic_loss = F.mse_loss(values, returns)
            
            loss = actor_loss + 0.001 * critic_loss
            print(loss)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.optimizer.step()
            
            # 更新优先级
            priorities = (advantages.abs() + 1e-6).cpu().numpy()
            buffer.update_priorities(np.arange(len(states)), priorities)

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
    state_dim = 1
    buffer = RolloutBuffer()
    ppo = PPO(state_dim)
    env = GraceEnv(config)
    
    with open("INDEX.txt", "r") as f:
        video_files = [line.strip() for line in f.readlines()]
    
    video = video_files[0]
    
    num_epochs = 30
    for epoch in range(num_epochs):
        state = env.reset()
        # print("[DEBUG] State shape:", np.array(state).shape) 
        frames = read_video_into_frames(video, nframes=128)
        # print("[DEBUG] Number of frames:", len(frames))
        for frame in frames:
            print("1")
            action_idx, log_prob, value, lamda = ppo.get_action(state, explore=True)
            print("state:", state)
            reward = env.reward(lamda, state, frame,is_training=True)
            next_state = env.reset()
            done = False
            
            priority = abs(reward) + 1e-6
            buffer.add(state, action_idx, log_prob, reward, done, priority)
            
            state = next_state
            if done:
                state = env.reset()
        
        ppo.update(buffer, batch_size=128)
        buffer.clear()
        
        # 更新 epsilon
        ppo.decay_epsilon()
    
    # state = env.reset()
    # for frame in frames:
    #     action_idx, log_prob, value, lamda = ppo.get_action(state, explore=False)
    #     print("Test action:", action_idx, "Lambda:", lamda)
    #     state = env.reset()
    
    ppo.save("./ppo_result/ppo_model_14_(test).pth")