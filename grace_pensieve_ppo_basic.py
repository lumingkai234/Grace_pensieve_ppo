import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import os
import random
from torch.distributions import Categorical
from grace_pensieve_ppo_model import   CNNActor, CNNCritic


class PPO:
    def __init__(self, state_dim, action_dim=6, 
                 actor_lr=1e-6, critic_lr=1e-6, 
                 gamma=0.99, eps_clip=0.2, 
                 K_epochs=10, entropy_coef=0.98,
                 max_grad_norm=0.5, seq_len=5):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_len = seq_len
    
        # 使用CNN网络替换原有网络
        self.actor = CNNActor(state_dim, action_dim, hidden_dim=128, seq_len=seq_len).to(self.device)
        self.critic = CNNCritic(state_dim, hidden_dim=128, seq_len=seq_len).to(self.device)
        
        self.optimizer = optim.Adam([
            {'params': self.actor.parameters(), 'lr': actor_lr},
            {'params': self.critic.parameters(), 'lr': critic_lr}
        ], eps=1e-5)
        
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # 调整探索参数
        self.epsilon = 1.0
        self.epsilon_decay = 0.998
        self.epsilon_min = 0.05
        self.lamda_values = [4096, 2048, 6144, 8192, 12288, 16384]

    def get_action(self, state_seq, explore=True):
        state_tensor = torch.FloatTensor(state_seq).unsqueeze(0).to(self.device)
        
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
       
        print(f"Selected action: {action}")
        return action, log_prob.item(), value.item(), entropy.item()

    def update(self, states_seq, actions, log_probs, rewards, dones, batch_size=256):
        """
        使用采样的序列数据进行 PPO 更新。
        :param states_seq: 状态序列，形状为 (batch_size, seq_len, state_dim)。
        :param actions: 动作序列，形状为 (batch_size, seq_len)。
        :param log_probs: 动作概率序列，形状为 (batch_size, seq_len)。
        :param rewards: 奖励序列，形状为 (batch_size, seq_len)。
        :param dones: 结束标志序列，形状为 (batch_size, seq_len)。
        :param batch_size: 每次更新的批量大小。
        """
        states_seq = torch.FloatTensor(states_seq).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(log_probs).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 计算GAE
        with torch.no_grad():
            values = self.critic(states_seq)
            next_value = self.critic(states_seq[-1].unsqueeze(0)).item()
        
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
            indices = torch.randperm(len(states_seq))[:batch_size]
            batch_states_seq = states_seq[indices]
            batch_actions = actions[indices]
            batch_old_log_probs = old_log_probs[indices]
            batch_advantages = advantages[indices]
            
            # 计算新策略的概率
            probs = self.actor(batch_states_seq)
            dist = Categorical(probs)
            log_probs = dist.log_prob(batch_actions)
            entropy = dist.entropy().mean()
            
            # 计算ratio
            ratios = torch.exp(log_probs - batch_old_log_probs)
            
            # 计算策略损失
            surr1 = ratios * batch_advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # 计算价值函数损失
            values_pred = self.critic(batch_states_seq)
            value_loss = F.mse_loss(values_pred, rewards[indices] + self.gamma * (1 - dones[indices]) * next_value)

            # 总损失
            loss = policy_loss + 0.05 * value_loss
           

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