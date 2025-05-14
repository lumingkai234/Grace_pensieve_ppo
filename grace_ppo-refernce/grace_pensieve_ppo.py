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

# ================== 新增模块 ==================
class Normalizer:
    def __init__(self, state_dim):
        self.state_dim = state_dim
        self.mean = np.zeros(state_dim)
        self.var = np.ones(state_dim)
        self.count = 1e-4

    def update(self, state):
        if not isinstance(state, np.ndarray):
            state = np.array(state).flatten()
        if state.shape != (self.state_dim,):
            raise ValueError(f"State dimension must be {self.state_dim}, got {state.shape}")
        self.count += 1
        self.mean += (state - self.mean) / self.count
        self.var += (state - self.mean) ** 2 / self.count

    def normalize(self, state):
        if not isinstance(state, np.ndarray):
            state = np.array(state).flatten()  # 确保 state 是一个 numpy 数组
        return (state - self.mean) / (np.sqrt(self.var) + 1e-8)
    
class TemporalAttention(nn.Module):
    """时间注意力机制"""
    def __init__(self, hidden_size):
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x,hidden_size):
        # x shape: (seq_len, batch, hidden_size)
        Q = self.query(x)  # (seq_len, batch, hidden)
        K = self.key(x)    # (seq_len, batch, hidden)
        V = self.value(x)  # (seq_len, batch, hidden)
        
        # 计算注意力权重
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(hidden_size)
        attn_weights = F.softmax(scores, dim=-1)
        
        # 应用注意力
        context = torch.matmul(attn_weights, V)  # (seq_len, batch, hidden)
        return context + x  # 残差连接

class LSTMActor(nn.Module):
    """带有时序注意力的LSTM Actor"""
    def __init__(self, state_dim, action_dim=6, seq_len=5):
        super().__init__()
        self.seq_len = seq_len
        self.lstm = nn.LSTM(state_dim, 256, batch_first=True)
        self.attention = TemporalAttention(256)
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        # 初始化
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
        nn.init.orthogonal_(self.fc[-1].weight, gain=0.01)
    
    def forward(self, states_seq):
        # states_seq: (batch, seq_len, state_dim)
        lstm_out, _ = self.lstm(states_seq)  # (batch, seq_len, 256)
        lstm_out = lstm_out.permute(1, 0, 2)  # (seq_len, batch, 256)
        
        # 时序注意力
        attn_out = self.attention(lstm_out)  # (seq_len, batch, 256)
        last_out = attn_out[-1]  # 取最后时间步 (batch, 256)
        
        logits = self.fc(last_out)
        return Categorical(logits=logits)

class LSTMCritic(nn.Module):
    """带有时序处理的Critic"""
    def __init__(self, state_dim, seq_len=5):
        super().__init__()
        self.lstm = nn.LSTM(state_dim, 256, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1))
        
        # 初始化
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
        nn.init.orthogonal_(self.fc[-1].weight, gain=1.0)
    
    def forward(self, states_seq):
        lstm_out, _ = self.lstm(states_seq)
        lstm_out = lstm_out[:, -1, :]  # 取最后时间步 (batch, 256)
        return self.fc(lstm_out).squeeze(-1)

# ================== 修改PPO类 ==================
class PPO:
    def __init__(self, state_dim, action_dim=6, 
                 actor_lr=3e-4, critic_lr=1e-3,
                 gamma=0.99, eps_clip=0.2, 
                 K_epochs=10, entropy_coef=0.01,
                 seq_len=5):  # 新增序列长度参数
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_len = seq_len
        
        # 使用LSTM网络替换原有网络
        self.actor = LSTMActor(state_dim, action_dim, seq_len).to(self.device)
        self.critic = LSTMCritic(state_dim, seq_len).to(self.device)
        
        self.optimizer = optim.Adam([
            {'params': self.actor.parameters(), 'lr': actor_lr},
            {'params': self.critic.parameters(), 'lr': critic_lr}
        ], eps=1e-5)
        
        # 其余参数保持不变...
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.entropy_coef = entropy_coef
        self.normalizer = Normalizer(state_dim)
        self.epsilon = 1.0
        self.epsilon_decay = 0.998
        self.epsilon_min = 0.05
        self.lamda_values = [4096, 2048, 6144, 8192, 12288, 16384]

    def get_action(self, state_seq, explore=True):  # 修改为接收状态序列
        """
        输入: state_seq - (seq_len, state_dim)
        输出: action选择
        """
        # 确保 state_seq[-1] 是一个一维数组
        if isinstance(state_seq[-1], list):
            state_seq[-1] = np.array(state_seq[-1])
        
        # 归一化处理
        self.normalizer.update(state_seq[-1])  # 只更新最新状态
        norm_seq = np.array([self.normalizer.normalize(s) for s in state_seq])
        
        # 转换为张量
        state_tensor = torch.FloatTensor(norm_seq).unsqueeze(0).to(self.device)  # (1, seq_len, state_dim)
        
        with torch.no_grad():
            dist = self.actor(state_tensor)
            value = self.critic(state_tensor)
            
            if explore and random.random() < self.epsilon:
                action = dist.sample().item()
            else:
                action = torch.argmax(dist.probs).item()
                
            log_prob = dist.log_prob(torch.tensor(action).to(self.device))
            entropy = dist.entropy().mean().item()
            
        print(f"Action probs: {dist.probs.cpu().numpy()}")
        print(f"Selected action: {action}, Log probability: {log_prob.item()}, Value: {value.item()}, Entropy: {entropy}")
        return action, log_prob.item(), value.item(), entropy

    def update(self, buffer, batch_size=256):
        # 从buffer获取序列数据
        states_seq = torch.FloatTensor(buffer.states_seq).to(self.device)  # (batch, seq_len, state_dim)
        actions = torch.LongTensor(buffer.actions).to(self.device)
        old_log_probs = torch.FloatTensor(buffer.log_probs).to(self.device)
        
        # 计算GAE
        with torch.no_grad():
            values = self.critic(states_seq)
            next_value = self.critic(states_seq[-1:]).item()
        
        # 后续计算与之前类似，但使用序列数据...
        
        # 策略更新循环
        for _ in range(self.K_epochs):
            # 随机采样
            indices = torch.randperm(len(states_seq))[:batch_size]
            batch_states = states_seq[indices]
            
            # 计算新策略
            dist = self.actor(batch_states)
            log_probs = dist.log_prob(actions[indices])
            
            # 其余PPO更新逻辑保持不变...

# ================== 修改经验回放缓冲区 ==================
class SequenceRolloutBuffer:
    def __init__(self, seq_len=5):
        self.seq_len = seq_len
        self.state_buffer = deque(maxlen=1000)  # 保存所有状态
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        
    def add(self, state, action, log_prob, reward, done):
        self.state_buffer.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        
    def sample_sequences(self, batch_size):
        """采样状态序列"""
        indices = []
        while len(indices) < batch_size:
            idx = np.random.randint(self.seq_len-1, len(self.state_buffer)-1)
            if self.dones[idx - self.seq_len + 1]:  # 确保序列在同一个episode内
                continue
            indices.append(idx)
        
        states_seq = []
        for idx in indices:
            seq = []
            for i in range(idx-self.seq_len+1, idx+1):
                seq.append(self.state_buffer[i])
            states_seq.append(np.array(seq))
        
        return (
            np.array(states_seq),
            np.array(self.actions)[indices],
            np.array(self.log_probs)[indices],
            np.array(self.rewards)[indices],
            np.array(self.dones)[indices]
        )

# ================== 修改训练循环 ==================
if __name__ == "__main__":
    state_dim = 13  # 状态维度增加到 13（5 个过去的带宽 + 空间复杂度 + 时间复杂度 + 丢包率 + 缓冲区大小）
    seq_len = 5  # 使用5帧历史
    trace_file = "/home/lmk/Grace_project/cooked_traces/trace_866_http---www.amazon.com"
    env = GraceEnv(config, trace_file, scale_factor=0.5)
    ppo = PPO(state_dim, seq_len=seq_len)
    buffer = SequenceRolloutBuffer(seq_len)
    num_epochs = 1000

    # 在训练循环中维护状态序列
    for epoch in range(num_epochs):
        state_history = deque(maxlen=seq_len)  # 维护状态序列
        state = env.reset()[0]
        state_history.extend([state]*seq_len)  # 初始填充
        
        while True:
            # 获取当前序列
            current_seq = list(state_history)
            
            # 选择动作
            action, log_prob, value, _ = ppo.get_action(current_seq)
            
            # 执行动作，获取新状态
            next_state, reward, done = env.step(action)
            
            # 保存经验
            buffer.add(next_state, action, log_prob, reward, done)
            
            # 更新状态序列
            state_history.append(next_state)
            
            # 终止判断
            if done:
                break
            
        # 采样时使用序列数据
        states_seq, actions, log_probs, rewards, dones = buffer.sample_sequences(256)
        
        # 更新网络
        ppo.update(states_seq, actions, log_probs, rewards, dones)
    ppo.save("./ppo_result/ppo_model_19_(test).pth")
