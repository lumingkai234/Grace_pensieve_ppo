# 超参数：状态四个特征的归一化因子
BANDWIDTH_SCALE = 5      # 带宽归一化因子
SSIM_SCALE = 12          # SSIM 归一化因子
SPACE_COMPLEXITY_SCALE = 60  # 空间复杂度归一化因子
TIME_COMPLEXITY_SCALE = 120  # 时间复杂度归一化因子
SIZE_SCALE = 40000

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class CNNActor(nn.Module):
    def __init__(self, state_dim, action_dim=6, hidden_dim=128, seq_len=5):
        super(CNNActor, self).__init__()
        self.seq_len = seq_len

        # 独立的卷积层，用于处理不同的特征
        self.conv_bandwidth = nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1)
        self.conv_ssim = nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1)
        self.conv_space_complexity = nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1)
        self.conv_time_complexity = nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1)
        self.conv_size = nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1)  # 新增帧大小特征的卷积层

        # 全连接层
        incoming_size = 5 * hidden_dim * seq_len  # 5 个特征，每个特征经过卷积后展平
        self.fc1 = nn.Linear(incoming_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, states_seq):
        # states_seq: (batch_size, seq_len, state_dim)
        # 对状态进行归一化
        scale_factors = torch.FloatTensor([BANDWIDTH_SCALE, SSIM_SCALE, SPACE_COMPLEXITY_SCALE, TIME_COMPLEXITY_SCALE, SIZE_SCALE]).to(states_seq.device)
        states_seq = states_seq / scale_factors

        # 分离不同特征
        bandwidth = states_seq[:, :, 0:1].permute(0, 2, 1)  # 带宽特征
        ssim = states_seq[:, :, 1:2].permute(0, 2, 1)  # SSIM 特征
        space_complexity = states_seq[:, :, 2:3].permute(0, 2, 1)  # 空间复杂度特征
        time_complexity = states_seq[:, :, 3:4].permute(0, 2, 1)  # 时间复杂度特征
        size = states_seq[:, :, 4:5].permute(0, 2, 1)  # 帧大小特征

        # 分别通过独立的卷积层
        x_bandwidth = F.relu(self.conv_bandwidth(bandwidth))
        x_ssim = F.relu(self.conv_ssim(ssim))
        x_space_complexity = F.relu(self.conv_space_complexity(space_complexity))
        x_time_complexity = F.relu(self.conv_time_complexity(time_complexity))
        x_size = F.relu(self.conv_size(size))

        # 展平卷积输出
        x_bandwidth = x_bandwidth.view(x_bandwidth.size(0), -1)
        x_ssim = x_ssim.view(x_ssim.size(0), -1)
        x_space_complexity = x_space_complexity.view(x_space_complexity.size(0), -1)
        x_time_complexity = x_time_complexity.view(x_time_complexity.size(0), -1)
        x_size = x_size.view(x_size.size(0), -1)

        # 拼接所有特征
        x = torch.cat([x_bandwidth, x_ssim, x_space_complexity, x_time_complexity, x_size], dim=1)

        # 全连接层
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        probs = F.softmax(logits, dim=-1)
        return probs

class CNNCritic(nn.Module):
    def __init__(self, state_dim, hidden_dim=128, seq_len=5):
        super(CNNCritic, self).__init__()
        self.seq_len = seq_len

        # 独立的卷积层，用于处理不同的特征
        self.conv_bandwidth = nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1)
        self.conv_ssim = nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1)
        self.conv_space_complexity = nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1)
        self.conv_time_complexity = nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1)
        self.conv_size = nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1)  # 新增帧大小特征的卷积层

        # 全连接层
        incoming_size = 5 * hidden_dim * seq_len  # 5 个特征，每个特征经过卷积后展平
        self.fc1 = nn.Linear(incoming_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, states_seq):
        # states_seq: (batch_size, seq_len, state_dim)
        # 对状态进行归一化
        scale_factors = torch.FloatTensor([BANDWIDTH_SCALE, SSIM_SCALE, SPACE_COMPLEXITY_SCALE, TIME_COMPLEXITY_SCALE, 1]).to(states_seq.device)
        states_seq = states_seq / scale_factors

        # 分离不同特征
        bandwidth = states_seq[:, :, 0:1].permute(0, 2, 1)  # 带宽特征
        ssim = states_seq[:, :, 1:2].permute(0, 2, 1)  # SSIM 特征
        space_complexity = states_seq[:, :, 2:3].permute(0, 2, 1)  # 空间复杂度特征
        time_complexity = states_seq[:, :, 3:4].permute(0, 2, 1)  # 时间复杂度特征
        size = states_seq[:, :, 4:5].permute(0, 2, 1)  # 帧大小特征

        # 分别通过独立的卷积层
        x_bandwidth = F.relu(self.conv_bandwidth(bandwidth))
        x_ssim = F.relu(self.conv_ssim(ssim))
        x_space_complexity = F.relu(self.conv_space_complexity(space_complexity))
        x_time_complexity = F.relu(self.conv_time_complexity(time_complexity))
        x_size = F.relu(self.conv_size(size))

        # 展平卷积输出
        x_bandwidth = x_bandwidth.view(x_bandwidth.size(0), -1)
        x_ssim = x_ssim.view(x_ssim.size(0), -1)
        x_space_complexity = x_space_complexity.view(x_space_complexity.size(0), -1)
        x_time_complexity = x_time_complexity.view(x_time_complexity.size(0), -1)
        x_size = x_size.view(x_size.size(0), -1)

        # 拼接所有特征
        x = torch.cat([x_bandwidth, x_ssim, x_space_complexity, x_time_complexity, x_size], dim=1)

        # 全连接层
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value.squeeze(-1)