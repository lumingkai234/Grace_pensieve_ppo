import torch
import numpy as np
import matplotlib.pyplot as plt
from grace_ppo_env_new import GraceEnv  # 导入 grace_env 类
from grace_new import read_video_into_frames  # 导入读取视频帧的函数
from ppo_config import config
from grace_ppo import PPO  # 导入 PPO 类
import os  # 导入 os 模块
import json  # 导入 json 模块
import pandas as pd  # 导入 pandas 库

# 设置随机数种子
np.random.seed(42)
torch.manual_seed(42)

# 初始化环境和 PPO 模型
state_dim = 1  # 根据实际环境修改
ppo = PPO(state_dim)
env = GraceEnv(config)  # 初始化环境

# 加载模型参数
model_path = "ppo_result/ppo_model_14_(test).pth"
ppo.load(model_path)

# 读取视频文件列表
with open("INDEX.txt", "r") as f:
    video_files = [line.strip() for line in f.readlines()]

# 只处理第一个视频文件
video = video_files[0]

# 读取 128 帧视频数据
frames = read_video_into_frames(video, nframes=128)

# 生成随机的 state 并计算 SSIM 值（使用 PPO 训练的模型）
ssim_values_ppo = []
actions_ppo = []
bandwidth_constraints = []
for i, frame in enumerate(frames):
    state = env.reset(seed=i)  # 使用固定的随机数种子生成 state
    action_idx, log_prob, value, lamda = ppo.get_action(state, explore=False)  # 测试时不使用探索策略   
    ssim_value = env.reward(lamda, state, frame, is_training=False)
    ssim_value_db = 10*np.log10(1.0/(1.0-ssim_value))
    ssim_values_ppo.append(ssim_value_db)
    actions_ppo.append(action_idx)
    bandwidth_constraints.append(state)

# 生成随机的 state 并计算 SSIM 值（使用固定模型 4096）
fixed_model_id = 12288  # 选择固定的模型 ID
ssim_values_fixed = []
for i, frame in enumerate(frames):
    state = env.reset(seed=i)  # 使用固定的随机数种子生成 state
    ssim_value = env.reward(fixed_model_id, state, frame, is_training=False)
    ssim_value_db = 10*np.log10(1.0/(1.0-ssim_value))
    ssim_values_fixed.append(ssim_value_db)

# 保存 SSIM 值和动作到文件
os.makedirs("ppo_result", exist_ok=True)
with open("ppo_result/ssim_values_ppo.json", "w") as f:
    json.dump(ssim_values_ppo, f)
with open("ppo_result/ssim_values_fixed.json", "w") as f:
    json.dump(ssim_values_fixed, f)
with open("ppo_result/bandwidth_constraints.json", "w") as f:
    json.dump(bandwidth_constraints, f)

# 保存动作到 CSV 文件
actions_df = pd.DataFrame(actions_ppo, columns=["action"])
actions_df.to_csv("ppo_result/actions_ppo.csv", index=False)

# 绘制 SSIM 值对比图表并保存
plt.figure(figsize=(10, 6))
plt.plot(ssim_values_ppo, label="PPO Model")
plt.plot(ssim_values_fixed, label=f"Fixed Model (ID: {fixed_model_id})")
plt.xlabel("Frame Index")
plt.ylabel("SSIM Value")
plt.title(f"SSIM Values for Frames\nModel: {model_path}")
plt.legend()
plt.savefig("ppo_result/ssim_comparison_model_test_13.png")
plt.show()