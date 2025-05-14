import os
import numpy as np
from grace_pensieve_ppo_env import GraceEnv
from grace_pensieve_ppo_basic import PPO
from grace_pensieve_ppo_train import split_trace_dataset, load_and_scale_trace, get_cached_frames
import re

def parse_log_file(log_file_path):
    """
    从日志文件中解析每个模型的轨迹 SSIM 值和轨迹长度。
    返回一个字典，键为模型路径，值为一个包含轨迹 SSIM 和轨迹长度的列表。
    """
    model_results = {}
    current_model = None

    with open(log_file_path, "r") as file:
        for line in file:
            # 检测模型路径
            model_match = re.match(r"Testing PPO model: (.+)", line)
            if model_match:
                current_model = model_match.group(1)
                model_results[current_model] = {"ssim": [], "lengths": []}
                continue

            # 检测轨迹 SSIM 值
            trace_match = re.match(r"Trace (\d+): PPO Average SSIM: ([\d.]+)", line)
            if trace_match and current_model:
                ssim_value = float(trace_match.group(2))
                model_results[current_model]["ssim"].append(ssim_value)
                model_results[current_model]["lengths"].append(1)  # 每条轨迹的长度为 1 帧（可扩展）

    return model_results

def calculate_weighted_average_ssim(model_results, trace_lengths):
    """
    计算每个模型的加权平均 SSIM 值。
    加权依据是轨迹的带宽轨迹长度。
    """
    weighted_averages = {}
    for model, results in model_results.items():
        ssim_values = results["ssim"]
        lengths = trace_lengths[:len(ssim_values)]  # 确保长度匹配
        total_weight = sum(lengths)
        weighted_average = sum(ssim * length for ssim, length in zip(ssim_values, lengths)) / total_weight
        weighted_averages[model] = weighted_average
    return weighted_averages

# 日志文件路径
log_file_path = "./grace_ppo_result_new1/test_log_fixed_video1.txt"

# 指定轨迹文件路径
trace_dir = "./cooked_traces"

# 划分数据集（使用随机种子 42）
RANDOM_SEED = 42
train_traces, valid_traces, test_traces = split_trace_dataset(trace_dir, seed=RANDOM_SEED)

# 加载测试集轨迹的带宽长度
trace_lengths = [len(load_and_scale_trace(trace_file, scale_factor=1.0)) for trace_file in test_traces]

# 解析日志文件
model_results = parse_log_file(log_file_path)

# 计算加权平均 SSIM 值
weighted_averages = calculate_weighted_average_ssim(model_results, trace_lengths)

# 打印结果
print("Weighted Average SSIM for each model:")
for model, weighted_avg in weighted_averages.items():
    print(f"{model}: {weighted_avg:.4f}")