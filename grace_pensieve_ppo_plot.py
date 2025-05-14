import matplotlib.pyplot as plt
import numpy as np

# 读取日志文件
log_file_path = "./grace_ppo_result_new1/test_log_trace_6.txt"
data = []

with open(log_file_path, "r") as file:
    lines = file.readlines()[2:]  # 跳过前两行标题
    for line in lines:
        values = list(map(float, line.strip().split(", ")))
        data.append(values)

data = np.array(data)

# 提取数据
frame_indices = data[:, 0]  # 轨迹序号
bitrates = data[:, 1]       # 码率 (kbps)
bandwidths = data[:, 2]     # 带宽 (kbps)
ppo_ssim = data[:, 3]       # PPO 的 SSIM
fixed_action_ssim = data[:, 4:]  # 固定动作的 SSIM (6 列)

# 计算每个模型的平均 SSIM
average_ssim = {}
average_ssim["PPO"] = np.mean(ppo_ssim)
for i in range(fixed_action_ssim.shape[1]):
    average_ssim[f"Fixed Action {i}"] = np.mean(fixed_action_ssim[:, i])

# 将平均 SSIM 追加到日志文件末尾
with open(log_file_path, "a") as file:
    file.write("\nAverage SSIM for each model:\n")
    for model, avg_ssim in average_ssim.items():
        file.write(f"{model}: {avg_ssim:.4f}\n")

print(f"平均 SSIM 已追加到文件: {log_file_path}")

# 绘制第一幅图：带宽和码率
plt.figure(figsize=(10, 6))
plt.plot(frame_indices, bandwidths, label="Bandwidth (kbps)", color="blue", marker="o")
plt.plot(frame_indices, bitrates, label="Bitrate (kbps)", color="orange", marker="x")
plt.xlabel("Frame Index")
plt.ylabel("Value (kbps)")
plt.title("Bandwidth and Bitrate vs Frame Index")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("./grace_ppo_result_new1/bandwidth_bitrate_plot.png")  # 保存图像
plt.show()

# 绘制第二幅图：PPO 和每个固定动作的 SSIM 对比
for i in range(fixed_action_ssim.shape[1]):
    plt.figure(figsize=(10, 6))
    plt.plot(frame_indices, fixed_action_ssim[:, i], label=f"Fixed Action {i} SSIM (dB)", marker="o", color="blue")
    plt.plot(frame_indices, ppo_ssim, label="PPO SSIM (dB)", color="red", linestyle="--", linewidth=2)
    plt.xlabel("Frame Index")
    plt.ylabel("SSIM (dB)")
    plt.title(f"SSIM vs Frame Index: PPO vs Fixed Action {i}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"./grace_ppo_result_new1/ssim_comparison_fixed_action_{i}.png")  # 保存图像
    plt.show()