import os
import matplotlib.pyplot as plt

def plot_ssim_from_logs(log_dir, output_dir):
    """
    根据日志文件绘制每条轨迹的 SSIM 曲线图。
    每个图包含六条曲线（对应六个动作）。
    :param log_dir: 日志文件所在目录。
    :param output_dir: 输出图像保存目录。
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 遍历日志目录中的所有 .txt 文件
    for log_file in sorted(os.listdir(log_dir)):
        if not log_file.endswith(".txt"):
            continue  # 跳过非 .txt 文件

        log_path = os.path.join(log_dir, log_file)
        print(f"Processing {log_path}...")

        # 初始化存储每个动作的帧序号和 SSIM 值
        action_data = {action_idx: {"frame_indices": [], "ssim_values": []} for action_idx in range(6)}

        # 读取日志文件
        with open(log_path, "r") as f:
            lines = f.readlines()

        # 跳过表头，从第二行开始解析
        for line in lines[1:]:
            frame_idx, action_idx, ssim_db = line.strip().split(", ")
            frame_idx = int(frame_idx)
            action_idx = int(action_idx)
            ssim_db = float(ssim_db)

            # 存储数据
            action_data[action_idx]["frame_indices"].append(frame_idx)
            action_data[action_idx]["ssim_values"].append(ssim_db)

        # 绘制图像
        plt.figure(figsize=(30, 6))  # 横向更长的图像
        for action_idx in range(6):
            plt.plot(
                action_data[action_idx]["frame_indices"],
                action_data[action_idx]["ssim_values"],
                label=f"Action {action_idx}",
                linewidth=1  # 更细的线条
            )

        # 设置图像标题和标签
        plt.title(f"SSIM Curves for {log_file}", fontsize=16)
        plt.xlabel("Frame Index", fontsize=14)
        plt.ylabel("SSIM (dB)", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)

        # 保存图像
        output_path = os.path.join(output_dir, f"{log_file.replace('.txt', '.png')}")
        plt.savefig(output_path)
        plt.close()
        print(f"Saved plot to {output_path}")

# 示例调用
if __name__ == "__main__":
    log_dir = "./grace_ppo_result_new2/trace_logs_5"  # 日志文件目录
    output_dir = "./grace_ppo_result_new2/plots_5"   # 输出图像目录
    plot_ssim_from_logs(log_dir, output_dir)