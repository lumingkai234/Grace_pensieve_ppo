import os
import numpy as np
from grace_pensieve_ppo_env import GraceEnv
from grace_pensieve_ppo_basic import PPO
from grace_pensieve_ppo_train import split_trace_dataset, load_and_scale_trace, get_cached_frames

def test_single_trace_with_logging(model_path, test_trace, fixed_video, log_file_path, max_frames_per_video=250):
    """
    测试单条轨迹，记录每一帧的码率、当前带宽、PPO 和固定动作的 SSIM（以 dB 为单位）。
    """
    # 初始化 PPO 模型
    S_INFO = 5
    S_LEN = 5
    ppo = PPO(S_INFO, seq_len=S_LEN)

    # 加载模型权重
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        ppo.load(model_path)
        print("Model loaded successfully.")
    else:
        raise FileNotFoundError(f"Model not found at {model_path}")

    # 初始化环境
    env = GraceEnv()

    # 缓存固定视频帧
    cached_frames = get_cached_frames(fixed_video, max_frames=max_frames_per_video)

    # 打开日志文件
    with open(log_file_path, "w") as log_file:
        log_file.write("Frame-by-frame testing results:\n")
        log_file.write("Frame, Bitrate (kbps), Bandwidth (kbps), PPO SSIM (dB), " +
                       ", ".join([f"Fixed Action {i} SSIM (dB)" for i in range(6)]) + "\n")

        # 初始化存储固定动作的 SSIM
        fixed_action_ssim_results = {i: [] for i in range(6)}

        # 测试 PPO 模型
        env.load_trace(test_trace)
        env.reset()
        state = np.zeros((S_LEN, S_INFO), dtype=np.float32)

        ppo_ssim_results = []  # 用于存储 PPO 的逐帧 SSIM

        for frame_idx, frame in enumerate(cached_frames):
            # 获取当前带宽（从轨迹中获取）
            current_bandwidth = test_trace[frame_idx] * 1000 / 25  # 假设轨迹是一个带宽值的列表

            # 测试 PPO 模型
            action_idx, _, _, _ = ppo.get_action(state, explore=False)
            next_state, reward_ppo, _ = env.step(frame, action_idx)

            # 获取当前帧的码率（从 next_state 的最后一维）
            current_bitrate_bits = next_state[-1]  # 当前帧的码率（以 bits 为单位）
            current_bitrate_kbps = current_bitrate_bits / 1000  # 转换为 kbps

            # 计算 PPO 的 SSIM（以 dB 为单位）
            ssim_ppo_db = 10 * np.log10(1 / (1 - reward_ppo + 1e-6))
            ppo_ssim_results.append((frame_idx + 1, current_bitrate_kbps, current_bandwidth, ssim_ppo_db))

            # 更新状态
            state = np.roll(state, -1, axis=0)
            state[-1] = next_state

        # 测试固定动作策略
        for action_idx in range(6):  # 假设有 6 个固定动作
            env.reset()  # 重置环境状态
            env.load_trace(test_trace)  # 重新加载轨迹

            for frame_idx, frame in enumerate(cached_frames):
                next_state, reward_fixed, _ = env.step(frame, action_idx)

                # 计算固定动作的 SSIM（以 dB 为单位）
                ssim_fixed_db = 10 * np.log10(1 / (1 - reward_fixed + 1e-6))

                # 存储固定动作的 SSIM
                fixed_action_ssim_results[action_idx].append(ssim_fixed_db)

        # 将所有模型的结果写入日志文件
        for frame_idx in range(len(cached_frames)):
            # 写入 PPO 的结果
            frame_number, bitrate_kbps, bandwidth_kbps, ssim_ppo_db = ppo_ssim_results[frame_idx]
            log_file.write(f"{frame_number}, {bitrate_kbps:.2f}, {bandwidth_kbps:.2f}, {ssim_ppo_db:.4f}")

            # 写入固定动作的结果
            for action_idx in range(6):
                log_file.write(f", {fixed_action_ssim_results[action_idx][frame_idx]:.4f}")

            log_file.write("\n")  # 换行

    print(f"Testing completed. Results saved to {log_file_path}")


if __name__ == "__main__":
    # 设置随机种子，确保划分一致
    RANDOM_SEED = 42

    # 指定轨迹文件路径
    trace_dir = "./cooked_traces"

    # 划分数据集
    train_traces, valid_traces, test_traces = split_trace_dataset(trace_dir, seed=RANDOM_SEED)

    # 加载并放缩测试轨迹
    scaled_test_traces = [load_and_scale_trace(trace_file, scale_factor=1.0) for trace_file in test_traces]

    # 打印测试集轨迹总数
    print(f"Total number of test traces: {len(scaled_test_traces)}")

    # 固定视频路径
    fixed_video = "./videos/testvideos/video-42.mp4"

    # 模型路径
    model_path = "./grace_ppo_result_new1/ppo_model_epoch_11_update_33.pth"

    # 日志文件路径
    log_file_path = "./grace_ppo_result_new1/test_log_trace_6.txt"

    # 选择测试集的第 6 条轨迹
    trace_index = 5  # 第 6 条轨迹的索引（从 0 开始）
    test_trace = scaled_test_traces[trace_index]

    # 打印选定的轨迹信息
    print(f"Testing trace {trace_index + 1}: {test_traces[trace_index]}")

    # 测试单条轨迹
    test_single_trace_with_logging(model_path, test_trace, fixed_video, log_file_path)

