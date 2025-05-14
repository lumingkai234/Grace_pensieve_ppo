import numpy as np
from grace_pensieve_ppo_env import GraceEnv
from grace_pensieve_ppo_basic import PPO
from grace_pensieve_ppo_train import split_trace_dataset, load_and_scale_trace, get_cached_frames

def measure_fixed_actions_bitrate(fixed_video, env, log_file_path, test_traces, max_frames_per_video=250):
    """
    测量固定视频在 6 个固定动作下的码率，并将结果保存到文件中。
    """
    # 缓存固定视频帧
    cached_frames = get_cached_frames(fixed_video, max_frames=max_frames_per_video)

    # 初始化结果存储
    fixed_action_bitrate_results = {action_idx: [] for action_idx in range(6)}

    # 遍历测试轨迹
    for trace_idx, trace_data in enumerate(test_traces):
        print(f"Testing trace {trace_idx + 1}/{len(test_traces)}...")

        # 加载当前轨迹到环境
        env.load_trace(trace_data)
        env.reset()  # 重置环境状态

        # 测试每个固定动作
        for action_idx in range(6):  # 假设有 6 个固定动作
            action_bitrates = []  # 存储当前动作的码率

            for frame_idx, frame in enumerate(cached_frames):
                next_state, _, _ = env.step(frame, action_idx)

                # 获取当前帧的码率（从 next_state 的最后一维）
                current_bitrate_bits = next_state[-1]  # 当前帧的码率（以 bits 为单位）
                current_bitrate_kbps = current_bitrate_bits / 1000  # 转换为 kbps

                # 存储码率
                action_bitrates.append(current_bitrate_kbps)

            # 将当前动作的码率存储到结果中
            fixed_action_bitrate_results[action_idx].append(np.mean(action_bitrates))  # 计算平均码率

    # 将结果写入日志文件
    with open(log_file_path, "w") as log_file:
        log_file.write("Trace Index, " + ", ".join([f"Fixed Action {i} Avg Bitrate (kbps)" for i in range(6)]) + "\n")
        for trace_idx in range(len(test_traces)):
            log_file.write(f"{trace_idx + 1}, " + ", ".join(
                [f"{fixed_action_bitrate_results[action_idx][trace_idx]:.2f}" for action_idx in range(6)]
            ) + "\n")

    print(f"Fixed action bitrate results saved to {log_file_path}")

def record_fixed_action_bitrates(fixed_video, env, log_file_path, max_frames_per_video=250):
    """
    测量固定视频在每个固定动作下的每帧码率，并保存到日志文件中。
    """
    # 缓存固定视频帧
    cached_frames = get_cached_frames(fixed_video, max_frames=max_frames_per_video)

    # 加载一个伪轨迹到环境中，确保 self.trace 不为 None
    dummy_trace = [1.0] * len(cached_frames)  # 创建一个伪轨迹，长度与视频帧数一致
    env.load_trace(dummy_trace)

    # 初始化结果存储
    fixed_action_bitrate_results = {action_idx: [] for action_idx in range(6)}  # 假设有 6 个固定动作

    # 遍历每个固定动作
    for action_idx in range(6):
        print(f"Processing Fixed Action {action_idx}...")
        env.reset()  # 重置环境状态

        for frame_idx, frame in enumerate(cached_frames):
            next_state, _, _ = env.step(frame, action_idx)

            # 获取当前帧的码率（从 next_state 的最后一维）
            current_bitrate_bits = next_state[-1]  # 当前帧的码率（以 bits 为单位）
            current_bitrate_kbps = current_bitrate_bits / 1000  # 转换为 kbps

            # 存储码率
            fixed_action_bitrate_results[action_idx].append(current_bitrate_kbps)

    # 将结果写入日志文件
    with open(log_file_path, "w") as log_file:
        # 写入表头
        log_file.write("Frame Index, " + ", ".join(
            [f"Fixed Action {i} Bitrate (kbps)" for i in range(6)]
        ) + "\n")

        # 写入每帧的结果
        for frame_idx in range(len(cached_frames)):
            log_file.write(f"{frame_idx + 1}, " + ", ".join(
                [f"{fixed_action_bitrate_results[action_idx][frame_idx]:.2f}" for action_idx in range(6)]
            ) + "\n")

    print(f"Fixed action bitrate results saved to {log_file_path}")

if __name__ == "__main__":
    # 初始化环境
    env = GraceEnv()

    # 固定视频路径
    fixed_video = "./videos/testvideos/video-42.mp4"

    # 日志文件路径
    log_file_path = "./grace_ppo_result_new1/fixed_action_bitrate_results.txt"

    # 测量固定动作的码率
    record_fixed_action_bitrates(fixed_video, env, log_file_path)

