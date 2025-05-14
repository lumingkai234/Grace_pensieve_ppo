import os
import numpy as np
from grace_pensieve_ppo_env import GraceEnv
from grace_pensieve_ppo_basic import PPO
from grace_pensieve_ppo_train import split_trace_dataset, load_and_scale_trace, get_cached_frames

def test_fixed_actions_on_traces(test_traces, fixed_video, log_dir, max_frames_per_video=250, start_trace=0):
    """
    测试固定视频在每个固定动作下的性能，记录每条轨迹的每一帧的 SSIM，并为每条轨迹生成单独的日志文件。
    """
    # 初始化环境
    env = GraceEnv()

    # 缓存固定视频帧
    cached_frames = get_cached_frames(fixed_video, max_frames=max_frames_per_video)

    # 确保日志目录存在
    os.makedirs(log_dir, exist_ok=True)

    # 测试每条轨迹
    for trace_idx, trace_data in enumerate(test_traces):
        if trace_idx + 1 < start_trace:
            continue  # 跳过已完成的轨迹

        print(f"Testing Trace {trace_idx + 1}/{len(test_traces)}...")
        env.load_trace(trace_data)  # 加载测试轨迹
        env.reset()  # 重置环境状态

        # 为当前轨迹创建单独的日志文件
        trace_log_file_path = os.path.join(log_dir, f"trace_{trace_idx + 1}_results.txt")
        with open(trace_log_file_path, "w") as log_file:
            # 写入日志文件表头
            log_file.write("Frame Index, Action Index, SSIM (dB)\n")

            # 遍历每个固定动作
            for action_idx in range(6):
                print(f"  Testing Fixed Action {action_idx}...")
                action_ssim_totals = 0  # 当前动作的 SSIM 累计值

                # 遍历每一帧
                for frame_idx, frame in enumerate(cached_frames):
                    next_state, _, _, ssim_value = env.step(frame, action_idx)

                    # 计算 SSIM（以 dB 为单位）
                    ssim_db = 10 * np.log10(1 / (1 - ssim_value + 1e-6))
                    action_ssim_totals += ssim_db

                    # 写入每一帧的 SSIM 到日志文件
                    log_file.write(f"{frame_idx + 1}, {action_idx}, {ssim_db:.4f}\n")

                # 打印当前动作的平均 SSIM
                avg_ssim = action_ssim_totals / len(cached_frames)
                print(f"    Fixed Action {action_idx} Average SSIM: {avg_ssim:.4f}")

        print(f"Results for Trace {trace_idx + 1} saved to {trace_log_file_path}")

if __name__ == "__main__":
    # 设置随机种子，确保划分一致
    RANDOM_SEED = 42

    # 指定轨迹文件路径
    trace_dir = "./cooked_traces"

    # 划分数据集
    train_traces, valid_traces, test_traces = split_trace_dataset(trace_dir, seed=RANDOM_SEED)

    # 从训练集中抽取部分轨迹进行测试
    num_traces_to_test = 16  # 选择测试的轨迹数量
    selected_traces = train_traces[:num_traces_to_test]

    # 加载并放缩测试轨迹
    scaled_test_traces = [load_and_scale_trace(trace_file, scale_factor=1.0) for trace_file in selected_traces]

    # 打印测试集轨迹总数
    print(f"Total number of selected traces: {len(scaled_test_traces)}")

    # 固定视频路径
    fixed_video = "./videos/testvideos/video-42.mp4"

    # 日志目录路径
    log_dir = "./grace_ppo_result_new2/trace_logs_5"

    # 测试固定动作
    test_fixed_actions_on_traces(scaled_test_traces, fixed_video, log_dir)


