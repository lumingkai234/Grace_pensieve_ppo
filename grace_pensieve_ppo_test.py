import os
import numpy as np
from grace_pensieve_ppo_env import GraceEnv
from grace_pensieve_ppo_basic import PPO
from grace_pensieve_ppo_train import split_trace_dataset, load_and_scale_trace, get_cached_frames

def test_ppo_model_only(model_path, test_traces, fixed_video, log_file_path, max_frames_per_video=250, start_trace=0):
    """
    测试训练好的 PPO 模型在测试集上的性能，并计算加权平均 SSIM。
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

    # 存储每条轨迹的 SSIM 和轨迹长度
    ssim_values = []
    trace_lengths = []

    # 测试 PPO 模型
    with open(log_file_path, "a") as log_file:  # 确保所有写操作都在这个上下文中
        log_file.write(f"\nTesting PPO model: {model_path}\n")  # 添加模型信息到日志文件
        if start_trace == 0:
            log_file.write("Testing PPO model on test set...\n")

        for trace_idx, trace_data in enumerate(test_traces):
            if trace_idx + 1 < start_trace:
                continue  # 跳过已完成的轨迹

            env.load_trace(trace_data)  # 加载测试轨迹
            env.reset()  # 重置环境状态

            # 初始化状态
            state = np.zeros((S_LEN, S_INFO), dtype=np.float32)

            # 测试 PPO 模型
            total_ssim_ppo = 0
            for frame in cached_frames:
                action_idx, _, _, _ = ppo.get_action(state, explore=False)
                next_state, reward, _ ,ssim_value= env.step(frame, action_idx)

                # 更新状态
                state = np.roll(state, -1, axis=0)
                state[-1] = next_state

                # 计算 SSIM（以 dB 为单位）
                ssim_db = 10 * np.log10(1 / (1 - ssim_value + 1e-6))
                total_ssim_ppo += ssim_db

            avg_ssim_ppo = total_ssim_ppo / len(cached_frames)
            log_message = f"Trace {trace_idx + 1}: PPO Average SSIM: {avg_ssim_ppo:.4f}\n"
            log_file.write(log_message)
            print(log_message.strip())  # 打印到终端

            # 记录 SSIM 和轨迹长度
            ssim_values.append(avg_ssim_ppo)
            trace_lengths.append(len(cached_frames))

        # 计算加权平均 SSIM
        total_weight = sum(trace_lengths)
        weighted_avg_ssim = sum(ssim * length for ssim, length in zip(ssim_values, trace_lengths)) / total_weight

        # 写入加权平均 SSIM 到日志文件
        log_file.write(f"\nWeighted Average SSIM: {weighted_avg_ssim:.4f}\n")
        print(f"Weighted Average SSIM: {weighted_avg_ssim:.4f}")  # 打印到终端


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

    # PPO 模型路径列表
    model_paths = [
        "./grace_ppo_result_new2/ppo_model_epoch_16_update_48.pth",
        "./grace_ppo_result_new2/ppo_model_epoch_20_update_60.pth",
        "./grace_ppo_result_new2/ppo_model_epoch_24_update_72.pth",
        "./grace_ppo_result_new2/ppo_model_epoch_28_update_84.pth",
        "./grace_ppo_result_new2/ppo_model_epoch_34_update_102.pth",
    ]

    # 日志文件路径
    log_file_path = "./grace_ppo_result_new2/test_log_fixed_video1.txt"

    # 确保日志目录存在
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    # 手动设置从 Trace 0 开始
    start_trace = 0

    # 循环测试多个 PPO 模型
    for model_path in model_paths:
        print(f"\nTesting model: {model_path}")
        with open(log_file_path, "a") as log_file:
            log_file.write(f"\nTesting PPO model: {model_path}\n")
        # 调用测试函数
        test_ppo_model_only(
            model_path, scaled_test_traces, fixed_video, log_file_path, start_trace=start_trace
        )