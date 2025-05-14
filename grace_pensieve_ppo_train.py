from ppo_config import config
import numpy as np
from grace_pensieve_ppo_env import GraceEnv
from grace_new import read_video_into_frames
import os
import random
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
from grace_pensieve_ppo_replaymemory import ReplayMemory
from grace_pensieve_ppo_basic import PPO


def split_trace_dataset(trace_dir, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1, seed=42):
    """
    按轨迹文件名的首字母分层划分轨迹数据集为训练集、验证集和测试集。
    仅使用轨迹文件中第二列的带宽数据。
    """
    assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6, "数据集比例之和必须为 1"
    
    # 获取轨迹文件列表
    trace_files = [os.path.join(trace_dir, f) for f in os.listdir(trace_dir)]
    
    # 分层划分：根据文件名的首字母分组
    grouped_traces = defaultdict(list)
    for trace_file in trace_files:
        first_letter = os.path.basename(trace_file)[0].lower()  # 获取文件名首字母
        grouped_traces[first_letter].append(trace_file)
    
    # 打乱每组内的文件顺序
    random.seed(seed)
    for group in grouped_traces.values():
        random.shuffle(group)
    
    # 按比例划分每组数据
    train_traces, valid_traces, test_traces = [], [], []
    for group in grouped_traces.values():
        n = len(group)
        train_end = int(n * train_ratio)
        valid_end = train_end + int(n * valid_ratio)
        
        train_traces.extend(group[:train_end])
        valid_traces.extend(group[train_end:valid_end])
        test_traces.extend(group[valid_end:])
    
    # 打乱最终的划分结果
    random.shuffle(train_traces)
    random.shuffle(valid_traces)
    random.shuffle(test_traces)
    
    return train_traces, valid_traces, test_traces

def load_and_scale_trace(trace_file, scale_factor=2.5):
    """
    加载轨迹文件并对带宽值进行放缩，仅使用第二列的带宽数据。
    """
    scaled_trace = []
    with open(trace_file, "r") as f:
        for line in f:
            _, bandwidth = map(float, line.strip().split())  # 忽略第一列时间戳
            scaled_bandwidth = bandwidth * scale_factor
            scaled_trace.append(scaled_bandwidth)
    return scaled_trace

# 新增缓存目录配置
CACHE_DIR = "./video_frame_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cached_frames(video_path, max_frames):
    """获取缓存视频帧，如果不存在则创建"""
    # 生成唯一缓存文件名
    cache_filename = os.path.join(
        CACHE_DIR, 
        os.path.basename(video_path).replace('.', '_') + f"_{max_frames}.pkl"
    )
    
    # 如果缓存存在直接加载
    if os.path.exists(cache_filename):
        with open(cache_filename, 'rb') as f:
            return pickle.load(f)
    
    # 否则读取并保存
    frames = read_video_into_frames(video_path, nframes=max_frames)
    with open(cache_filename, 'wb') as f:
        pickle.dump(frames, f)
    return frames

def validate(ppo, valid_traces, env, fixed_valid_frames, log_file_path):
    """
    在验证集上计算平均 SSIM，并监测带宽变化。
    """
    total_ssim = 0
    total_frames = 0

    # 遍历验证轨迹
    for trace_idx, trace_data in enumerate(valid_traces):
        env.load_trace(trace_data)  # 加载当前验证轨迹
        env.reset(preserve_history=False)  # 重置环境状态

        # 初始化状态
        state = np.zeros((S_LEN, S_INFO), dtype=np.float32)
        video_ssim = 0
        video_frame_count = 0

        # 遍历固定视频帧
        for frame in fixed_valid_frames:
            action_idx, _, _, _ = ppo.get_action(state, explore=False)
            next_state, reward, _ , ssim_value = env.step(frame, action_idx)

            # 更新状态
            state = np.roll(state, -1, axis=0)
            state[-1] = next_state

            # 计算 SSIM（假设 reward 是 SSIM 的值）
            ssim_db = 10 * np.log10(1 / (1 - ssim_value + 1e-6))
            video_ssim += ssim_db
            video_frame_count += 1

        # 每跑完一遍固定视频后重置环境
        env.reset()

        # 累计总的 SSIM 和帧数
        total_ssim += video_ssim
        total_frames += video_frame_count

    # 计算整体平均 SSIM
    avg_ssim = total_ssim / total_frames if total_frames > 0 else 0

    # 写入日志
    with open(log_file_path, "a") as log_file:
        log_file.write(f"Validation completed. Average SSIM: {avg_ssim:.4f}\n")

    return avg_ssim

# 初始化轨迹索引
trace_index = 0  # 全局变量

def get_next_trace(traces):
    """
    循环获取下一条轨迹。
    如果轨迹用完，则从第一个轨迹重新开始。
    """
    global trace_index
    if not traces:
        raise ValueError("The traces list is empty.")
    trace = traces[trace_index]
    trace_index = (trace_index + 1) % len(traces)  # 循环索引
    return trace

if __name__ == "__main__":
    S_INFO = 5  # 状态特征维度，例如：带宽、SSIM、时间复杂度、空间复杂度、帧大小
    S_LEN = 5   # 时间序列长度

    # 指定轨迹文件路径
    trace_dir = "./cooked_traces"
    train_traces, valid_traces, test_traces = split_trace_dataset(trace_dir)

    print("Training traces:", train_traces)

    # 缓存固定视频帧
    fixed_video = "./videos/testvideos/video-42.mp4"
    max_frames_per_video = 250
    cached_frames = get_cached_frames(fixed_video, max_frames=max_frames_per_video)

    # 初始化 PPO 和 ReplayMemory
    ppo = PPO(S_INFO, seq_len=S_LEN)
    memory = ReplayMemory()
    
    steps_per_update = len(train_traces) // 3  # 每跑完三分之一的训练轨迹后更新一次模型
    
    # 日志文件路径
    log_file_path = "./grace_ppo_result_new4/training_log.txt"
    os.makedirs("./grace_ppo_result_new4", exist_ok=True)

    if not os.path.exists(log_file_path):
        with open(log_file_path, "w") as log_file:
            log_file.write("Training Log\n")

    # 初始化环境
    env = GraceEnv()

    # 加载并放缩训练轨迹文件
    scaled_train_traces = [load_and_scale_trace(trace_file, scale_factor=1.0) for trace_file in train_traces]
    scaled_valid_traces = [load_and_scale_trace(trace_file, scale_factor=1.0) for trace_file in valid_traces]

    print(scaled_train_traces[0][:10])

    # 加载模型
    model_path = "./grace_ppo_result_new4/ppo_model_epoch_9_update_27.pth"
    if os.path.exists(model_path):
        ppo.load(model_path)
        print(f"Loaded model from {model_path}")
    else:
        print(f"Model {model_path} not found. Starting training from scratch.")

    # 设置初始 epoch 和 update_counter
    epoch = 0  # 从 epoch_17 开始训练
    update_counter = 0  # 接着之前的更新计数

    step = 0

    while True:
        print(f"Starting epoch {epoch + 1}...")
        rewards = []

        # 遍历训练轨迹
        for trace_idx, trace_data in enumerate(scaled_train_traces):
            env.load_trace(trace_data)  # 加载当前带宽轨迹

            state = np.zeros((S_LEN, S_INFO), dtype=np.float32)

            # 跑固定视频
            for frame in cached_frames:
                action_idx, log_prob, value, entropy = ppo.get_action(state, explore=True)
                next_state, reward, done, ssim_value = env.step(frame, action_idx)

                state = np.roll(state, -1, axis=0)
                state[-1] = next_state

                rewards.append(reward)
                memory.add(next_state, action_idx, log_prob, reward, done)
                step += 1

            # 每跑完一遍固定视频后重置环境
            env.reset()

            # 每跑完三分之一的训练轨迹后更新 PPO 模型
            if (trace_idx + 1) % steps_per_update == 0:
                batch_size = max(64, len(memory) // 10)
                sampled_states_seq, sampled_actions, sampled_log_probs, sampled_rewards, sampled_dones = memory.sample(
                    batch_size=batch_size, seq_len=S_LEN
                )
                ppo.update(sampled_states_seq, sampled_actions, sampled_log_probs, sampled_rewards, sampled_dones)
                memory.clear()

                # 保存模型
                update_counter += 1
                model_path = f"./grace_ppo_result_new4/ppo_model_epoch_{epoch + 1}_update_{update_counter}.pth"
                ppo.save(model_path)
                print(f"Model saved to {model_path}")

        # 每跑完一遍训练轨迹后进行验证
        avg_ssim = validate(ppo, scaled_valid_traces, env, cached_frames, log_file_path)
        print(f"Validation SSIM: {avg_ssim:.4f}")

        avg_reward = np.mean(rewards)
        print(f"Epoch {epoch} completed. Average reward: {avg_reward:.4f}")

        # 写入日志
        with open(log_file_path, "a") as log_file:
            log_file.write(f"Epoch {epoch + 1} completed. Average reward: {avg_reward:.4f}\n")

        epoch += 1
