from ppo_config import config
import numpy as np
from grace_pensieve_ppo_env import GraceEnv
from grace_new import read_video_into_frames
import os
import random
from collections import defaultdict
import matplotlib.pyplot as plt
from grace_pensieve_ppo_replaymemory import ReplayMemory
from grace_pensieve_ppo_basic import PPO

def validate(ppo, valid_files, env, video_files, max_frames_per_video):
    """
    在验证集上计算平均 SSIM。
    :param ppo: PPO 模型。
    :param valid_files: 验证集带宽轨迹文件列表。
    :param env: 环境实例。
    :param video_files: 验证集视频文件列表。
    :param max_frames_per_video: 每个视频处理的最大帧数。
    :return: 平均 SSIM。
    """
    total_ssim = 0
    total_frames = 0

    for trace_file in valid_files:
        env.load_trace(trace_file)
        env.set_random_start()
        print(f"[Validation] Using trace: {trace_file}, start index: {env.current_bandwidth_idx}")

        for video in video_files:
            # 强制统一验证帧尺寸为 96x96
            frames = read_video_into_frames(video, frame_size=(96, 96), nframes=max_frames_per_video)
            state = np.zeros((S_LEN, S_INFO), dtype=np.float32)

            for step, frame in enumerate(frames):
                action_idx, _, _, _ = ppo.get_action(state, explore=False)
                next_state, reward, done = env.step(frame, action_idx)
                state = np.roll(state, -1, axis=0)
                state[-1] = next_state

                total_ssim += reward
                total_frames += 1

                if done:
                    break

    avg_ssim = total_ssim / total_frames if total_frames > 0 else 0
    print(f"[Validation] Average SSIM: {avg_ssim:.4f}")
    return avg_ssim

def split_video_dataset(video_dir, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1, seed=42):
    """划分视频数据集"""
    assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6, "数据集比例之和必须为 1"
    
    video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mkv'))]
    random.seed(seed)
    random.shuffle(video_files)
    
    n = len(video_files)
    train_end = int(n * train_ratio)
    valid_end = train_end + int(n * valid_ratio)
    
    return video_files[:train_end], video_files[train_end:valid_end], video_files[valid_end:]

if __name__ == "__main__":
    # 超参数
    S_INFO = 4   # 状态特征维度
    S_LEN = 5    # 时间序列长度
    MAX_FRAMES = 128    # 每个视频最大帧数
    STEPS_PER_UPDATE = 64  # 每16步更新模型
    BATCH_SIZE = 32
    FRAME_SIZE = (96, 96)  # 强制统一帧尺寸

    # 数据集路径
    cooked_traces_dir = "./cooked_traces"
    videos_dir = "/home/lmk/Grace_project/videos/testvideos"

    # 划分数据集
    train_trace, valid_trace, test_trace = GraceEnv.split_traces(cooked_traces_dir)
    train_videos, valid_videos, test_videos = split_video_dataset(videos_dir)

    # 初始化环境和PPO
    env = GraceEnv(train_trace[0])
    env.set_random_start()
    print(f"[Training] Start bandwidth index: {env.current_bandwidth_idx}")

    ppo = PPO(S_INFO, seq_len=S_LEN)
    memory = ReplayMemory()

    # 初始化实时绘图
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))
    line, = ax.plot([], [], 'b-', label="Validation SSIM")
    ax.set_xlabel("Update Step")
    ax.set_ylabel("SSIM")
    ax.set_title("Real-time Validation SSIM Curve")
    ax.grid(True)
    ax.legend()
    plt.show(block=False)

    validation_log = []
    global_step = 0  # 全局步数计数器

    try:
        epoch = 0
        while True:
            print(f"\n=== Epoch {epoch+1} ===")
            
            for video in train_videos:
                # 读取训练视频并统一尺寸
                frames = read_video_into_frames(video, frame_size=FRAME_SIZE, nframes=MAX_FRAMES)
                state = np.zeros((S_LEN, S_INFO), dtype=np.float32)

                for step, frame in enumerate(frames):
                    # 选择动作
                    action_idx, log_prob, value, entropy = ppo.get_action(state, explore=True)
                    
                    # 环境交互
                    next_state, reward, done = env.step(frame, action_idx)
                    state = np.roll(state, -1, axis=0)
                    state[-1] = next_state

                    # 存储经验
                    memory.add(next_state, action_idx, log_prob, reward, done)

                    # 定期更新模型
                    if (step + 1) % STEPS_PER_UPDATE == 0:
                        # 采样并更新PPO
                        states, actions, log_probs, rewards, dones = memory.sample(
                            batch_size=BATCH_SIZE, 
                            seq_len=S_LEN
                        )
                        ppo.update(states, actions, log_probs, rewards, dones)
                        memory.clear()

                        # 验证并更新曲线
                        avg_ssim = validate(ppo, valid_trace, env, valid_videos, MAX_FRAMES)
                        validation_log.append(avg_ssim)
                        
                        # 动态更新绘图
                        line.set_xdata(np.arange(len(validation_log)))
                        line.set_ydata(validation_log)
                        ax.relim()
                        ax.autoscale_view()
                        plt.draw()
                        plt.pause(0.01)  # 重要：允许图像更新
                        
                        # 保存检查点和曲线
                        os.makedirs("./grace_ppo_result1", exist_ok=True)
                        ppo.save(f"./grace_ppo_result1/ppo_step_{global_step}.pth")
                        plt.savefig(f"./grace_ppo_result1/validation_curve_step_{global_step}.png")
                        
                        global_step += 1

            # 衰减探索率
            ppo.decay_epsilon()
            epoch += 1

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        # 保存最终结果
        plt.savefig("./grace_ppo_result1/final_validation_curve.png")
        ppo.save("./grace_ppo_result1/ppo_final.pth")
        plt.ioff()
        plt.close()
        print("All results saved to ./grace_ppo_result1/")