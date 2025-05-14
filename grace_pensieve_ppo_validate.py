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
import cv2
S_LEN = 5
S_INFO = 4
import copy
import math
from functools import lru_cache
from threading import Lock
from ppo_config import config
from grace_new import encode_frame, decode_frame, to_tensor, SSIM, AEModel
from grace.grace_gpu_interface import GraceInterface
import numpy as np
import torch
from prometheus_client import Gauge
from PIL import Image
import cv2
import os
import random

def validate(ppo, valid_files, env, fixed_valid_frames, max_frames_per_video, fixed_start_indices):
    """
    在验证集上计算平均 SSIM。
    :param ppo: PPO 模型。
    :param valid_files: 验证集轨迹文件列表。
    :param env: 环境实例。
    :param fixed_valid_frames: 固定验证视频的帧数据列表。
    :param max_frames_per_video: 每个视频处理的最大帧数。
    :param fixed_start_indices: 固定的起点索引列表。
    :return: 平均 SSIM。
    """
    total_ssim = 0
    total_frames = 0

    # 缓存时空复杂度
    complexity_cache = {}

    for trace_file, start_idx, video_frames in zip(valid_files, fixed_start_indices, fixed_valid_frames):
        env.load_trace(trace_file)  # 加载轨迹文件
        env.current_bandwidth_idx = start_idx  # 设置固定起点
        print(f"Validating with trace: {trace_file}, start index: {start_idx}")

        # 缓存当前视频的时空复杂度
        if trace_file not in complexity_cache:
            space_complexities = []
            time_complexities = []
            previous_frame = np.zeros((env.frame_height, env.frame_width), dtype=np.uint8)

            for frame in video_frames[:max_frames_per_video]:
                # 如果 frame 是 PIL 图像对象，转换为 NumPy 数组
                if isinstance(frame, Image.Image):
                    frame = np.array(frame)

                # 转换为灰度图像并调整尺寸
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_frame = cv2.resize(gray_frame, (env.frame_width, env.frame_height))

                # 计算空间复杂度
                sobel_x = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(gray_frame, cv2.CV_64F, 0, 1, ksize=3)
                space_complexity = np.mean(np.sqrt(sobel_x**2 + sobel_y**2))

                # 计算时间复杂度
                time_complexity = np.std(np.abs(gray_frame - previous_frame))

                # 更新 previous_frame
                previous_frame = gray_frame

                # 缓存复杂度
                space_complexities.append(space_complexity)
                time_complexities.append(time_complexity)

            complexity_cache[trace_file] = (space_complexities, time_complexities)

        # 获取缓存的复杂度
        space_complexities, time_complexities = complexity_cache[trace_file]

        # 初始化状态
        state = np.zeros((S_LEN, S_INFO), dtype=np.float32)

        for step, frame in enumerate(video_frames[:max_frames_per_video]):
            # 使用 PPO 获取动作，选择概率最大的动作
            action_idx, _, _, _ = ppo.get_action(state, explore=False)

            # 模拟带宽更新
            env.current_bandwidth_idx = (env.current_bandwidth_idx + 1) % len(env.trace)
            bandwidth_mbps = env.trace[env.current_bandwidth_idx][1]

            # 模拟 SSIM 计算
            ae_model = env.models[env.model_keys[action_idx]]
            is_iframe = (ae_model.frame_counter % ae_model.gop == 0)
            ref_frame = None

            if not is_iframe:
                with env.reference_frame_lock:
                    ref_frame = env.last_valid_iframe_cache.get(env.model_keys[action_idx], None)

            size, eframe = encode_frame(ae_model, is_iframe, ref_frame, frame)

            # 模拟丢包率
            if not is_iframe:
                bandwidth_bps = bandwidth_mbps * 1e6
                available_bandwidth_per_frame = bandwidth_bps / 25
                frame_bits = size * 8
                loss_rate = max(0, (frame_bits - available_bandwidth_per_frame) / frame_bits) if frame_bits > 0 else 0.0
                if loss_rate > 0:
                    eframe.apply_loss(loss_rate, blocksize=1)
            else:
                loss_rate = 0.0

            # 解码帧
            try:
                decoded_frame = decode_frame(ae_model, eframe, ref_frame, 0)
            except Exception as e:
                if ref_frame is not None:
                    decoded_frame = ref_frame
                else:
                    decoded_frame = torch.zeros_like(to_tensor(frame))
                print(f"Decoding failed: {e}, using fallback")

            # 更新参考帧缓存
            with env.reference_frame_lock:
                if is_iframe:
                    env.last_valid_iframe_cache[env.model_keys[action_idx]] = copy.deepcopy(decoded_frame)

            # 计算 SSIM
            frame_tensor = to_tensor(frame).float() / 255.0
            decoded_frame_tensor = decoded_frame.float() / 255.0

            if frame_tensor.shape != decoded_frame_tensor.shape:
                decoded_frame_tensor = torch.nn.functional.interpolate(
                    decoded_frame_tensor.unsqueeze(0),
                    size=frame_tensor.shape[1:],
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)

            ssim_value = SSIM(frame_tensor, decoded_frame_tensor)
            ssim_value = np.log(1 / (1 - ssim_value + 1e-6))

            # 更新状态
            state = np.roll(state, -1, axis=0)
            state[-1] = [
                bandwidth_mbps,
                ssim_value,
                space_complexities[step],
                time_complexities[step]
            ]

            # 累计 SSIM 和帧数
            total_ssim += ssim_value
            total_frames += 1

    # 计算平均 SSIM
    avg_ssim = total_ssim / total_frames if total_frames > 0 else 0
    print(f"Validation completed. Average SSIM: {avg_ssim:.4f}")
    return avg_ssim