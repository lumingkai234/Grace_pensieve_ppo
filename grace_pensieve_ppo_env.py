import copy
import math
from functools import lru_cache
from threading import Lock
from ppo_config import config
from grace_new import encode_frame, decode_frame, to_tensor, SSIM, AEModel, decode_with_loss
from grace.grace_gpu_interface import GraceInterface
import numpy as np
import torch
from prometheus_client import Gauge
from PIL import Image
import cv2
import time
from collections import deque
import os
import random

def init_models(scale_factor=0.5):
    GRACE_MODEL = "models/grace"
    models = {
        2048: AEModel(None, GraceInterface({"path": f"{GRACE_MODEL}/2048_freeze.model", "scale_factor": scale_factor})),
        4096: AEModel(None, GraceInterface({"path": f"{GRACE_MODEL}/4096_freeze.model", "scale_factor": scale_factor})),
        6144: AEModel(None, GraceInterface({"path": f"{GRACE_MODEL}/6144_freeze.model", "scale_factor": scale_factor})),
        8192: AEModel(None, GraceInterface({"path": f"{GRACE_MODEL}/8192_freeze.model", "scale_factor": scale_factor})),
        12288: AEModel(None, GraceInterface({"path": f"{GRACE_MODEL}/12288_freeze.model", "scale_factor": scale_factor})),
        16384: AEModel(None, GraceInterface({"path": f"{GRACE_MODEL}/16384_freeze.model", "scale_factor": scale_factor})),
    }
    return models

models = init_models(scale_factor=0.5)

class GraceEnv:
    def __init__(self, scale_factor=0.5):
        """
        初始化 GraceEnv 环境。
        :param scale_factor: 缩放因子。
        """
        # 初始化模型
        self.models = init_models(scale_factor=scale_factor)
        self.model_keys = list(self.models.keys())  # 获取模型键值列表

        # 初始化带宽、SSIM、时间复杂度、空间复杂度和帧大小历史记录
        self.bandwidth_history = deque(maxlen=5)
        self.ssim_history = deque(maxlen=5)
        self.time_complexity_history = deque(maxlen=5)
        self.space_complexity_history = deque(maxlen=5)
        self.frame_bits_history = deque(maxlen=5)  # 新增帧大小历史记录

        # 初始化轨迹相关属性
        self.trace = None
        self.bandwidth_data = []
        self.current_bandwidth_idx = 0

        # 视频帧尺寸
        self.frame_height = int(768 * scale_factor)
        self.frame_width = int(1280 * scale_factor)

        # 初始化状态
        self.previous_frame = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
        self.current_frame = 0
        self.previous_ssim = None

        # 初始化线程锁和参考帧缓存
        self.reference_frame_lock = Lock()  # 用于线程安全的参考帧缓存
        self.last_valid_iframe_cache = {}  # 缓存最近的有效 I 帧
        self.last_valid_iframe = None  # 全局参考帧

        # 初始化状态边界
        self.state_max = np.full(5, -np.inf, dtype=np.float32)  # 初始化为负无穷，状态维度增加到 5
        self.state_min = np.full(5, np.inf, dtype=np.float32)   # 初始化为正无穷

    def load_trace(self, trace):
        """
        加载轨迹数据或轨迹文件。
        :param trace: 轨迹数据（列表）或轨迹文件路径（字符串）。
        """
        if isinstance(trace, list):
            # 如果是轨迹数据，直接加载
            self.trace = trace
        elif isinstance(trace, str):
            # 如果是文件路径，从文件中加载
            try:
                with open(trace, "r") as f:
                    self.trace = []
                    for line in (f,trace):
                        timestamp, bandwidth = map(float, line.strip().split())
                        self.trace.append((timestamp, bandwidth))
            except Exception as e:
                raise ValueError(f"Failed to load trace file {trace}: {e}")
        else:
            raise TypeError("Trace must be a list or a file path (string).")

    @staticmethod
    def split_traces(cooked_traces_dir, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1, seed=42):
        """
        按比例划分带宽数据集为训练集、验证集和测试集。
        :param cooked_traces_dir: 带宽轨迹文件夹路径。
        :param train_ratio: 训练集比例。
        :param valid_ratio: 验证集比例。
        :param test_ratio: 测试集比例。
        :param seed: 随机种子。
        :return: 训练集、验证集、测试集的文件路径列表。
        """
        assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6, "数据集比例之和必须为 1"
        
        # 获取所有轨迹文件
        trace_files = [os.path.join(cooked_traces_dir, f) for f in os.listdir(cooked_traces_dir)]
        
        # 设置随机种子
        random.seed(seed)
        random.shuffle(trace_files)  # 随机打乱轨迹文件列表
        
        # 按比例划分数据集
        n = len(trace_files)
        train_end = int(n * train_ratio)
        valid_end = train_end + int(n * valid_ratio)
        
        train_files = trace_files[:train_end]
        valid_files = trace_files[train_end:valid_end]
        test_files = trace_files[valid_end:]
        
        return train_files, valid_files, test_files

    def set_random_start(self):
        """
        设置带宽轨迹的随机起点。
        """
        if not self.trace:
            print("Trace is empty:", self.trace)
            raise ValueError("Trace is empty. Please load a valid trace file first.")
        self.current_bandwidth_idx = random.randint(0, len(self.trace) - 1)
        print(f"Random start index set to: {self.current_bandwidth_idx}")

    def reset(self, preserve_history=True):
        """
        重置环境状态。
        :param preserve_history: 是否保留历史记录（如带宽历史、SSIM 历史等）。
        """
        # 重置上一帧和当前帧计数器
        self.previous_frame = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
        self.current_frame = 0  # 重置当前帧计数器

        # 重置全局参考帧
        self.last_valid_iframe = None

        if not preserve_history:
            # 重置轨迹索引
            self.current_bandwidth_idx = 0

            # 清空历史记录
            self.bandwidth_history.clear()
            self.ssim_history.clear()
            self.time_complexity_history.clear()
            self.space_complexity_history.clear()
            self.frame_bits_history.clear()  # 清空帧大小历史记录

            # 初始化历史记录为零
            for _ in range(5):
                self.bandwidth_history.append(0)
                self.ssim_history.append(0)
                self.time_complexity_history.append(0)
                self.space_complexity_history.append(0)
                self.frame_bits_history.append(0)  # 初始化帧大小为 0
      
    def step(self, frame, action_idx):
        """
        执行一步环境交互。
        :param frame: 当前视频帧 (numpy array)。
        :param action_idx: 动作索引 (int)。
        :return: state (状态), reward (奖励), done (是否结束), ssim_value (原始 SSIM 值)。
        """
        # 更新带宽状态
        self.current_bandwidth_idx = (self.current_bandwidth_idx + 1) % len(self.trace)
        bandwidth_mbps = self.trace[self.current_bandwidth_idx]  # 当前带宽 (Mbps)
        self.bandwidth_history.append(bandwidth_mbps)

        # 将 action_idx 映射到模型键值
        if action_idx < 0 or action_idx >= len(self.model_keys):
            raise ValueError(f"Invalid action index: {action_idx}. Must be in range [0, {len(self.model_keys) - 1}].")
        
        model_id = self.model_keys[action_idx]  # 根据索引获取模型键值
        ae_model = self.models[model_id]

        # 检查并转换 frame 为 NumPy 数组
        if frame is None:
            raise ValueError("Invalid frame passed to env.step. Expected a NumPy array.")
        if not isinstance(frame, np.ndarray):
            frame = np.array(frame, dtype=np.uint8)

        # 转换为灰度图像并调整尺寸（用于复杂度计算）
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.resize(gray_frame, (self.frame_width, self.frame_height))

        # 计算时间复杂度和空间复杂度
        sobel_x = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_frame, cv2.CV_64F, 0, 1, ksize=3)
        space_complexity = np.mean(np.sqrt(sobel_x**2 + sobel_y**2))
        time_complexity = np.std(np.abs(gray_frame - self.previous_frame))

        # 更新 previous_frame
        self.previous_frame = gray_frame

        # 更新复杂度历史
        self.space_complexity_history.append(space_complexity)
        self.time_complexity_history.append(time_complexity)

        # 判断是否为 I 帧
        is_iframe = (self.current_frame == 0) or (ae_model.frame_counter % ae_model.gop == 0)  # 强制第一帧为 I 帧
        ref_frame = None

        # 如果是 P 帧，获取参考帧
        if not is_iframe:
            with self.reference_frame_lock:
                ref_frame = self.last_valid_iframe
            if ref_frame is None:
                raise ValueError("Reference frame is None! Cannot decode P-frame without a valid reference frame.")

        # 编码当前帧，生成 eframe 和 size
        size, eframe = encode_frame(ae_model, is_iframe, ref_frame, frame)  # 使用彩色帧进行编码

        # 丢包处理逻辑
        bandwidth_bps = bandwidth_mbps * 1e6  # 带宽转换为 bps
        available_bandwidth_per_frame = bandwidth_bps / 10 # 每帧可用的带宽，帧率为 25fps
        frame_bits = size * 8  # 帧大小转换为 bits
        self.frame_bits_history.append(frame_bits)  # 更新帧大小历史记录
        loss_rate = max(0, (frame_bits - available_bandwidth_per_frame) / frame_bits) if frame_bits > 0 else 0.0
        print(f"Calculated loss rate: {loss_rate}")

        # 调用 decode_frame 进行解码
        decoded_frame = decode_frame(ae_model, eframe, ref_frame, loss_rate, use_IND=False, lamda=int(model_id))

        # 更新参考帧缓存，无论是 I 帧还是 P 帧
        with self.reference_frame_lock:
            self.last_valid_iframe = copy.deepcopy(decoded_frame)
            print(f"Updated reference frame")

        # 确保两个图像具有相同的 dtype 和值范围
        frame_tensor = to_tensor(frame).float()  # 使用彩色帧
        decoded_frame_tensor = decoded_frame.float()

        # 计算 SSIM
        ssim_value = SSIM(frame_tensor, decoded_frame_tensor)
        print(f"SSIM value: {ssim_value:.4f}")
        self.ssim_history.append(ssim_value)

        # 构造状态
        state = self._get_state()

        # 奖励函数：将 SSIM 减去 0.9 后乘以 10
        reward = (ssim_value - 0.9) * 10

        # 检查是否结束
        done = self.current_bandwidth_idx == len(self.trace) - 1

        # 更新当前帧计数器
        self.current_frame += 1

        return state, reward, done, ssim_value

    def _update_state_bounds(self, state):
        """
        更新状态的最大值和最小值。
        :param state: 当前状态，形状为 (4,)。
        """
        self.state_max = np.maximum(self.state_max, state)
        self.state_min = np.minimum(self.state_min, state)

        print("state_max:", self.state_max)
        print("state_min:", self.state_min)

    def _get_state(self):
        """
        获取当前帧的状态特征。
        :return: 当前帧的状态特征，包含带宽、SSIM、空间复杂度、时间复杂度、帧大小。
        """
        state = np.array([
            self.bandwidth_history[-1] if self.bandwidth_history else 0,  # 当前帧的带宽
            self.ssim_history[-1] if self.ssim_history else 0,           # 当前帧的 SSIM
            self.space_complexity_history[-1] if self.space_complexity_history else 0,  # 当前帧的空间复杂度
            self.time_complexity_history[-1] if self.time_complexity_history else 0,    # 当前帧的时间复杂度
            self.frame_bits_history[-1] if self.frame_bits_history else 0               # 当前帧的大小 (bits)
        ], dtype=np.float32)

        # 更新状态的最大值和最小值
        self._update_state_bounds(state)

        return state

    def is_trace_finished(self):
        """
        检查当前轨迹是否已经结束。
        :return: 如果当前轨迹已经结束，则返回 True，否则返回 False。
        """
        return self.current_bandwidth_idx == len(self.trace) - 1
