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

def init_models():
    GRACE_MODEL = "models/grace"
    models = {
        2048: AEModel(None, GraceInterface({"path": f"{GRACE_MODEL}/2048_freeze.model"})),
        4096: AEModel(None, GraceInterface({"path": f"{GRACE_MODEL}/4096_freeze.model"})),
        6144: AEModel(None, GraceInterface({"path": f"{GRACE_MODEL}/6144_freeze.model"})),
        8192: AEModel(None, GraceInterface({"path": f"{GRACE_MODEL}/8192_freeze.model"})),
        12288: AEModel(None, GraceInterface({"path": f"{GRACE_MODEL}/12288_freeze.model"})),
        16384: AEModel(None, GraceInterface({"path": f"{GRACE_MODEL}/16384_freeze.model"})),
    }
    return models

models = init_models()

class GraceEnv(config):
    def __init__(self, config):
        super().__init__()
        self.min_val = config.min_val
        self.max_val = config.max_val
        self.reference_frame_lock = Lock()
        self.last_valid_iframe_cache = {}
        self.max_frame_bits = 0  # 初始化最大 frame_bits 值
        self.min_frame_bits = float('inf')  # 初始化最小 frame_bits 值
    
    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        return np.random.uniform(self.min_val, self.max_val)
    
    def calculate_loss_rate(self, bandwidth_kbps, frame_size_bytes):
        bandwidth_bits = bandwidth_kbps * 1000
        frame_bits = frame_size_bytes * 8
        frame_bits_kb = frame_bits / 1000  # 将 frame_bits 除以一千
        self.max_frame_bits = max(self.max_frame_bits, frame_bits_kb)  # 更新最大 frame_bits 值
        self.min_frame_bits = min(self.min_frame_bits, frame_bits_kb)  # 更新最小 frame_bits 值
        print("size:", frame_bits_kb)
        print("max size:", self.max_frame_bits)  # 打印最大 frame_bits 值
        print("min size:", self.min_frame_bits)  # 打印最小 frame_bits 值
        if frame_bits > bandwidth_bits:
            return (frame_bits - bandwidth_bits) / frame_bits
        return 0
    
    def reward(self, model_id, state_kbps, frame, is_training=True):
        ae_model = models[model_id]
        is_iframe = (ae_model.frame_counter % ae_model.gop == 0)
        ref_frame = None
        if not is_iframe:
            with self.reference_frame_lock:
                ref_frame = self.last_valid_iframe_cache.get(model_id, None)
    
        # 确保输入的 `frame` 是 numpy.ndarray（假设外部传入的是 PIL.Image）
        if isinstance(frame, Image.Image):
            frame_np = np.array(frame)  # 将 PIL.Image 转换为 numpy.ndarray
        else:
            frame_np = frame  # 例如：frame 是 [H, W, C] 的 numpy 数组，值范围 [0, 255]
    
        # 编码当前帧，生成 eframe 和 size
        size, eframe = encode_frame(ae_model, is_iframe, ref_frame, frame_np)
    
        if not is_iframe:
            loss_rate = self.calculate_loss_rate(state_kbps, size)
            if loss_rate > 0:
                eframe.apply_loss(loss_rate, blocksize=1)
    
        try:
            decoded_frame = decode_frame(ae_model, eframe, ref_frame, 0)
        except Exception as e:
            if ref_frame is not None:
                decoded_frame = ref_frame
            else:
                decoded_frame = torch.zeros_like(to_tensor(frame))
            print(f"Decoding failed: {e}, using fallback")
    
        with self.reference_frame_lock:
            if is_iframe:
                self.last_valid_iframe_cache[model_id] = copy.deepcopy(decoded_frame)
    
        # 确保两个图像具有相同的 dtype 和值范围
        frame_tensor = to_tensor(frame).float() / 255.0
        decoded_frame_tensor = decoded_frame.float() / 255.0
        
        # 确保两个图像的 dtype 和维度相同
        frame_tensor = frame_tensor.type_as(decoded_frame_tensor)
        
        # 确保两个图像的维度相同
        if frame_tensor.shape != decoded_frame_tensor.shape:
            decoded_frame_tensor = torch.nn.functional.interpolate(decoded_frame_tensor.unsqueeze(0), size=frame_tensor.shape[1:], mode='bilinear', align_corners=False).squeeze(0)
        
        print(f"frame_tensor shape: {frame_tensor.shape}, decoded_frame_tensor shape: {decoded_frame_tensor.shape}")
        
        ssim_value = SSIM(frame_tensor, decoded_frame_tensor)
        
    
        reward = ssim_value
    
        return reward

