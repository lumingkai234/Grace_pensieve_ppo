from ppo_config import config
import numpy as np
from scipy.stats import truncnorm
import torch
from torchvision.transforms.functional import to_tensor
from pytorch_msssim import ssim
from grace.grace_gpu_interface import GraceInterface, GraceBasicCode
from grace_new import encode_frame, decode_frame, SSIM, AEModel  # 导入相关函数和类
import math 
# 初始化 models 字典
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

class grace_env(config):
    def __init__(self, config):
        #self.mean = config.mean
        #self.std = config.std
        self.min_val = config.min_val
        self.max_val = config.max_val

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        return np.random.uniform(self.min_val, self.max_val)
    
    def calculate_loss_rate(self, bandwidth, frame_size):
        """
        Calculate the loss rate based on the given bandwidth and frame size.
        """
        if frame_size > bandwidth:
            return (frame_size - bandwidth) / frame_size
        else:
            return 0

    def reward(self, model_id, state, frame):
        """
        Calculate the reward based on the model, state (bandwidth), and current frame.
        
        Parameters:
        - model_id: The model to be used for encoding/decoding (lambda value).
        - state: The current bandwidth limit.
        - frame: The current video frame to be processed.
        
        Returns:
        - reward: The calculated reward based on SSIM.
        """
        # Initialize the model
        ae_model = models[model_id]
        
        # Encode the frame
        is_iframe = (ae_model.frame_counter % ae_model.gop == 0)
        ref_frame = ae_model.reference_frame if not is_iframe else None
        size, eframe = encode_frame(ae_model, is_iframe, ref_frame, frame)
        #print(size)

        # Check if the frame size exceeds the bandwidth limit
        if size > state:
            # Calculate loss rate
            loss_rate = self.calculate_loss_rate(state, size)
            # Apply loss to the encoded frame
            if eframe.frame_type == "I":
                print("Error! Cannot add loss on I frame, it will cause huge error!")
            else:
                eframe.apply_loss(loss_rate, blocksize=1)
                print("1")
        
        # Decode the frame
        decoded_frame = decode_frame(ae_model, eframe, ref_frame, 0)
        
        # Update the reference frame
        if is_iframe:
            ae_model.reference_frame = decoded_frame
        
        # Calculate SSIM
        ssim_value = SSIM(to_tensor(frame), decoded_frame)
        
        # Reward is the SSIM value itself
        reward = 10*math.log10(1/(1-ssim_value))
        
        return reward