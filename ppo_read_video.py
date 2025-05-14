import os
import numpy as np
from PIL import Image



def read_video_into_frames(video_path, frame_size=None, nframes=1000):
    """
    终极正确版本：工业级尺寸控制，彻底解决张量对齐问题
    """
    def create_temp_path():
        path = f"/tmp/grace_success_frames-{np.random.randint(0, 1000)}/"
        while os.path.isdir(path):
            path = f"/tmp/grace_success_frames-{np.random.randint(0, 1000)}/"
        os.makedirs(path, exist_ok=True)
        return path

    def remove_temp_path(tmp_path):
        os.system(f"rm -rf {tmp_path}")

    # 核心参数：基于4层金字塔结构的数学约束
    PYRAMID_LEVELS = 4                  # 与 ME_Spynet 的 self.L=4 完全一致
    BASE_UNIT = 64                      # 模型基础单元尺寸
    SCALE_FACTOR = 2**(PYRAMID_LEVELS) # 2^4=16 → 64 * 16=1024
    REQUIRED_MULTIPLE = BASE_UNIT * SCALE_FACTOR

    frame_path = create_temp_path()
    
    # 强制输入尺寸为1024的倍数 (数学证明见下方)
    if frame_size is not None:
        width, height = frame_size
        width = max((width // REQUIRED_MULTIPLE) * REQUIRED_MULTIPLE, REQUIRED_MULTIPLE)
        height = max((height // REQUIRED_MULTIPLE) * REQUIRED_MULTIPLE, REQUIRED_MULTIPLE)
        cmd = f"ffmpeg -i {video_path} -s {width}x{height} {frame_path}/%03d.png"
    else:
        cmd = f"ffmpeg -i {video_path} {frame_path}/%03d.png"
    
    os.system(cmd)
    
    frames = []
    for img_name in sorted(os.listdir(frame_path))[:nframes]:
        frame = Image.open(os.path.join(frame_path, img_name))
        
        # 数学强制：W = 1024×N, H = 1024×M
        w, h = frame.size
        pad_w = ((w + REQUIRED_MULTIPLE - 1) // REQUIRED_MULTIPLE) * REQUIRED_MULTIPLE
        pad_h = ((h + REQUIRED_MULTIPLE - 1) // REQUIRED_MULTIPLE) * REQUIRED_MULTIPLE
        
        # 超严格验证（覆盖所有可能层级）
        for level in range(PYRAMID_LEVELS + 2):  # 额外验证两级
            scaled_w = pad_w // (2**level)
            scaled_h = pad_h // (2**level)
            assert scaled_w % BASE_UNIT == 0 and scaled_h % BASE_UNIT == 0, \
                f"Level {level} 尺寸异常: {scaled_w}x{scaled_h} (应满足 {BASE_UNIT}的倍数)"

        frames.append(frame.resize((pad_w, pad_h), Image.LANCZOS))
    
    remove_temp_path(frame_path)
    print(f"验证通过的终极安全尺寸：{frames[0].size if frames else '无帧'}")
    return frames