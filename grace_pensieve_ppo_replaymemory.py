import numpy as np


class ReplayMemory:
    def __init__(self):
        self.frames = []  # 存储每一步的单帧特征
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
    
    def clear(self):
        del self.frames[:]
        del self.actions[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.dones[:]

    def add(self, frame, action, log_prob, reward, done):
        """
        添加单帧特征到 ReplayMemory。
        :param frame: 当前帧的特征 (1D array)。
        :param action: 动作。
        :param log_prob: 动作的 log 概率。
        :param reward: 奖励。
        :param done: 是否结束。
        """
        self.frames.append(frame)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)

    def sample(self, batch_size, seq_len):
        """
        从 ReplayMemory 中采样连续的状态序列。
        :param batch_size: 批量大小。
        :param seq_len: 时间序列长度。
        :return: 状态序列、动作、log 概率、奖励、done 标志。
        """
        indices = np.random.choice(len(self.frames) - seq_len, batch_size, replace=False)
        states_seq = [np.array(self.frames[i:i + seq_len]) for i in indices]
        actions = [self.actions[i + seq_len - 1] for i in indices]
        log_probs = [self.log_probs[i + seq_len - 1] for i in indices]
        rewards = [self.rewards[i + seq_len - 1] for i in indices]
        dones = [self.dones[i + seq_len - 1] for i in indices]
        return (
            np.array(states_seq),  # (batch_size, S_INFO, S_LEN)
            np.array(actions),
            np.array(log_probs),
            np.array(rewards),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.frames)