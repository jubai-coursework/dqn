import random
from collections import deque

import numpy as np
import torch


class ReplayBuffer:
    """固定容量的经验回放池。"""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """存储一条经验。"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size, device):
        """随机采样一个批次，并转换为 PyTorch 张量。"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=device)
        actions = torch.as_tensor(np.asarray(actions), dtype=torch.int64, device=device).unsqueeze(1)
        rewards = torch.as_tensor(np.asarray(rewards), dtype=torch.float32, device=device).unsqueeze(1)
        next_states = torch.as_tensor(np.asarray(next_states), dtype=torch.float32, device=device)
        dones = torch.as_tensor(np.asarray(dones), dtype=torch.float32, device=device).unsqueeze(1)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
