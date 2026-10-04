import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .networks import QNetwork
from .replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(self, config):
        self.config = config
        self.device = config.device

        self.policy_net = QNetwork(
            config.state_dim,
            config.action_dim,
            config.hidden_layers,
            config.activation,
        ).to(self.device)

        self.target_net = QNetwork(
            config.state_dim,
            config.action_dim,
            config.hidden_layers,
            config.activation,
        ).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.learning_rate)
        self.loss_fn = nn.MSELoss()
        self.memory = ReplayBuffer(config.buffer_capacity)

        self.epsilon = config.epsilon_start
        self.epsilon_end = config.epsilon_end
        self.epsilon_decay = config.epsilon_decay
        self.step_count = 0

    def select_action(self, state, evaluate=False):
        """根据当前状态选择动作。"""
        if not evaluate and np.random.random() < self.epsilon:
            return np.random.randint(self.config.action_dim)

        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return q_values.argmax(dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def learn(self):
        """从经验池采样一个批次，执行一次梯度下降更新。"""
        if len(self.memory) < self.config.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(
            self.config.batch_size, self.device
        )

        q_values = self.policy_net(states)
        q_pred = q_values.gather(1, actions)

        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            max_next_q = next_q_values.max(dim=1, keepdim=True)[0]
            target_q = rewards + self.config.gamma * max_next_q * (1.0 - dones)

        loss = self.loss_fn(q_pred, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        if self.config.target_update_freq > 0:
            if self.step_count % self.config.target_update_freq == 0:
                self.update_target_network_hard()
        else:
            self.update_target_network_soft()

        return loss.item()

    def update_target_network_hard(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_target_network_soft(self):
        tau = self.config.tau
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path):
        state_dict = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(self.policy_net.state_dict())
