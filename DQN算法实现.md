---

## 一、项目结构总览

```
dqn_cartpole/
├── config.py          # 所有超参数集中管理
├── replay_buffer.py   # 经验回放池类
├── networks.py        # Q网络（策略网络和目标网络共用结构）
├── agent.py           # DQN智能体，包含动作选择、学习更新、目标网络同步
├── train.py           # 训练主循环、测试、可视化
└── main.py            # 入口脚本，调用训练流程
```

下面我们逐个文件实现，并解释设计思路。

---

## 二、配置文件 `config.py`

把所有可调参数集中在一个文件里，修改超参数时不用满世界找。

```python
# config.py
import torch

class Config:
    # 环境参数
    env_name = "CartPole-v1"
    state_dim = 4
    action_dim = 2

    # 随机种子
    seed = 42

    # 神经网络结构
    hidden_layers = [128, 128]   # 两个隐藏层，每层128神经元
    activation = "relu"

    # 训练参数
    num_episodes = 500           # 训练回合数
    buffer_capacity = 10000      # 经验回放池容量
    batch_size = 64              # 批量采样大小
    learning_rate = 1e-3         # 学习率
    gamma = 0.99                 # 折扣因子

    # ε-greedy 探索策略
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995        # 每回合衰减系数（也可按步衰减）

    # 目标网络更新
    target_update_freq = 100     # 每100步硬更新一次（也可用软更新）
    tau = 0.005                  # 软更新系数（如果选择软更新）

    # 设备选择（GPU加速）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 可视化与日志
    render_train = False         # 训练时是否渲染（通常关闭以加速）
    render_test = True           # 测试时是否渲染
    log_interval = 10            # 每多少回合打印一次信息
    save_model = True            # 是否保存最终模型
    model_save_path = "dqn_cartpole.pth"

    # 滑动平均窗口
    smoothing_window = 20
```

**讲解：**
- `hidden_layers` 用一个列表表示，方便以后改成三层或调整神经元数量。
- 设备选择自动检测CUDA，代码可以在有GPU的机器上跑得更快。
- 参数都有注释，一目了然。

---

## 三、经验回放池 `replay_buffer.py`

经验回放池负责存储和随机采样。我们用 `collections.deque` 实现固定容量队列，并提供 `push` 和 `sample` 方法。

```python
# replay_buffer.py
import random
from collections import deque
import torch
import numpy as np

class ReplayBuffer:
    """固定容量的经验回放池"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """存储一条经验"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size, device):
        """随机采样一个批次，并转换为PyTorch张量"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 转换为张量并移到指定设备
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(device)  # shape: [batch,1]
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
```

**讲解：**
- `deque(maxlen=capacity)` 自动维护固定大小，满了以后最旧的数据会被丢弃。
- `sample` 返回的数据都已经转成PyTorch张量，并移动到 `device` 上，后续计算直接可用。
- `actions` 和 `dones` 使用 `LongTensor` 和 `FloatTensor` 分别对应整型和浮点型，注意形状保持 `[batch,1]`。

---

## 四、神经网络定义 `networks.py`

策略网络和目标网络结构完全一样，我们只定义一个 `QNetwork` 类，然后在 `Agent` 里实例化两次即可。

```python
# networks.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """用于近似Q函数的深度神经网络"""
    def __init__(self, state_dim, action_dim, hidden_layers, activation='relu'):
        super(QNetwork, self).__init__()
        layers = []
        input_dim = state_dim

        # 根据hidden_layers列表动态构建隐藏层
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unsupported activation: {activation}")
            input_dim = hidden_dim

        # 输出层，无激活函数
        layers.append(nn.Linear(input_dim, action_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
```

**讲解：**
- 通过循环 `hidden_layers` 列表来动态创建网络，修改层数或神经元数只需改配置文件，代码零变动。
- 输出层没有激活函数，因为Q值可以是任意实数。
- 如果你喜欢，也可以改用 `LeakyReLU` 等，只需在配置里增加相应选项。

---

## 五、DQN智能体 `agent.py`

智能体封装了所有与算法相关的行为：
- 根据 ε-greedy 选择动作
- 调用经验池存储数据
- 从经验池采样并更新网络参数（学习）
- 同步目标网络

```python
# agent.py
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from networks import QNetwork
from replay_buffer import ReplayBuffer
import copy

class DQNAgent:
    def __init__(self, config):
        self.config = config
        self.device = config.device

        # 初始化策略网络和目标网络
        self.policy_net = QNetwork(
            config.state_dim,
            config.action_dim,
            config.hidden_layers,
            config.activation
        ).to(self.device)

        self.target_net = QNetwork(
            config.state_dim,
            config.action_dim,
            config.hidden_layers,
            config.activation
        ).to(self.device)

        # 目标网络初始参数与策略网络相同
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # 目标网络只做前向传播，不训练

        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.learning_rate)
        self.loss_fn = nn.MSELoss()

        # 经验回放池
        self.memory = ReplayBuffer(config.buffer_capacity)

        # 探索率
        self.epsilon = config.epsilon_start
        self.epsilon_end = config.epsilon_end
        self.epsilon_decay = config.epsilon_decay

        # 步数计数器（用于目标网络硬更新）
        self.step_count = 0

    def select_action(self, state, evaluate=False):
        """
        根据当前状态选择动作。
        evaluate=True 时关闭探索（测试模式）。
        """
        if not evaluate and np.random.random() < self.epsilon:
            # 探索：随机选择一个动作
            return np.random.randint(self.config.action_dim)
        else:
            # 利用：选择当前Q值最大的动作
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            return q_values.argmax(dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        """存储一条经验到回放池"""
        self.memory.push(state, action, reward, next_state, done)

    def learn(self):
        """从经验池采样一个批次，执行一次梯度下降更新"""
        if len(self.memory) < self.config.batch_size:
            return  # 样本不足时不学习

        # 采样
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.config.batch_size, self.device
        )

        # 计算当前Q值
        q_values = self.policy_net(states)  # [batch, action_dim]
        q_pred = q_values.gather(1, actions)  # [batch, 1] 只取实际执行的动作的Q值

        # 计算目标Q值（使用目标网络）
        with torch.no_grad():
            next_q_values = self.target_net(next_states)  # [batch, action_dim]
            max_next_q = next_q_values.max(dim=1, keepdim=True)[0]  # [batch, 1]
            target_q = rewards + self.config.gamma * max_next_q * (1 - dones)

        # 计算损失
        loss = self.loss_fn(q_pred, target_q)

        # 反向传播与参数更新
        self.optimizer.zero_grad()
        loss.backward()
        # 可选：梯度裁剪，防止梯度爆炸
        # torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # 更新步数计数，并判断是否需要同步目标网络
        self.step_count += 1
        if self.config.target_update_freq > 0:  # 硬更新模式
            if self.step_count % self.config.target_update_freq == 0:
                self.update_target_network_hard()
        else:
            # 软更新模式（如果target_update_freq设为0则启用软更新）
            self.update_target_network_soft()

        return loss.item()

    def update_target_network_hard(self):
        """硬更新：直接复制策略网络参数到目标网络"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_target_network_soft(self):
        """软更新：θ_target = τ * θ_policy + (1-τ) * θ_target"""
        for target_param, policy_param in zip(self.target_net.parameters(),
                                              self.policy_net.parameters()):
            target_param.data.copy_(
                self.config.tau * policy_param.data + (1 - self.config.tau) * target_param.data
            )

    def decay_epsilon(self):
        """每回合结束后衰减探索率"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
```

**讲解：**
- `select_action` 通过 `evaluate` 参数控制是否探索，测试时设为 `True` 即可。
- `learn` 方法实现了完整的DQN更新流程，返回损失值以便监控。
- 目标网络更新支持硬更新和软更新两种模式：如果 `target_update_freq > 0` 则硬更新；否则用软更新。这是通过配置参数的简单逻辑判断。
- `decay_epsilon` 每回合调用一次，按指数衰减探索率。

---

## 六、训练与测试流程 `train.py`

这里包含训练主循环、测试函数以及可视化绘图。为了清晰，我们把训练和测试分开。

```python
# train.py
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from agent import DQNAgent
from config import Config

def test_agent(agent, config, render=True):
    """测试智能体一个回合，返回总奖励"""
    env = gym.make(config.env_name, render_mode='human' if render else None)
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.select_action(state, evaluate=True)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        total_reward += reward

    env.close()
    return total_reward

def moving_average(data, window_size):
    """计算滑动平均"""
    if len(data) < window_size:
        return data
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window_size + 1)
        smoothed.append(np.mean(data[start:i+1]))
    return smoothed

def train(config):
    # 设置随机种子
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # 创建环境和智能体
    env = gym.make(config.env_name)
    agent = DQNAgent(config)

    # 记录数据
    episode_rewards = []
    losses = []
    recent_rewards = deque(maxlen=100)  # 用于计算最近100回合平均奖励

    for episode in range(1, config.num_episodes + 1):
        state, _ = env.reset()
        total_reward = 0
        done = False
        episode_loss = []

        while not done:
            # 选择动作
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 存储经验
            agent.store_transition(state, action, reward, next_state, done)

            # 学习更新
            loss = agent.learn()
            if loss is not None:
                episode_loss.append(loss)

            state = next_state
            total_reward += reward

        # 回合结束后的处理
        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        recent_rewards.append(total_reward)

        if episode_loss:
            losses.append(np.mean(episode_loss))
        else:
            losses.append(0.0)

        # 打印训练信息
        if episode % config.log_interval == 0:
            avg_reward = np.mean(recent_rewards) if recent_rewards else total_reward
            print(f"Episode {episode:4d} | "
                  f"Reward: {total_reward:3.0f} | "
                  f"Avg100: {avg_reward:5.1f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Loss: {losses[-1]:.4f}")

        # 可选：提前停止条件（当最近100回合平均奖励>475，CartPole可认为已收敛）
        if len(recent_rewards) == 100 and np.mean(recent_rewards) >= 475:
            print(f"Solved in {episode} episodes!")
            break

    env.close()

    # 保存模型
    if config.save_model:
        agent.save_model(config.model_save_path)
        print(f"Model saved to {config.model_save_path}")

    # 测试训练好的模型
    print("\nTesting the trained agent...")
    test_reward = test_agent(agent, config, render=config.render_test)
    print(f"Test episode reward: {test_reward}")

    # 绘制奖励曲线
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, alpha=0.5, label='Raw')
    smoothed = moving_average(episode_rewards, config.smoothing_window)
    plt.plot(smoothed, label=f'Smoothed (win={config.smoothing_window})')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Reward Curve')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(losses)
    plt.xlabel('Episode')
    plt.ylabel('Average Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    config = Config()
    train(config)
```

**讲解：**
- `test_agent` 函数单独用于评估，可以选择是否渲染。
- 训练循环内，每一步都调用 `agent.learn()`，当经验池样本不足时不会更新。
- 记录并绘制奖励曲线和损失曲线，滑动平均让趋势更清晰。
- 加入了提前收敛判断：当最近100回合平均奖励达到475以上（CartPole-v1的最大奖励为500），可认为已学会，提前结束训练。

---

## 七、入口脚本 `main.py`

为了让项目启动更清晰，我们写一个简单的入口。

```python
# main.py
from train import train
from config import Config

if __name__ == "__main__":
    config = Config()
    # 你可以在这里临时修改某些参数，比如：
    # config.num_episodes = 600
    train(config)
```

---

## 八、运行与结果预期

## **预期现象**：

- 训练初期奖励很低（十几步就倒），随着训练进行，总奖励逐步上升。
- 大约300~500回合后，智能体能稳定达到200+，最后收敛到500附近。
- 终端会定期打印训练信息，最后弹出一个窗口渲染测试画面，可以看到小车左右移动保持杆子竖直。

---

