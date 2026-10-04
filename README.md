# DQN CartPole

基于 DQN 的 CartPole-v1 实现，包含策略网络、目标网络、经验回放池、训练与测试流程。

## 结构

- `dqn_cartpole/config.py`：超参数配置
- `dqn_cartpole/replay_buffer.py`：经验回放池
- `dqn_cartpole/networks.py`：Q 网络
- `dqn_cartpole/agent.py`：DQN 智能体
- `dqn_cartpole/train.py`：训练与测试流程
- `main.py`：入口脚本

## 安装

```bash
pip install -r requirements.txt
```

## 运行

```bash
python main.py
```
