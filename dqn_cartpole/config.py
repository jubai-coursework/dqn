import torch


class Config:
    # 环境参数
    env_name = "CartPole-v1"
    state_dim = 4
    action_dim = 2

    # 随机种子
    seed = 42

    # 神经网络结构
    hidden_layers = [128,64]
    activation = "relu"

    # 训练参数
    num_episodes = 600
    buffer_capacity = 20000
    batch_size = 32
    learning_rate = 1e-4
    gamma = 0.99

    # ε-greedy 探索策略
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.99

    # 目标网络更新
    target_update_freq = 100
    tau = 0.005

    # 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 可视化与日志
    render_train = False
    render_test = True
    log_interval = 10
    save_model = True
    model_save_path = "dqn_cartpole.pth"

    # 滑动平均窗口
    smoothing_window = 100
