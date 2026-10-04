import torch.nn as nn


class QNetwork(nn.Module):
    """用于近似 Q 函数的深度神经网络。"""

    def __init__(self, state_dim, action_dim, hidden_layers, activation="relu"):
        super().__init__()
        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU())
            else:
                raise ValueError(f"Unsupported activation: {activation}")
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
