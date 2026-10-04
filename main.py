from dqn_cartpole.config import Config
from dqn_cartpole.train import train


if __name__ == "__main__":
    config = Config()
    train(config)
