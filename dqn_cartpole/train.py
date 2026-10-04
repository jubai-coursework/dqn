from collections import deque

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from .agent import DQNAgent
from .config import Config
from .logger import TrainingLogger
from .visualizer import TrainingVisualizer


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def test_agent(agent, config, render=True):
    """测试智能体一个回合，返回总奖励。"""
    env = gym.make(config.env_name, render_mode="human" if render else None)
    state, _ = env.reset(seed=config.seed)
    total_reward = 0.0
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
    """计算滑动平均。"""
    if len(data) < window_size:
        return data

    smoothed = []
    for index in range(len(data)):
        start = max(0, index - window_size + 1)
        smoothed.append(float(np.mean(data[start: index + 1])))
    return smoothed


def train(config):
    set_seed(config.seed)

    # 初始化日志记录器和可视化器
    logger = TrainingLogger()
    visualizer = TrainingVisualizer()
    
    # 保存配置
    logger.save_config(config)
    print(f"实验日志目录: {logger.get_log_directory()}")

    env = gym.make(config.env_name)
    env.action_space.seed(config.seed)
    agent = DQNAgent(config)

    episode_rewards = []
    losses = []
    epsilons = []
    recent_rewards = deque(maxlen=100)
    best_avg_reward = 0.0

    # 用于计算成功率的滑动窗口
    success_window = deque(maxlen=100)

    for episode in range(1, config.num_episodes + 1):
        state, _ = env.reset(seed=config.seed + episode)
        total_reward = 0.0
        done = False
        episode_loss = []
        steps = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.learn()
            if loss is not None:
                episode_loss.append(loss)

            state = next_state
            total_reward += reward
            steps += 1

        agent.decay_epsilon()
        
        # 记录数据
        episode_rewards.append(total_reward)
        epsilons.append(agent.epsilon)
        avg_loss = float(np.mean(episode_loss)) if episode_loss else 0.0
        losses.append(avg_loss)
        recent_rewards.append(total_reward)
        
        # 判断是否成功（CartPole-v1成功标准：奖励>=475）
        is_success = total_reward >= 475
        success_window.append(is_success)
        
        # 计算最近100回合平均奖励
        avg_reward_100 = float(np.mean(recent_rewards)) if recent_rewards else total_reward

        # 记录回合日志
        logger.log_episode(
            episode=episode,
            total_reward=total_reward,
            avg_reward_100=avg_reward_100,
            epsilon=agent.epsilon,
            avg_loss=avg_loss,
            steps=steps,
            success=is_success
        )

        if episode % config.log_interval == 0:
            # 统计最近一段回合的平均回报，便于观察训练趋势是否稳定上升。
            avg_reward = float(np.mean(recent_rewards)) if recent_rewards else total_reward
            # 计算成功率
            success_rate = sum(success_window) / len(success_window) * 100 if success_window else 0.0
            
            # 打印当前回合、最近平均回报、探索率和损失，便于训练时快速判断收敛情况。
            print(
                f"Episode {episode:4d} | Reward: {total_reward:3.0f} | "
                f"Avg100: {avg_reward:5.1f} | SuccessRate: {success_rate:5.1f}% | "
                f"Epsilon: {agent.epsilon:.3f} | Loss: {losses[-1]:.4f}"
            )
            
            # 保存最佳模型（基于最近100回合平均奖励）
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                if config.save_model:
                    agent.save_model(config.model_save_path.replace(".pth", "_best.pth"))
                    print(f"New best model saved! Avg100: {best_avg_reward:.1f}")

        # 移除胜利条件，完成所有训练回合以追求更好的成绩

    env.close()

    if config.save_model:
        agent.save_model(config.model_save_path)
        print(f"Final model saved to {config.model_save_path}")
        print(f"Best model saved with Avg100: {best_avg_reward:.1f}")

    # 打印训练日志摘要
    logger.print_summary()

    print("\nTesting the trained agent...")
    test_reward = test_agent(agent, config, render=config.render_test)
    print(f"Test episode reward: {test_reward}")

    # 绘制并保存训练曲线
    print("\nGenerating training plots...")
    visualizer.plot_reward_curve(
        episode_rewards,
        smoothing_window=config.smoothing_window,
        title="DQN CartPole 训练奖励曲线",
        save_name="reward_curve.png"
    )
    
    visualizer.plot_success_rate_curve(
        episode_rewards,
        window_size=100,
        success_threshold=475,
        title="DQN CartPole 训练成功率曲线",
        save_name="success_rate_curve.png"
    )
    
    visualizer.plot_loss_curve(
        losses,
        smoothing_window=config.smoothing_window,
        title="DQN CartPole 训练损失曲线",
        save_name="loss_curve.png"
    )
    
    visualizer.plot_epsilon_curve(
        epsilons,
        title="DQN CartPole 探索率变化曲线",
        save_name="epsilon_curve.png"
    )
    
    visualizer.plot_summary(
        episode_rewards,
        losses,
        epsilons,
        smoothing_window=config.smoothing_window,
        window_size=100,
        success_threshold=475,
        save_name="summary_plot.png"
    )
    
    print(f"训练图表已保存到 {visualizer.save_dir} 目录")

    return episode_rewards, losses, epsilons


if __name__ == "__main__":
    config = Config()
    train(config)