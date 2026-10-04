import gymnasium as gym
import sys
import os

# 支持直接运行和作为模块导入
try:
    from .agent import DQNAgent
    from .config import Config
except ImportError:
    # 如果是直接运行，添加父目录到路径
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from dqn_cartpole.agent import DQNAgent
    from dqn_cartpole.config import Config


def test_trained_model(model_path, config=None, render=True, num_episodes=10):
    """
    使用保存的模型进行测试。
    
    参数：
        model_path: 模型文件路径
        config: 配置对象，若为None则使用默认配置
        render: 是否渲染环境
        num_episodes: 测试回合数
    """
    if config is None:
        config = Config()
    
    # 创建环境和智能体
    env = gym.make(config.env_name, render_mode="human" if render else None)
    agent = DQNAgent(config)
    
    # 加载训练好的模型
    agent.load_model(model_path)
    print(f"已加载模型: {model_path}")
    
    total_rewards = []
    
    for episode in range(1, num_episodes + 1):
        state, _ = env.reset(seed=config.seed + episode)
        total_reward = 0.0
        done = False
        
        while not done:
            # 使用评估模式选择动作（不进行探索）
            action = agent.select_action(state, evaluate=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward
        
        total_rewards.append(total_reward)
        print(f"测试回合 {episode:2d} | 奖励: {total_reward:.0f}")
    
    env.close()
    
    # 统计结果
    avg_reward = sum(total_rewards) / num_episodes
    max_reward = max(total_rewards)
    min_reward = min(total_rewards)
    
    print("\n" + "=" * 40)
    print(f"测试结果 (共 {num_episodes} 回合):")
    print(f"平均奖励: {avg_reward:.2f}")
    print(f"最高奖励: {max_reward:.0f}")
    print(f"最低奖励: {min_reward:.0f}")
    print("=" * 40)
    
    return total_rewards


if __name__ == "__main__":
    config = Config()
    # 优先使用最佳模型，如果不存在则使用最终模型
    best_model_path = config.model_save_path.replace(".pth", "_best.pth")
    import os
    model_path = best_model_path if os.path.exists(best_model_path) else config.model_save_path
    
    test_trained_model(
        model_path=model_path,
        config=config,
        render=True,
        num_episodes=5
    )