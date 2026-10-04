import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


class TrainingVisualizer:
    """训练数据可视化器，用于绘制各种训练曲线"""
    
    def __init__(self, save_dir="plots"):
        """
        初始化可视化器
        
        参数：
            save_dir: 图表保存目录
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 设置中文字体（避免中文乱码）
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_reward_curve(self, episode_rewards, smoothing_window=20, title="训练奖励曲线", 
                         save_name="reward_curve.png", show=True):
        """
        绘制训练奖励曲线
        
        参数：
            episode_rewards: 各回合奖励列表
            smoothing_window: 滑动平均窗口大小
            title: 图表标题
            save_name: 保存文件名
            show: 是否显示图表
        """
        plt.figure(figsize=(10, 6))
        
        # 原始奖励曲线
        plt.plot(episode_rewards, alpha=0.5, label='原始奖励', color='#1f77b4')
        
        # 滑动平均奖励曲线
        if smoothing_window > 1:
            smoothed = self._moving_average(episode_rewards, smoothing_window)
            plt.plot(smoothed, label=f'滑动平均 (窗口={smoothing_window})', 
                     color='#ff7f0e', linewidth=2)
        
        # 成功阈值线（CartPole-v1的成功标准）
        plt.axhline(y=475, color='#d62728', linestyle='--', label='成功阈值 (475)')
        
        plt.xlabel('训练回合', fontsize=12)
        plt.ylabel('总奖励', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xlim(0, len(episode_rewards))
        plt.ylim(0, 550)
        
        # 保存图表
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        plt.close()
    
    def plot_success_rate_curve(self, episode_rewards, window_size=100, success_threshold=475,
                               title="训练成功率曲线", save_name="success_rate_curve.png", show=True):
        """
        绘制训练成功率曲线
        
        参数：
            episode_rewards: 各回合奖励列表
            window_size: 计算成功率的滑动窗口大小
            success_threshold: 成功阈值
            title: 图表标题
            save_name: 保存文件名
            show: 是否显示图表
        """
        success_rates = []
        n_episodes = len(episode_rewards)
        
        for i in range(n_episodes):
            start = max(0, i - window_size + 1)
            window_rewards = episode_rewards[start:i+1]
            success_count = sum(1 for r in window_rewards if r >= success_threshold)
            success_rate = success_count / len(window_rewards) * 100
            success_rates.append(success_rate)
        
        plt.figure(figsize=(10, 6))
        
        plt.plot(success_rates, label=f'成功率 (窗口={window_size})', 
                 color='#2ca02c', linewidth=2)
        
        # 成功率100%的参考线
        plt.axhline(y=100, color='#d62728', linestyle='--', label='100% 成功')
        
        plt.xlabel('训练回合', fontsize=12)
        plt.ylabel('成功率 (%)', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xlim(0, n_episodes)
        plt.ylim(0, 105)
        
        # 保存图表
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        plt.close()
    
    def plot_loss_curve(self, losses, smoothing_window=20, title="训练损失曲线",
                       save_name="loss_curve.png", show=True):
        """
        绘制训练损失曲线
        
        参数：
            losses: 各回合损失列表
            smoothing_window: 滑动平均窗口大小
            title: 图表标题
            save_name: 保存文件名
            show: 是否显示图表
        """
        plt.figure(figsize=(10, 6))
        
        # 原始损失曲线
        plt.plot(losses, alpha=0.5, label='原始损失', color='#9467bd')
        
        # 滑动平均损失曲线
        if smoothing_window > 1:
            smoothed = self._moving_average(losses, smoothing_window)
            plt.plot(smoothed, label=f'滑动平均 (窗口={smoothing_window})', 
                     color='#8c564b', linewidth=2)
        
        plt.xlabel('训练回合', fontsize=12)
        plt.ylabel('平均损失', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xlim(0, len(losses))
        
        # 保存图表
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        plt.close()
    
    def plot_epsilon_curve(self, epsilons, title="探索率变化曲线",
                          save_name="epsilon_curve.png", show=True):
        """
        绘制探索率变化曲线
        
        参数：
            epsilons: 各回合探索率列表
            title: 图表标题
            save_name: 保存文件名
            show: 是否显示图表
        """
        plt.figure(figsize=(10, 6))
        
        plt.plot(epsilons, label='探索率 ε', color='#17becf', linewidth=2)
        
        plt.xlabel('训练回合', fontsize=12)
        plt.ylabel('探索率', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xlim(0, len(epsilons))
        plt.ylim(0, 1.05)
        
        # 保存图表
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        plt.close()
    
    def plot_summary(self, episode_rewards, losses, epsilons, smoothing_window=20,
                    window_size=100, success_threshold=475, save_name="summary_plot.png", show=True):
        """
        绘制综合训练图表（包含奖励、损失、成功率）
        
        参数：
            episode_rewards: 各回合奖励列表
            losses: 各回合损失列表
            epsilons: 各回合探索率列表
            smoothing_window: 滑动平均窗口大小
            window_size: 成功率计算窗口
            success_threshold: 成功阈值
            save_name: 保存文件名
            show: 是否显示图表
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # 奖励曲线
        axes[0, 0].plot(episode_rewards, alpha=0.5, label='原始奖励', color='#1f77b4')
        smoothed_rewards = self._moving_average(episode_rewards, smoothing_window)
        axes[0, 0].plot(smoothed_rewards, label=f'滑动平均', color='#ff7f0e', linewidth=2)
        axes[0, 0].axhline(y=success_threshold, color='#d62728', linestyle='--', label='成功阈值')
        axes[0, 0].set_xlabel('训练回合')
        axes[0, 0].set_ylabel('总奖励')
        axes[0, 0].set_title('训练奖励曲线')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(0, 550)
        
        # 损失曲线
        axes[0, 1].plot(losses, alpha=0.5, label='原始损失', color='#9467bd')
        smoothed_losses = self._moving_average(losses, smoothing_window)
        axes[0, 1].plot(smoothed_losses, label=f'滑动平均', color='#8c564b', linewidth=2)
        axes[0, 1].set_xlabel('训练回合')
        axes[0, 1].set_ylabel('平均损失')
        axes[0, 1].set_title('训练损失曲线')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 成功率曲线
        success_rates = []
        n_episodes = len(episode_rewards)
        for i in range(n_episodes):
            start = max(0, i - window_size + 1)
            window_rewards = episode_rewards[start:i+1]
            success_count = sum(1 for r in window_rewards if r >= success_threshold)
            success_rate = success_count / len(window_rewards) * 100
            success_rates.append(success_rate)
        
        axes[1, 0].plot(success_rates, label=f'成功率', color='#2ca02c', linewidth=2)
        axes[1, 0].axhline(y=100, color='#d62728', linestyle='--', label='100% 成功')
        axes[1, 0].set_xlabel('训练回合')
        axes[1, 0].set_ylabel('成功率 (%)')
        axes[1, 0].set_title('训练成功率曲线')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(0, 105)
        
        # 探索率曲线
        axes[1, 1].plot(epsilons, label='探索率 ε', color='#17becf', linewidth=2)
        axes[1, 1].set_xlabel('训练回合')
        axes[1, 1].set_ylabel('探索率')
        axes[1, 1].set_title('探索率变化曲线')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim(0, 1.05)
        
        plt.tight_layout()
        
        # 保存图表
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        plt.close()
    
    def _moving_average(self, data, window_size):
        """计算滑动平均"""
        if len(data) < window_size:
            return data
        
        smoothed = []
        for i in range(len(data)):
            start = max(0, i - window_size + 1)
            smoothed.append(float(np.mean(data[start:i+1])))
        return smoothed