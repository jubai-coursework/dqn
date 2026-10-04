import csv
import json
import os
from datetime import datetime
from collections import deque


class TrainingLogger:
    """训练日志记录器，用于记录训练过程中的关键数据"""
    
    def __init__(self, log_dir="logs", experiment_name=None):
        """
        初始化日志记录器
        
        参数：
            log_dir: 日志保存目录
            experiment_name: 实验名称，用于区分不同实验
        """
        self.log_dir = log_dir
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name
        self.full_log_dir = os.path.join(log_dir, experiment_name)
        
        # 创建日志目录
        os.makedirs(self.full_log_dir, exist_ok=True)
        
        # 日志数据存储
        self.episode_data = []
        self.epoch_data = []
        
        # CSV文件路径
        self.episode_csv_path = os.path.join(self.full_log_dir, "episode_log.csv")
        self.epoch_csv_path = os.path.join(self.full_log_dir, "epoch_log.csv")
        self.config_path = os.path.join(self.full_log_dir, "config.json")
        
        # 初始化CSV文件
        self._init_csv_files()
    
    def _init_csv_files(self):
        """初始化CSV文件，写入表头"""
        # 回合级别日志表头
        with open(self.episode_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'episode', 'total_reward', 'avg_reward_100', 'epsilon', 
                'avg_loss', 'steps', 'success'
            ])
        
        # 周期级别日志表头
        with open(self.epoch_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'avg_reward', 'max_reward', 'min_reward', 
                'avg_loss', 'success_rate', 'epsilon'
            ])
    
    def log_episode(self, episode, total_reward, avg_reward_100, epsilon, avg_loss, steps, success):
        """
        记录单个回合的数据
        
        参数：
            episode: 回合数
            total_reward: 回合总奖励
            avg_reward_100: 最近100回合平均奖励
            epsilon: 当前探索率
            avg_loss: 回合平均损失
            steps: 回合步数
            success: 是否成功（如CartPole达到500步）
        """
        data = {
            'episode': episode,
            'total_reward': total_reward,
            'avg_reward_100': avg_reward_100,
            'epsilon': epsilon,
            'avg_loss': avg_loss,
            'steps': steps,
            'success': int(success)
        }
        self.episode_data.append(data)
        
        # 写入CSV
        with open(self.episode_csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                episode, total_reward, avg_reward_100, epsilon, 
                avg_loss, steps, int(success)
            ])
    
    def log_epoch(self, epoch, rewards, losses, epsilon, success_threshold=475):
        """
        记录一个周期（多个回合）的数据
        
        参数：
            epoch: 周期数
            rewards: 该周期内所有回合的奖励列表
            losses: 该周期内所有回合的损失列表
            epsilon: 当前探索率
            success_threshold: 成功阈值
        """
        avg_reward = float(np.mean(rewards)) if rewards else 0.0
        max_reward = float(np.max(rewards)) if rewards else 0.0
        min_reward = float(np.min(rewards)) if rewards else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0
        success_rate = float(sum(1 for r in rewards if r >= success_threshold) / len(rewards)) if rewards else 0.0
        
        data = {
            'epoch': epoch,
            'avg_reward': avg_reward,
            'max_reward': max_reward,
            'min_reward': min_reward,
            'avg_loss': avg_loss,
            'success_rate': success_rate,
            'epsilon': epsilon
        }
        self.epoch_data.append(data)
        
        # 写入CSV
        with open(self.epoch_csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, avg_reward, max_reward, min_reward, 
                avg_loss, success_rate, epsilon
            ])
    
    def save_config(self, config):
        """保存配置参数到JSON文件"""
        config_dict = {k: v for k, v in vars(config).items() if not k.startswith('_')}
        # 处理不可序列化的对象（如torch.device）
        for key, value in config_dict.items():
            if isinstance(value, torch.device):
                config_dict[key] = str(value)
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=4, ensure_ascii=False)
    
    def get_log_directory(self):
        """获取日志保存目录"""
        return self.full_log_dir
    
    def print_summary(self):
        """打印训练日志摘要"""
        if not self.episode_data:
            print("No episode data recorded.")
            return
        
        total_episodes = len(self.episode_data)
        total_rewards = [d['total_reward'] for d in self.episode_data]
        avg_total_reward = sum(total_rewards) / total_episodes
        max_reward = max(total_rewards)
        min_reward = min(total_rewards)
        success_count = sum(1 for d in self.episode_data if d['success'])
        success_rate = success_count / total_episodes * 100
        
        print("\n" + "=" * 50)
        print("训练日志摘要")
        print("=" * 50)
        print(f"实验名称: {self.experiment_name}")
        print(f"总回合数: {total_episodes}")
        print(f"平均奖励: {avg_total_reward:.2f}")
        print(f"最高奖励: {max_reward:.2f}")
        print(f"最低奖励: {min_reward:.2f}")
        print(f"成功率: {success_rate:.2f}% ({success_count}/{total_episodes})")
        print(f"日志保存目录: {self.full_log_dir}")
        print("=" * 50)


import numpy as np
import torch