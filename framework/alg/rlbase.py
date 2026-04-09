import abc
from typing import Any, Dict
import torch
import torch.nn as nn
from lightning import LightningModule
from tensordict import TensorDict

from framework.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class RLBase(LightningModule, metaclass=abc.ABCMeta):
    """强化学习算法基类
    
    基于PyTorch Lightning的RL算法基类，提供了完整的训练框架和配置化能力。
    该基类处理了所有通用的训练逻辑，子类只需实现算法特定的部分。
    
    设计原则:
        1. 环境驱动而非数据集驱动
        2. 配置化的训练流程控制  
        3. 清晰的抽象接口定义
        4. 完整的日志和监控系统
        
    子类必须实现的方法:
        - collect_rollouts(): 数据收集策略
        - compute_loss(): 损失计算逻辑  
        - configure_optimizers(): 优化器配置
        - update_policy(): 策略更新方式
        
    可选重写的方法:
        - validation_step(): 验证逻辑
        - on_train_epoch_end(): 训练epoch结束钩子
        - post_setup_hook(): setup后的自定义逻辑
    
    Args:
        train_env: 训练环境实例
        model: 神经网络模型
        batch_size: 批次大小
        train_batch_nums: 训练批次数
    """

    def __init__(
        self,
        train_env: Any,     # 训练环境
        model: nn.Module,   # 神经网络模型
        batch_size: int = 256,             # 每次训练的批次大小
    ):
        super().__init__()
        
        
        self.train_env = train_env
        self.model = model
        self.batch_size = batch_size
        
        self.save_hyperparameters(logger=False)

    def train_dataloader(self):
        """训练数据加载器
        
        返回一个适配RL环境交互的数据加载器。
        每个batch实际上是一个从rollout中抽样的数据。
        
        Returns:
            DataLoader: 包装了环境交互逻辑的数据加载器

        eg:
        def train_dataloader(self):
            # 创建一个简单的范围数据集
            dataset = list(range(self.train_batch_nums))

            # collect rollout data
            self.envs = self.train_env
            rollout_data = self.collect_rollouts()                  # 收集轨迹数据
            rollout_data = self.compute_advantages(rollout_data)    # 计算优势函数
            rollout_data = self.validate_rollouts(rollout_data)     # 筛选有效样本
            self.rollout_data = rollout_data                        # 保存rollout data以便调试
            
            # 把rollout data 按照约定好的batch进行分配训练
            def collate_fn(batch):
                batch = self.select_rollout_data(self.rollout_data)          # 筛选批次数据
                return batch
            
            train_dataloader = DataLoader(
                dataset,
                batch_size=1,  # 每次处理一个环境实例
                shuffle=False,  # RL不需要shuffle
                num_workers=0,  # 改为0，因为数据已经预收集了，避免多进程问题
                collate_fn=collate_fn,  # 使用自定义 collate 函数
            )
            return train_dataloader

        """
        raise NotImplementedError()

    def training_step(self, batch: Any, batch_idx: int):
        """这是算法的核心抽象方法，子类必须实现。包含了数据收集、损失计算等算法特定逻辑。
        
        Args:
            batch: 当前batch数据(通常是一个环境实例)
            batch_idx: batch索引
        """
        raise NotImplementedError(
            "子类必须实现 training_step 方法。这是算法的核心逻辑，应该包含:\n"
            "1. 环境交互和数据收集 (collect_rollouts)\n"
            "2. 损失计算 (compute_loss)\n"
            "3. 策略更新 (update_policy)\n"
            "4. 指标计算和日志记录"
        )
    
    def configure_optimizers(self):
        """配置优化器和学习率调度器
        
        这是算法特定的抽象方法，子类必须实现。不同的RL算法可能需要不同的优化策略。
        
        Returns:
            Union[Optimizer, Dict]: 优化器或包含优化器和调度器的字典
            
        示例返回格式:
            # 简单优化器
            return torch.optim.Adam(self.model.parameters(), lr=1e-4)
            
            # 带调度器的配置
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',  # 'epoch' or 'step'
                    'frequency': 1,
                    'monitor': 'val/mean_reward'
                }
            }
        """
        raise NotImplementedError(
            "子类必须实现 configure_optimizers 方法。应该根据算法特点配置:\n"
            "1. 优化器类型和参数 (Adam, SGD, RMSprop等)\n"
            "2. 学习率调度策略 (StepLR, CosineAnnealingLR等)\n" 
            "3. 多优化器配置 (如Actor-Critic算法)"
        )
    
    @abc.abstractmethod
    def collect_rollouts(self, env: Any, num_rollouts: int) -> TensorDict:
        """收集环境交互数据
        
        这是RL算法的核心数据收集逻辑，子类必须实现。不同算法有不同的数据收集策略。
        
        Args:
            env: 环境实例
            num_rollouts: 要收集的rollout数量
            
        Returns:
            TensorDict: 收集到的经验数据，通常包含:
                - states: 状态序列 [batch_size, seq_len, state_dim]
                - actions: 动作序列 [batch_size, seq_len, action_dim] 
                - rewards: 奖励序列 [batch_size, seq_len]
                - dones: 结束标志 [batch_size, seq_len]
                - log_probs: 动作概率(在线策略算法需要)
                - values: 状态价值(Actor-Critic算法需要)
                
        实现要点:
            - On-Policy算法: 使用当前策略收集数据
            - Off-Policy算法: 可以使用经验回放缓冲区
            - 注意处理episode边界和padding
        """
        pass
    
    @abc.abstractmethod  
    def compute_loss(self, rollout_data: TensorDict) -> Dict[str, torch.Tensor]:
        """计算算法特定的损失函数
        
        根据收集的rollout数据计算损失，这是不同RL算法的核心区别所在。
        
        Args:
            rollout_data: collect_rollouts返回的经验数据
            
        Returns:
            Dict[str, torch.Tensor]: 损失字典，必须包含'loss'键，例如:
                {
                    'loss': total_loss,           # 总损失(必需)
                    'policy_loss': policy_loss,   # 策略损失
                    'value_loss': value_loss,     # 价值损失  
                    'entropy_loss': entropy_loss, # 熵损失
                }
                
        算法示例:
            - PPO: clip_loss + value_loss + entropy_loss
            - A2C: policy_loss + value_loss + entropy_loss  
            - DQN: mse_loss(q_values, target_q_values)
            - SAC: policy_loss + q_loss + alpha_loss
        """
        pass