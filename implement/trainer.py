from typing import Iterable, List, Optional, Union

import lightning.pytorch as pl
import torch
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo

from lightning import Callback, Trainer
from lightning.fabric.accelerators.cuda import num_cuda_devices
from lightning.pytorch.accelerators import Accelerator
from lightning.pytorch.core.datamodule import LightningDataModule
from lightning.pytorch.loggers import Logger
from lightning.pytorch.strategies import DDPStrategy, Strategy
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

from framework.utils.pylogger import get_pylogger
from implement.environment import MTHFVRPEnv
from implement.evaluation import MTHFVRPEvaluator


log = get_pylogger(__name__)


class OurTrainer(Trainer):
    """Wrapper around Lightning Trainer, for efficient training.

    Note:
        The most important hyperparameter to use is `reload_dataloaders_every_n_epochs`.
        This allows for datasets to be re-created on the run and distributed by Lightning across
        devices on each epoch. Setting to a value different than 1 may lead to overfitting to a
        specific (such as the initial) data distribution.

    Args:
        accelerator: hardware accelerator to use.
        callbacks: list of callbacks.
        logger: logger (or iterable collection of loggers) for experiment tracking.
        min_epochs: minimum number of training epochs.
        max_epochs: maximum number of training epochs.
        strategy: training strategy to use (if any), such as Distributed Data Parallel (DDP).
        devices: number of devices to train on (int) or which GPUs to train on (list or str) applied per node.
        gradient_clip_val: 0 means don't clip. Defaults to 1.0 for stability.
        precision: allows for mixed precision training. Can be specified as a string (e.g., '16').
            This also allows to use `FlashAttention` by default.
        disable_profiling_executor: Disable JIT profiling executor. This reduces memory and increases speed.
        auto_configure_ddp: Automatically configure DDP strategy if multiple GPUs are available.
        reload_dataloaders_every_n_epochs: Set to a value different than 1 to reload dataloaders every n epochs.
        matmul_precision: Set matmul precision for faster inference https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
        **kwargs: Additional keyword arguments passed to the Lightning Trainer. See :class:`lightning.pytorch.trainer.Trainer` for details.
    """

    def __init__(
        self,
        accelerator: Union[str, Accelerator] = "auto",
        callbacks: Optional[List[Callback]] = None,
        logger: Optional[Union[Logger, Iterable[Logger]]] = None,
        min_epochs: Optional[int] = None,
        max_epochs: Optional[int] = None,
        strategy: Union[str, Strategy] = "auto",
        devices: Union[List[int], str, int] = "auto",
        gradient_clip_val: Union[int, float] = 1.0,
        precision: Union[str, int] = "16-mixed",
        reload_dataloaders_every_n_epochs: int = 1,
        disable_profiling_executor: bool = True,
        auto_configure_ddp: bool = True,
        matmul_precision: Union[str, int] = "medium",
        **kwargs,
    ):
        # Disable JIT profiling executor. This reduces memory and increases speed.
        # Reference: https://github.com/HazyResearch/safari/blob/111d2726e7e2b8d57726b7a8b932ad8a4b2ad660/train.py#LL124-L129C17
        if disable_profiling_executor:
            try:
                torch._C._jit_set_profiling_executor(False)
                torch._C._jit_set_profiling_mode(False)
            except AttributeError:
                pass

        # Configure DDP automatically if multiple GPUs are available
        if auto_configure_ddp and strategy == "auto":
            if devices == "auto":
                n_devices = num_cuda_devices()
            elif isinstance(devices, Iterable):
                n_devices = len(devices)
            else:
                n_devices = devices
            if n_devices > 1:
                log.info(
                    "Configuring DDP strategy automatically with {} GPUs".format(
                        n_devices
                    )
                )
                strategy = DDPStrategy(
                    find_unused_parameters=True,  # We set to False due to RL envs
                    gradient_as_bucket_view=True,  # https://pytorch-lightning.readthedocs.io/en/stable/advanced/advanced_gpu.html#ddp-optimizations
                )

        # Set matmul precision for faster inference https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
        if matmul_precision is not None:
            torch.set_float32_matmul_precision(matmul_precision)

        # Check if gradient_clip_val is set to None
        if gradient_clip_val is None:
            log.warning(
                "gradient_clip_val is set to None. This may lead to unstable training."
            )

        # We should reload dataloaders every epoch for RL training
        if reload_dataloaders_every_n_epochs != 1:
            log.warning(
                "We reload dataloaders every epoch for RL training. Setting reload_dataloaders_every_n_epochs to a value different than 1 "
                + "may lead to unexpected behavior since the initial conditions will be the same for `n_epochs` epochs."
            )

        # Main call to `Trainer` superclass
        super().__init__(
            accelerator=accelerator,
            callbacks=callbacks,
            logger=logger,
            min_epochs=min_epochs,
            max_epochs=max_epochs,
            strategy=strategy,
            gradient_clip_val=gradient_clip_val,
            devices=devices,
            precision=precision,
            reload_dataloaders_every_n_epochs=reload_dataloaders_every_n_epochs,
            **kwargs,
        )

    def fit(
        self,
        model: "pl.LightningModule",
        train_dataloaders: Optional[Union[TRAIN_DATALOADERS, LightningDataModule]] = None,
        val_dataloaders: Optional[EVAL_DATALOADERS] = None,
        datamodule: Optional[LightningDataModule] = None,
        ckpt_path: Optional[str] = None,
    ) -> None:
        """
        We override the `fit` method to automatically apply and handle magic
        to 'self.automatic_optimization = False' models, such as PPO

        It behaves exactly like the original `fit` method, but with the following changes:
        - if the given model is 'self.automatic_optimization = False', we override 'gradient_clip_val' as None
        """

        if not model.automatic_optimization:
            if self.gradient_clip_val is not None:
                log.warning(
                    "Overriding gradient_clip_val to None for 'automatic_optimization=False' models"
                )
                self.gradient_clip_val = None

        super().fit(
            model=model,
            train_dataloaders=train_dataloaders,
            val_dataloaders=val_dataloaders,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
        )


# === 一些训练时需要用的callback ===
class MyCallback(Callback):
    """基于奖励的验证和模型保存回调
    
    功能：
    1. 每隔指定epoch数进行一次验证
    2. 根据验证奖励保存最佳模型
    3. 记录验证指标
    """
    
    def __init__(
        self,
        eval_env: MTHFVRPEnv,          # 验证环境
        evaluator: MTHFVRPEvaluator,   # 验证评估器
        val_every_n_epochs: int = 10,  # 每隔多少个epoch进行一次验证
        save_top_k: int = 1,           # 保存最好的k个模型
        save_dir: str = "checkpoints", # 保存目录
        filename_prefix: str = "best_model",  # 文件名前缀
        monitor_metric: str = "val_mean_reward", # 监控的指标
        mode: str = "max",             # max表示越大越好，min表示越小越好
        verbose: bool = True,          # 是否打印详细信息
        save_last: bool = True,        # 是否保存最后一个模型

        # 日志目录
        log_dir: str = "logs",         # 日志目录
        use_tensorboard: bool = False,  # 是否使用tensorboard记录日志

        # 早停相关参数
        early_stopping: bool = False,  # 是否启用早停
        patience: int = 20,            # 早停耐心值（多少个验证周期没有改善就停止）
        min_delta: float = 0.0001,     # 最小改善阈值
    ):
        super().__init__()
        
        self.eval_env = eval_env
        self.evaluator = evaluator
        self.val_every_n_epochs = val_every_n_epochs
        self.save_top_k = save_top_k
        self.save_dir = save_dir
        self.filename_prefix = filename_prefix + f"_{datetime.now(tz=ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')}"  # 添加时间戳
        self.monitor_metric = monitor_metric
        self.mode = mode
        self.verbose = verbose
        self.save_last = save_last
        self.log_dir = log_dir + '/' + self.filename_prefix
        self.use_tensorboard = use_tensorboard

        # 用于跟踪最佳模型
        self.best_scores = []  # 存储 (score, epoch, model_path) 的列表
        self.best_score = float('-inf') if mode == 'max' else float('inf')
        self.last_val_epoch = -1
        
        # 早停
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta
        self.wait_count = 0        # 等待计数器
        self.stopped_epoch = 0     # 停止的epoch
        self.best_score_for_patience = self.best_score  # 用于耐心机制的最佳分数
        self.should_stop = False   # 是否应该停止训练
        self.current_epoch = 0

    def on_fit_start(self, trainer, pl_module):
        """在训练开始时创建保存目录和日志目录"""
        import os

        if trainer.global_rank == 0:
            # 创建保存目录和日志目录
            os.makedirs(self.log_dir, exist_ok=True)
            os.makedirs(self.save_dir, exist_ok=True)

            # 配置日志同时输出到文件和控制台
            import logging
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)
            for handler in logger.handlers[:]:  
                logger.removeHandler(handler)   # 清除之前的处理器
                
            # 文件处理器
            fh = logging.FileHandler(os.path.join(self.log_dir, f"{self.filename_prefix}.log"))    # 日志文件路径
            fh.setLevel(logging.INFO)
            
            # 控制台处理器
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            
            # 设置日志格式
            formatter = logging.Formatter('%(message)s')
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            
            # 添加处理器
            logger.addHandler(fh)
            logger.addHandler(ch)
            self.log = logger   # 保存logger以便后续使用
            
            # 创建TensorBoard日志写入器
            if self.use_tensorboard:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(self.log_dir)  # 创建TensorBoard日志写入器

            # 打印多GPU训练信息
            self.log.info("=" * 60)
            self.log.info(f"🚀 Training Started")
            self.log.info(f"   Number of GPUs: {trainer.num_devices}")
            self.log.info(f"   Strategy: {trainer.strategy.__class__.__name__}")
            self.log.info(f"   Per-GPU batch size: {pl_module.batch_size}")
            self.log.info("=" * 60)

        # 同步所有进程，确保目录创建完成
        if trainer.world_size > 1:
            trainer.strategy.barrier()

        # 第一次验证 - 只在主进程执行
        if trainer.global_rank == 0:
            if self.verbose:
                self. log.info("Starting validation before training...")
            self._run_validation(trainer, pl_module, 0)
        
        # 同步所有进程
        if trainer.world_size > 1:
            trainer.strategy.barrier()

    # == 验证回调 ==
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """在每个训练epoch结束时, 输出日志，并检查是否需要验证"""
        if trainer.global_rank == 0:
            if self.verbose:
                if hasattr(pl_module, 'log_msgs') and pl_module.log_msgs is not None:
                    self.log. info(pl_module.log_msgs)
 
            if hasattr(pl_module, 'records') and self.use_tensorboard:
                records = pl_module.records
                for key, value in records.items():
                    self.writer.add_scalar(key, value, self.current_epoch)
                
        # 检查是否需要进行验证
        if pl_module.current_epoch > self.current_epoch:
            self.current_epoch = pl_module.current_epoch
            if trainer.world_size > 1:
                trainer.strategy.barrier()

            if trainer.global_rank == 0:
                if self.verbose:
                    self.log.info(f"Epoch {self.current_epoch + 1}:  Running validation...")
                self._run_validation(trainer, pl_module, self.current_epoch)
                
            if trainer.world_size > 1:
                trainer.strategy.barrier()

            if self.early_stopping and self.should_stop: 
                trainer.should_stop = True
                if trainer.global_rank == 0 and self.verbose:
                    self. log.info(f"Early stopping triggered at epoch {self.current_epoch + 1}")
                    self.log.info(f"Best validation score was {self.best_score_for_patience:.4f}")

    def on_train_end(self, trainer, pl_module):
        """训练结束时保存最后一个模型"""
        if trainer. global_rank != 0:
            return

        if self.save_last:
            last_model_path = f"{self.save_dir}/last_{self.filename_prefix}.ckpt"
            
            # 保存checkpoint
            self._save_checkpoint(trainer, pl_module, last_model_path)
            
            if self.verbose:
                self.log.info(f"Last model saved: {last_model_path}")
                
        # 打印训练总结
        if self.verbose:
            if self.early_stopping and self.should_stop:
                self.log.info(f"Training stopped early at epoch {self.stopped_epoch + 1}")
                self.log.info(f"Best validation score: {self.best_score_for_patience:.4f}")
            
            # 打印最佳模型总结
            if self.best_scores:
                self.log.info("=" * 50)
                self.log.info("Best models summary:")
                for i, (score, epoch, path) in enumerate(self.best_scores):
                    self.log.info(f"  {i+1}. Epoch {epoch+1}: {score:.4f} - {path}")
                self.log.info("=" * 50)


    # == 具体的验证逻辑 ==
    def _run_validation(self, trainer, pl_module, current_epoch):
        """执行验证并保存模型"""
        # 设置模型为评估模式
        pl_module.model.eval()
        
        with torch.no_grad():
            if hasattr(self, 'evaluator') and hasattr(self, 'eval_env'):
                # 使用evaluator进行验证
                eval_results = self.evaluator.evaluate(
                    model=pl_module.model,
                    env=self.eval_env.to(pl_module.device),
                    greedy=True,
                )
                
                # 提取关键指标
                total_rewards = eval_results
                avg_reward = np.mean(total_rewards)
                
                # 记录验证指标到TensorBoard
                if self.use_tensorboard:
                    self.writer.add_scalar('Val Mean Reward', avg_reward, current_epoch)
                
                if self.verbose:
                    self.log.info(f"Validation results - Mean Reward: {avg_reward:.4f}")

                # 检查是否是最佳模型
                current_score = avg_reward if self.monitor_metric == 'val_mean_reward' else sum(total_rewards)
                self._check_and_save_best_model(trainer, pl_module, current_score, current_epoch)

                # 检查早停条件
                if self.early_stopping:
                    self._check_early_stopping(current_score, current_epoch, trainer)
                
            else:
                self.log.warning("No evaluator or eval_env found in pl_module, skipping validation")
        
        # 恢复训练模式
        pl_module.model.train()
        
    def _check_and_save_best_model(self, trainer, pl_module, score, current_epoch):
        """检查并保存最佳模型"""
        is_better = False
        
        if self.mode == 'max':
            is_better = score > self.best_score
        else:
            is_better = score < self.best_score
            
        if is_better:
            self.best_score = score
            
            # 使用Lightning的checkpoint保存机制
            model_path = f"{self.save_dir}/{self.filename_prefix}_epoch_{current_epoch+1}.ckpt"
            
            # 保存checkpoint
            self._save_checkpoint(trainer, pl_module, model_path)
            
            # 更新最佳模型列表
            self.best_scores.append((score, current_epoch, model_path))
            
            # 按分数排序（根据mode）
            if self.mode == 'max':
                self.best_scores.sort(key=lambda x: x[0], reverse=True)
            else:
                self.best_scores.sort(key=lambda x: x[0])
            
            # 保持最多save_top_k个模型
            if len(self.best_scores) > self.save_top_k:
                # 删除多余的模型文件
                _, _, path_to_delete = self.best_scores.pop()
                try:
                    import os
                    os.remove(path_to_delete)
                except Exception as e:
                    self.log.warning(f"Failed to remove old model {path_to_delete}: {e}")
            
            if self.verbose:
                self.log.info(f"New best model saved: {model_path} (score: {score:.4f})")
                
    def _check_early_stopping(self, current_score, current_epoch, trainer):
        """检查早停条件"""

        # 检查是否有改善
        if self.mode == 'max':
            is_improvement = current_score > self.best_score_for_patience + self.min_delta
        else:
            is_improvement = current_score < self.best_score_for_patience - self.min_delta

        if is_improvement:
            self.best_score_for_patience = current_score
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.verbose:
                self.log.info(f"No improvement for {self.wait_count}/{self.patience} validation checks")

        # 检查是否超过耐心值
        if self.wait_count >= self.patience:
            self.should_stop = True
            self.stopped_epoch = current_epoch
            if self.verbose:
                self.log.info(f"Early stopping: No improvement for {self.patience} validation checks")

    def _save_checkpoint(self, trainer, pl_module, filepath):
        """保存模型检查点"""
        torch.save({
            'model_state_dict': pl_module.model.state_dict(),
            'optimizer_state_dict': pl_module.optimizer.state_dict(),
            'reward_ema': pl_module.reward_ema,
        }, filepath)
        self.log.info(f"Saved model checkpoint to {filepath}")