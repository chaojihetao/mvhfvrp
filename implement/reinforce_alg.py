import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from framework.alg.rlbase import RLBase
from implement.environment import MTHFVRPEnv
from implement.evaluation import MTHFVRPEvaluator
from framework.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class REINFORCELightning(RLBase):
    """基于Lightning框架的REINFORCE算法实现
    """
    def __init__(
        self,
        train_env: MTHFVRPEnv,
        eval_env: MTHFVRPEnv, 
        model: nn.Module,
        evaluator: MTHFVRPEvaluator,
        batch_size: int = 256,
        train_batch_nums: int = 100,
        accumulate_grad_batches: int = 1,
        # 优化参数
        learning_rate=1e-4,
        warmup_iterations=100,
        min_lr=1e-6,
        weight_decay=0.01,
        max_grad_norm=1.0,
        # 熵控参数
        ent_coef=0.01,          
        ent_decay_start=0.5,    
        # 协方差控制参数 (新增)
        clip_cov_range=(0.5, 5.0),  
        cov_grad_detach_prob=0.05,
        # baseline选择
        use_num_starts_baseline=False,
        action_selection_strategy="sampling",  # 动作选择策略: 'sampling' 或 'greedy'
    ):
        super().__init__(train_env, model, batch_size)
        self.save_hyperparameters(ignore=['model', 'envs', 'eval_env', 'evaluator', 'train_env'])
        self.automatic_optimization = False 

        self.train_env = train_env
        self.batch_size = train_env.batch_size
        self.eval_env = eval_env
        self.evaluator = evaluator
        self.model = model
        
        # 参数保存
        self.learning_rate = learning_rate
        self.initial_lr = learning_rate
        self.warmup_iterations = warmup_iterations
        self.min_lr = min_lr
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        
        self.initial_ent_coef = ent_coef
        self.current_ent_coef = ent_coef
        self.ent_decay_start = ent_decay_start
        
        # 协方差参数
        self.clip_cov_range = clip_cov_range
        self.cov_grad_detach_prob = cov_grad_detach_prob
        self.cov_threshold = None

        # baseline选择
        self.use_num_starts_baseline = use_num_starts_baseline
        self.action_selection_strategy = action_selection_strategy
        
        self.reward_ema = None
        self.ema_alpha = 0.95
        self.accumulate_grad_batches = accumulate_grad_batches  # 梯度累积步数
        self.current_accumulate_step = 0    # 当前累积步数
        self.train_batch_nums = train_batch_nums * accumulate_grad_batches  # 实际训练批次数

        self._is_setup = False  # 标记是否已完成设置

    def setup(self, stage = None):
        """在训练开始前进行必要的设置"""
        if self._is_setup:
            return
        
        if self.trainer.world_size > 1:
            self.train_batch_nums = self.train_batch_nums * self.trainer.world_size
        
        super().setup(stage)

        # 将环境和模型移动到指定设备
        device = self.device
        self.train_env.to(device)
        self.eval_env.to(device)
        self.model.to(device)
        self._is_setup = True

    def collect_rollouts(self):
        """运行一个完整的 Episode"""
        state = self.train_env.reset()
        self.model.train()
        state_feature = self.train_env.get_global_features(state)
        self.model.feature(state_feature)
        
        if self.use_num_starts_baseline:
            num_starts, v_starts, c_starts = self.train_env.select_start_nodes(state)
            # action_v = v_starts.repeat(self.batch_size)     # 选车动作扩展 [B*S]
            action_c = c_starts.repeat(self.batch_size)     # 选点动作扩展 [B*S]

            state = state.repeat_interleave(num_starts, dim=0)
            total_rewards = torch.zeros(self.batch_size * num_starts, device=self.device)
            zero_tensor = torch.zeros_like(action_c, dtype=torch.int64, device=action_c.device)  
            # log_probs_list = [zero_tensor, zero_tensor]  # 预先添加两个零log_prob占位符
            # entropies_list = [zero_tensor, zero_tensor]   # 预先添加两个零entropy占位符
            log_probs_list = []
            entropies_list = []
            
            # 第一步：选车辆
            current_features, illegal_mask = self.train_env.get_current_feature_and_mask(state)
            action_logits = self.model.policy(current_features, illegal_mask)
            action_probs = torch.softmax(action_logits, dim=-1)
            dist = torch.distributions.Categorical(action_probs)

            # 采样动作
            action_v = dist.sample()
            log_probs = dist.log_prob(action_v)
            entropy = dist.entropy()
            log_probs_list.append(log_probs)
            entropies_list.append(entropy)

            # 选车辆
            state, reward_v, done = self.train_env.step(state, action_v)
            total_rewards += reward_v.squeeze()
            all_done = done.all()
            
            # 第二步：选第一个点
            # current_features, illegal_mask = self.train_env.get_current_feature_and_mask(state)
            # action_logits = self.model.policy(current_features, illegal_mask)
            # action_probs = torch.softmax(action_logits, dim=-1)
            # dist = torch.distributions.Categorical(action_probs)

            state, reward_c, done = self.train_env.step(state, action_c)
            total_rewards += reward_c.squeeze()
            all_done = done.all()

            log_probs_list.append(zero_tensor)
            entropies_list.append(zero_tensor)

            # log_probs = dist.log_prob(action_c)
            # entropy = dist.entropy()
            # log_probs_list.append(log_probs)
            # entropies_list.append(entropy)
            
        else:
            num_starts = state["locs"].shape[1] - 1  # 起点数量
            total_rewards = torch.zeros(self.batch_size * num_starts, device=self.device)
            state = state.repeat_interleave(num_starts, dim=0)  # 状态扩展 [B*S]
            log_probs_list = []
            entropies_list = []
            done = [False] * (self.batch_size * num_starts)
            all_done = False

        # rollout 过程
        while not all_done:
            active_mask = ~done
            current_features, illegal_mask = self.train_env.get_current_feature_and_mask(state)
            action_logits = self.model.policy(current_features, illegal_mask)
            action_probs = torch.softmax(action_logits, dim=-1)
            dist = torch.distributions.Categorical(action_probs)
            if self.action_selection_strategy == "greedy":
                # 贪心选择
                action = torch.argmax(action_probs, dim=-1)
            if self.action_selection_strategy == "sampling":
                # 采样选择
                action = dist.sample()

            log_probs = dist.log_prob(action)
            entropy = dist.entropy()
            log_probs_list.append(log_probs)
            entropies_list.append(entropy)

            state, reward, done = self.train_env.step(state, action)
            all_done = done.all()

            # 计算累积奖励
            reward = torch.where(
                active_mask.unsqueeze(-1), 
                reward, 
                torch.tensor(0.0, device=reward.device)
            )

            total_rewards += reward.squeeze()

        rewards = total_rewards # [B*S]

        # over time: [T, B*S]
        log_probs = torch.stack(log_probs_list, dim=0)
        entropy = torch.stack(entropies_list, dim=0)

        # [T, B*S] -> [T, B, S]
        T = log_probs.size(0)
        rewards_view = rewards.view(self.batch_size, num_starts)  # [B, S]
        log_probs_view = log_probs.view(T, self.batch_size, num_starts)   # [T, B, S]
        entropy_view = entropy.view(T, self.batch_size, num_starts)       # [T, B, S]

        return rewards_view, log_probs_view, entropy_view
        
    def configure_optimizers(self):
        """配置PPO优化器"""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            eps=1e-8,
        )
        self.optimizer = optimizer

        return {
            'optimizer': optimizer,
        }

    def compute_token_covariance(self, log_probs, advantages):
        """
        计算token-wise的log概率和优势函数的协方差
        
        Args:
            log_probs (torch.Tensor): 策略的log概率 [batch_size]
            advantages (torch.Tensor): 优势函数 [batch_size]
            
        Returns:
            cov (torch.Tensor): token-wise的协方差 [batch_size]
        """
        probs = torch.exp(log_probs)
        weighted_advantages = probs * advantages    

        # 计算均值
        log_probs_mean = log_probs.mean()
        weighted_adv_mean = weighted_advantages.mean()
        
        # 计算每个token的协方差: (log_pi(y_i) - mean(log_pi(y_j))) * (A(y_i) - mean(A(y_j)))
        token_cov = (log_probs - log_probs_mean) * (weighted_advantages - weighted_adv_mean)
        
        return token_cov

    def apply_covariance_strategy(self, token_cov):
        """
        基于协方差应用策略：从高协方差动作中抽样，进行梯度分离
        
        Args:
            token_cov (torch.Tensor): token协方差 [batch_size]
            batch_size (int): 批次大小
            
        Returns:
            cov_mask (torch.Tensor): 协方差动作掩码 [batch_size]
        """
        cov_min, cov_max = self.clip_cov_range
        if self.cov_threshold is None:
            # self.cov_threshold = torch.quantile(token_cov, 0.95)
            self.cov_threshold = torch.clamp(
                torch.quantile(token_cov, 0.95), 
                cov_min, 
                cov_max,
            )
        
        if self.cov_threshold > torch.quantile(token_cov, 0.95):
            self.cov_threshold = torch.clamp(
                torch.quantile(token_cov, 0.95), 
                cov_min, 
                cov_max,
            )

        # 识别高协方差范围的动作
        cov_mask = token_cov >= self.cov_threshold
        
        # 3. 对中等协方差动作进行随机采样决定是否梯度分离
        if cov_mask.any():
            random_mask = torch.rand_like(token_cov) < self.cov_grad_detach_prob
            cov_mask = cov_mask & random_mask
        
        return cov_mask

    def compute_loss(self, batch, batch_idx):
        rewards = batch['rewards']      # [B, S]
        log_probs = batch['log_probs']  # [T, B, S]
        entropy = batch['entropy']      # [T, B, S]

        # Advantage = (Sample - Baseline) / Std
        baseline = rewards.mean(dim=1, keepdim=True)
        # adv_std = rewards.std(dim=1, keepdim=True)
        advantage = rewards - baseline
        # advantage = (rewards - baseline) / (adv_std + 1e-8)  # [B, S]
        advantage = advantage.detach()
        # advantage = (rewards - baseline).detach()  # [B, S]
        
        # [B, S] -> [1, B, S] -> [T, B, S]
        T = log_probs.shape[0]
        advantage_expanded = advantage.unsqueeze(0).expand(T, -1, -1)
        
        # 展平所有维度 [T * B * S]
        advantage_flat = advantage_expanded.reshape(-1)
        log_probs_flat = log_probs.reshape(-1)
        entropy_flat = entropy.reshape(-1)

        # === 协方差控制逻辑 ===
        # 1. 计算协方差
        token_cov = self.compute_token_covariance(log_probs_flat, advantage_flat)
        
        # 2. 生成 Mask (哪些样本需要被 Detach)
        cov_mask = self.apply_covariance_strategy(token_cov)
        
        # 3. 应用协方差熵控
        if cov_mask.any():
            active_log_probs = log_probs_flat.clone()
            active_log_probs[cov_mask] = log_probs_flat[cov_mask].detach()
        else:
            active_log_probs = log_probs_flat

        # === 计算 Loss ===
        reinforce_loss = -(advantage_flat * active_log_probs).mean()
        entropy_loss = -entropy_flat.mean()
        total_loss = reinforce_loss + self.current_ent_coef * entropy_loss
        
        # 反向传播
        scaled_loss = total_loss / self.accumulate_grad_batches
        self.manual_backward(scaled_loss)

        # # 只有在累积了足够的步数，或者是最后一个batch时，才更新参数
        if (batch_idx + 1) % self.accumulate_grad_batches == 0: 
            # 使用 self.clip_gradients 代替直接调用 clip_grad_norm_
            # 这样Lightning可以正确处理多GPU情况
            self.clip_gradients(
                self.optimizer, 
                gradient_clip_val=self.max_grad_norm, 
                gradient_clip_algorithm="norm"
            )
            self.optimizer.step()
            self.optimizer.zero_grad()

        return {
            "sample_rewards": rewards.mean(), # 记录平均奖励
            "reinforce_loss": reinforce_loss,
            "entropy": entropy_flat,
            "cov_mask": cov_mask,
            "total_loss": total_loss / self.accumulate_grad_batches,
            # 协方差信息
            "cov_max": token_cov.max(),
            "cov_mean": token_cov.mean(),
            "cov_threshold": self.cov_threshold,
        }

    def train_dataloader(self):
        dataset = list(range(self.train_batch_nums))
        
        def collate_fn(batch):
            rewards, log_probs, entropy = self.collect_rollouts()
            
            return {
                'rewards': rewards,
                'log_probs': log_probs,
                'entropy': entropy
            }
        
        train_dataloader = DataLoader(
            dataset,
            batch_size=1,  # 每次处理一个环境实例 (内部处理了 B * S)
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )
        return train_dataloader
    
    def training_step(self, batch, batch_idx):
        # 迭代次数和梯度累积处理
        current_batch_idx = (batch_idx + 1) // self.accumulate_grad_batches - 1
        iteration = self.current_epoch * (self.train_batch_nums / (self.trainer.world_size * self.accumulate_grad_batches)) + current_batch_idx
        self.iteration = iteration 
        total_iterations = self.trainer.max_epochs * self.train_batch_nums
        total_iterations = total_iterations // self.accumulate_grad_batches // self.trainer.world_size

        # 梯度累积索引更新
        self.current_accumulate_step = self.current_accumulate_step + 1
        if self.current_accumulate_step >= self.accumulate_grad_batches:
            self.current_accumulate_step = 0

        is_accumulation_done = (self.current_accumulate_step == 0)

        # 计算loss，并获取日志信息
        loss_info = self.compute_loss(batch, batch_idx)
        sample_rewards = loss_info['sample_rewards']
        reinforce_loss = loss_info['reinforce_loss']
        entropy = loss_info['entropy']
        cov_mask = loss_info['cov_mask']
        total_loss = loss_info['total_loss']

        # 获取协方差统计信息
        cov_max = loss_info['cov_max']
        cov_mean = loss_info['cov_mean']
        cov_threshold = loss_info['cov_threshold']

        # 日志记录和指标更新
        current_reward = sample_rewards.mean().item()
        if is_accumulation_done:    # 仅在完成梯度累积后更新指标和日志
            if self.reward_ema is None:
                self.reward_ema = current_reward
            else:
                self.reward_ema = self.ema_alpha * self.reward_ema + (1 - self.ema_alpha) * current_reward
            
            # 更新学习率
            if iteration < self.warmup_iterations:
                # Warmup阶段: 线性增加学习率从 min_lr 到 initial_lr
                # 避免在 warmup_iterations 为 0 时除以零
                if self.warmup_iterations > 0:
                    lr_progress = iteration / self.warmup_iterations
                    self.current_lr = self.min_lr + (self.initial_lr - self.min_lr) * lr_progress
                else:
                    # 如果没有 warmup，直接使用 initial_lr (虽然理论上不会进入此分支，因为 iteration < 0 不成立)
                    self.current_lr = self.initial_lr 
            else:
                # Cosine Annealing阶段: 从 initial_lr 退火到 min_lr
                total_annealing_iterations = total_iterations - self.warmup_iterations
                # 避免在退火迭代次数为 0 或负数时除以零或计算错误
                if total_annealing_iterations <= 0:
                    self.current_lr = self.min_lr # 如果没有退火阶段，直接设为 min_lr
                else:
                    annealing_progress = (iteration - self.warmup_iterations) / total_annealing_iterations
                    # 确保 progress 不超过 1.0 (可能由于浮点数精度问题略微超过)
                    annealing_progress = min(annealing_progress, 1.0) 
                    self.current_lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1 + math.cos(math.pi * annealing_progress))

            # 应用计算出的学习率
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.current_lr
        
            # 更新熵系数
            if iteration / total_iterations > self.ent_decay_start:
                # 在剩余的训练过程中线性衰减到0
                decay_progress = (iteration / total_iterations - self.ent_decay_start) / (1 - self.ent_decay_start)
                decay_progress = min(max(decay_progress, 0.0), 1.0)  # 限制在[0,1]
                self.current_ent_coef = max(0.0, self.initial_ent_coef * (1 - decay_progress))

        # 记录训练指标（保留Lightning的日志记录方式）
        if self.global_rank == 0 and is_accumulation_done:
            log_msg = (f"[Iter {int(self.iteration)}/{total_iterations}] "
                      f"EMA Reward: {self.reward_ema:.4f} | "
                      f"Loss: {reinforce_loss.item():.4f} | "
                      f"Entropy: {entropy.mean().item():.4f} | "
                      f"CovDetached: {cov_mask.float().mean().item():.2%} | ")
                
            log_msg += (f"Total Loss: {total_loss.item():.4f} | "
                      f"LR: {self.current_lr:.6f} | "
                      f"Ent Coef: {self.current_ent_coef:.6f} | "
                      f"Cov Max: {cov_max.item():.4f} | "
                      f"Cov Mean: {cov_mean.item():.4f} | "
                      f"Cov Thr: {cov_threshold. item():.4f} ")
            
            self.log_msgs = log_msg
        elif self.global_rank == 0:
            self.log_msgs = None
        
        if is_accumulation_done:
            records = {
                'EMA Reward': self.reward_ema,
                'Loss': reinforce_loss.item(),
                'Entropy': entropy.mean().item(),
                'CovDetached': cov_mask.float().mean().item(),
                'Total Loss': total_loss.item(),
                'LR': self.current_lr,
                'Ent Coef': self.current_ent_coef,
                'Cov Max': cov_max.item(),
                'Cov Mean': cov_mean.item(),
                'Cov Thr': cov_threshold.item(),
            }
            self.records = records