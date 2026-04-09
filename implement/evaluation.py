import torch
from typing import Dict, Any, Optional

from implement.environment import MTHFVRPEnv
from framework.eval import EvaluatorBase

class MTHFVRPEvaluator(EvaluatorBase):
    """评估器
    """

    def _evaluate_impl(self, 
                      model: torch.nn.Module,
                      env: MTHFVRPEnv,
                      greedy: bool = True,
                      ) -> Dict[str, Any]:
        """实现评估逻辑。
        
        Args:
            model: 要评估的模型
            env: 评估环境
            greedy: 是否使用贪心策略
            
        Returns:
            评估结果字典
        """
        # 重置环境
        states = env.reset()
        batch_size, total_routes_num = states['available_vehicles'].shape    # 获取batch_size和总路线数
        total_rewards = [0.0] * batch_size
        dones = [False] * batch_size

        # 设置动作选择策略
        if greedy:
            current_selector = self._greedy_action_selector
        else:
            current_selector = lambda x: self._mixed_action_selector(x, batch_size)

        # 1. 预计算全局特征
        batch = env.get_global_features(states)
        batch = self._move_to_device(batch)
        model.eval()
        with torch.no_grad():
            model.feature(batch)

        # 开始评估
        while not all(dones):
            # 检查活跃的环境
            active_mask = torch.tensor([not done for done in dones], dtype=torch.bool, device=states.device)  # [B]
            if not active_mask.any():
                break

            model.eval()
            with torch.no_grad():
                # 前向传播获取动作概率
                current_features, illegal_mask = env.get_current_feature_and_mask(states)
                action_probs = model.policy(current_features, illegal_mask)
                action_probs = torch.softmax(action_probs, dim=-1)

                # 选择动作
                actions = current_selector(action_probs)

            # 环境step
            new_state, reward, done = env.step(states, actions)
            
            # 更新状态和奖励
            states = new_state
            # total_cost = states['total_cost']  # [B, 1]
            reward = torch.where(
                active_mask.unsqueeze(-1), 
                reward, 
                torch.tensor(0.0, device=reward.device)
            )

            # 累积奖励
            for i in range(len(reward)):
                total_rewards[i] += reward[i].squeeze().item()

            # 更新完成状态
            dones = [done[i].item() for i in range(len(done))]
        

        return {
            'total_rewards': total_rewards,
            }
