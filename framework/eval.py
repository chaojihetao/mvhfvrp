import abc
import torch
from typing import List, Dict, Any, Optional, Union, Callable
from tensordict import TensorDict


class EvaluatorBase(metaclass=abc.ABCMeta):
    """评估器基类，定义了模型评估的通用接口和流程。
    
    该基类提供了评估的基本框架，包括：
    1. 设备管理
    2. 动作选择策略
    3. 结果收集和可视化
    
    子类需要实现具体的评估逻辑。
    """
    
    def __init__(self, 
                 device: Optional[torch.device] = None,
                 action_selector: Optional[Callable] = None):
        """初始化评估器。
        
        Args:
            device: 计算设备，默认自动选择GPU或CPU
            action_selector: 动作选择策略，默认使用贪心策略
        """
        self.eval_data = None   # 存储评估的过程数据
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_selector = action_selector or self._greedy_action_selector
        
    def evaluate(self, 
                model: torch.nn.Module,
                env: any,
                greedy: bool = True,
                ) -> Dict[str, Any]:
        """评估模型在环境中的性能。
        
        Args:
            model: 要评估的模型
            env: 评估环境
            greedy: 是否使用贪心策略
            
        Returns:
            全部奖励
        """
        # 设置模型为评估模式
        model.eval()
        
        with torch.no_grad():
            eval_results = self._evaluate_impl(model, env, greedy=greedy)
            total_rewards = eval_results['total_rewards']
            return total_rewards

    @abc.abstractmethod
    def _evaluate_impl(self, 
                       model: torch.nn.Module,
                       env: any,
                       greedy: bool = True,
                       ) -> Dict[str, Any]:
        """具体的评估实现，由子类实现。
        
        Args:
            model: 要评估的模型
            env: 评估环境
            greedy: 是否使用贪心策略
            
        Returns:
            包含评估结果的字典
        """
        raise NotImplementedError
    
    def _greedy_action_selector(self, action_probs: torch.Tensor) -> torch.Tensor:
        """贪心动作选择策略。
        
        Args:
            action_probs: 动作概率分布
            
        Returns:
            选择的动作
        """
        return action_probs.argmax(-1)
    
    def _sampling_action_selector(self, action_probs: torch.Tensor) -> torch.Tensor:
        """采样动作选择策略。
        
        Args:
            action_probs: 动作概率分布
            
        Returns:
            选择的动作
        """
        return torch.multinomial(action_probs, 1).squeeze(-1)
    
    def _mixed_action_selector(self, 
                              action_probs: torch.Tensor,
                              batch_size: int,
                              greedy_first: bool = True) -> torch.Tensor:
        """混合动作选择策略（第一个使用贪心，其余使用采样）。
        
        Args:
            action_probs: 动作概率分布
            batch_size: 批次大小
            greedy_first: 是否第一个使用贪心策略
            
        Returns:
            选择的动作
        """
        if greedy_first:
            is_first_beam = torch.tensor(
                [idx == 0 for idx in range(batch_size)], 
                device=action_probs.device, 
                dtype=torch.bool
            )
            greedy_actions = action_probs.argmax(-1)
            sample_actions = torch.multinomial(action_probs, 1).squeeze(-1)
            return torch.where(is_first_beam, greedy_actions, sample_actions)
        else:
            return self._sampling_action_selector(action_probs)
    
    def set_action_selector(self, selector: Union[str, Callable]):
        """设置动作选择策略。
        
        Args:
            selector: 策略名称或自定义函数
        """
        if isinstance(selector, str):
            if selector == "greedy":
                self.action_selector = self._greedy_action_selector
            elif selector == "sampling":
                self.action_selector = self._sampling_action_selector
            elif selector == "mixed":
                self.action_selector = self._mixed_action_selector
            else:
                raise ValueError(f"Unknown action selector: {selector}")
        else:
            self.action_selector = selector
    
    def _move_to_device(self, data: Any) -> Any:
        """将数据移动到指定设备。
        
        Args:
            data: 要移动的数据
            
        Returns:
            移动后的数据
        """
        if hasattr(data, 'to'):
            return data.to(self.device)
        return data