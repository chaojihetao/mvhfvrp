import abc
import torch
from os.path import join as pjoin
from typing import Iterable, Optional
from tensordict.tensordict import TensorDict

from framework.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class EnvBase(metaclass=abc.ABCMeta):
    """Base class for RL4CO environments.
    The environment has the usual methods for stepping, resetting, and getting the specifications of the environment
    that shoud be implemented by the subclasses of this class.
    It also has methods for getting the reward, action mask, and checking the validity of the solution, and
    for generating and loading the datasets (supporting multiple dataloaders as well for validation and testing).

    # Environment
    #   0. __init__()
    #   1. reset() --> env info, state, action set
    #   2. get_legal_action() --> legal action
    #   3. step() --> next_state, reward, done
    #       3.1 apply_action() --> next state
    #       3.2 get_reward() --> reward
    #       3.3 is_terminal() --> done
    #   4. get_nn_model_input()
    """

    def __init__(
            self,
            generator = None,
            batch_size: Optional[list] = None,
            device: str = 'cpu',
            
    ):
        self.generator = generator
        self.batch_size = batch_size if batch_size is not None else [1]
        self.device = device

    # === 功能性函数 ===
    def reset(self):
        """Reset function to call at the beginning of each episode,
           update environment information, action set and get current state
        """
        problem = self.generator(self.batch_size)
        problem = problem.to(self.device)
        state = self._reset(problem)
        return state

    def step(self, state, action):
        """Step function to call at each step of the episode containing an action."""
        # next_state = self.apply_action(state, action)
        # reward = self.get_reward(state, action)
        # done = self.is_terminal(next_state)
        # next_state = self._get_action_mask(next_state)    # Check if any environment has no legal actions
        next_state, reward, done = self._step(state, action)
        return next_state, reward, done

    def get_action_mask(self, state):
        """Get the action mask for the current state."""
        state = self._get_action_mask(state)
        return state

    # def get_nn_model_input(self, state):
    #     """Get the nn model input from the current state.
    #         1. state_feature
    #         2. legal_action_feature
    #     """
    #     td = self._get_nn_model_input(state)
    #     return td

    # def apply_action(self, state, action):
    #     """Apply the action to the current state and return the next state."""
    #     next_state = self._apply_action(state, action)
    #     return next_state

    # def get_reward(self, state, action):
    #     """Function to compute the reward. Can be called by the agent to compute the reward of the current state
    #     This is faster than calling step() and getting the reward from the returned TensorDict at each time for CO tasks
    #     """
    #     if self.check_solution:
    #         self._check_solution_validity(state, action)
    #     return self._get_reward(state, action)

    # def is_terminal(self, state):
    #     """Check if the episode is done."""
    #     done = self._is_terminal(state)
    #     return done
    

    # === 实例化函数 ===
    @abc.abstractmethod     
    def _reset(self, problem):
        """Reset function to call at the beginning of each episode"""
        raise NotImplementedError  # 实例化时，必须实现该函数

    @abc.abstractmethod  
    def _get_action_mask(self, state):
        """Get the legal actions for the current state."""
        raise NotImplementedError

    @abc.abstractmethod   
    def _step(self, state, action):
        """Step function to call at each step of the episode containing an action.
        Gives the next observation, reward, done
        """
        raise NotImplementedError

    # @abc.abstractmethod  
    # def _get_nn_model_input(self, state):
    #     """Get the nn model input from the current state."""
    #     raise NotImplementedError
    