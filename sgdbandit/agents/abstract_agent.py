from abc import ABCMeta, abstractmethod

import numpy as np


class AbstractAgent(metaclass=ABCMeta):
    def __init__(self, n_actions, remember_reward_history=True):
        self._total_pulls: int = 0
        self.n_actions: int = n_actions
        self._history_pull = np.zeros(self.n_actions).astype(int)

        self.remember_reward_history = remember_reward_history
        if self.remember_reward_history:
            self._rewards = [[] for i in range(self.n_actions)]
        if not hasattr(self, "_name"):
            self._name = self.__class__.__name__

    @property
    def name(self):
        return str(self.__class__.__name__)
    
    @abstractmethod
    def reset(self):
        """
        reset parameters that accumulated during usage.
        """
        raise NotImplementedError

    @abstractmethod
    def get_action(self):
        """
        select action and return it
        """
        raise NotImplementedError

    def update(self, action, reward):
        """
        update accumulated parameters
        """
        self._total_pulls += 1
        self._history_pull[action] += 1
        if self.remember_reward_history:
            self._rewards[action].append(reward)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, arg):
        self._name = arg
