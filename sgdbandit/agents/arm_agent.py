from copy import deepcopy

import numpy as np

from ..utils.arms import BaseArm
from .abstract_agent import AbstractAgent


class ArmAgent(AbstractAgent):
    def __init__(self, n_actions: int, coeff: float = 1.0, *, arm: BaseArm, init_steps: int = 0) -> None:
        """

        Parameters:
        init_arm : arm with another parameters? that will be used to init
        """
        assert init_steps >= 0
        super().__init__(n_actions=n_actions, remember_reward_history=False)
        self._clear_arm = deepcopy(arm)
        self._total_calls = 0  # this is the number of times the arms has been selected
        self.coeff = coeff
        self.delta = arm.delta
        self.arms = [deepcopy(arm) for i in range(n_actions)]
        self._init_steps = init_steps
        self._un_select()

    def reset(self):
        tmp_name = self.name
        self.__init__(self.n_actions, self.coeff, arm=self._clear_arm, init_steps=self._init_steps)
        self.name = tmp_name

    def _select(self, action):
        self.selected = True
        self.selected_arm = action
        self.selected_count = 0
        self.selected_rewards = []

    def _un_select(self):
        self.selected = False
        self.selected_arm = None
        self.selected_rewards = []
        self.selected_count = 0

    def get_action(self):
        if self.selected:
            return self.selected_arm

        t = self._total_calls
        t_pulls = self._total_pulls
        if t < self.n_actions:
            self._select(t)
            return t
        x_s = np.array([self.arms[i].x for i in range(self.n_actions)]).reshape(-1)

        bounds = x_s + (self.coeff * np.sqrt(np.log(t_pulls) / self._history_pull)).reshape(-1)
        arm = np.argmax(bounds)
        self._select(arm)
        return arm

    def _update(self):
        arm = self.arms[self.selected_arm]
        arm.update(self.selected_rewards)
        return

    def _init_update(self):
        x0 = np.median(self.selected_rewards)
        self.arms[self.selected_arm].x = np.array([x0])
        return

    def update(self, action, reward):
        super().update(action, reward)
        if self.selected:
            self.selected_rewards.append(reward)
            self.selected_count += 1
            if self._total_calls < self.n_actions and self._init_steps:
                if self.selected_count >= self._init_steps:
                    self._init_update()
                    self._un_select()
                    self._total_calls += 1
            else:
                if self.selected_count >= self._clear_arm.pulls_for_update:
                    self._update()
                    self._un_select()
                    self._total_calls += 1
