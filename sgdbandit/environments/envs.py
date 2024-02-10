import typing as tp
from abc import ABC, abstractmethod

import numpy as np


class AbstractEnv(ABC):
    """
    Abstract class for environment.
    Custum environment should contain pull method to pull one arm

    For testing optimmal_reward and action_reward methods should be implemented.
    See examples.
    """

    def __init__(self, n_actions) -> None:

        self.n_actions = n_actions

    @property
    def num_actions(self):
        return self.n_actions

    @abstractmethod
    def pull(self, action: tp.Any) -> int | float:
        raise NotImplementedError

    def optimal_reward(self) -> int | float:
        """
        returns expected reward of the best arm
        """
        pass

    def action_reward(self, action: tp.Any) -> int | float:
        """
        returns expected reward of given action
        """
        pass


class RewardArrEnv(AbstractEnv):
    """
    envorinment with predefined rewards array
    """

    def __init__(
        self,
        reward_arr: tp.Iterable[float],
    ) -> None:
        n_actions = len(reward_arr)
        self.reward_arr = reward_arr
        self.n_actions = n_actions

    def _sample_noise(self) -> float:
        return 0.0

    def pull(self, action: int) -> float:
        rez = self.reward_arr[action] + self._sample_noise()
        return rez

    def optimal_reward(self) -> float:
        return np.max(self.reward_arr)

    def action_reward(self, action: int) -> float:
        return self.reward_arr[action]


class NormalDistributionEnv(RewardArrEnv):
    """ """

    def __init__(
        self,
        reward_arr: tp.Iterable[float],
        sigma: float = 1.0,
    ) -> None:
        super().__init__(reward_arr)
        self.sigma = sigma

    def _sample_noise(self) -> float:
        return self.sigma * np.random.rand()


class CauchyDistributionEnv(RewardArrEnv):
    """
    Environment with arms with noise distributed as Cauchy distribution

    https://en.wikipedia.org/wiki/Cauchy_distribution

    parameters:
        reward_arr: tp.Iterable[float]
        array of parameters

        gamma: float
        distribution parameter
    """

    def __init__(
        self,
        reward_arr: tp.Iterable[float],
        gamma: float = 1.0,
    ) -> None:
        super().__init__(reward_arr)
        self.gamma = gamma

    def _sample_noise(self) -> float:
        unif = np.random.rand()
        rez = self.gamma * np.tan(np.pi * (unif - 0.5))
        return rez


class CauchyPlusExpDistributionEnv(RewardArrEnv):
    """
    arm noise distribution is
    f(x) = 0.7 * Cauchy(x) + 0.3 * Exp(x+1)
    """

    def __init__(
        self,
        reward_arr: tp.Iterable[float],
        gamma: float = 1.0,
    ) -> None:
        super().__init__(reward_arr)
        self.gamma = gamma

    def _sample_noise(self) -> float:
        unif = np.random.rand()
        unif2 = np.random.rand()
        if unif > 0.3:
            rez = self.gamma * np.tan(np.pi * (unif2 - 0.5))
        else:
            rez = -1 - np.log(1 - unif2)
        return rez


class CauchyPlusParetoEnv(RewardArrEnv):
    """
    f(x) = 0.7 * Cauchy(x) + 0.3 * 3/((x + 1.5)^4) * I[x >= -1.5]
    """

    def __init__(
        self,
        reward_arr: tp.Iterable[float],
        gamma: float = 1.0,
    ) -> None:
        super().__init__(reward_arr)
        self.gamma = gamma

    def reset(self):
        pass

    def _sample_noise(self) -> float:
        unif = np.random.rand()
        unif2 = np.random.rand()
        if unif > 0.3:
            rez = self.gamma * np.tan(np.pi * (unif2 - 0.5))
        else:
            rez = -1 + (1 - unif2) ** (1 / 3)
        return rez


class FrechetDistribution(RewardArrEnv):
    """
    Environment with arms with noise distributed as Fréchet distribution

    https://en.wikipedia.org/wiki/Fréchet_distribution

    parameters:
        reward_arr: tp.Iterable[float]
        array of parameters

        alpha: float
        distribution parameter. Noise will has standardized moment k for k < alpha
    """

    def __init__(
        self,
        reward_arr: tp.Iterable[float],
        alpha: float = 0.5,
    ) -> None:
        assert alpha > 0
        super().__init__(reward_arr=reward_arr)
        self.alpha = alpha

    def _sample_noise(self) -> float:
        unif = np.random.rand()
        noise = (-np.log(unif)) ** (-1.0 / self.alpha)
        return noise
