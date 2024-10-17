from abc import ABC, abstractmethod, abstractproperty

import numpy as np

from . import algorithms, mean_estimators
from .algorithm_function import BaseFunc


class ArmFunc(BaseFunc):
    def __init__(self) -> None:
        self._rewards: list[float] = []

    @property
    def L(self):
        return 1.0

    @property
    def rewards(self):
        raise ValueError("can`t get")

    @rewards.setter
    def rewards(self, arg: list[float]):
        self._rewards = arg

    def generate_grad_sample(self, x, n=1):
        assert len(self._rewards) == n, f"not enought observations to generate {n} gradients"
        try:
            return x - np.array(self._rewards)
        finally:
            pass
            self.rewards = []


class BaseArm(ABC):
    @abstractmethod
    def __init__(self, mean_estimator: mean_estimators.BaseMeanEstimator, K: int = 10_000, R: float = 10.0):
        pass

    @abstractproperty
    def mean(self):
        raise NotImplementedError

    @abstractmethod
    def update(self, reward_list) -> None:
        raise NotImplementedError

    @abstractproperty
    def pulls_for_update(self) -> int:
        raise NotImplementedError


class SGDArm(BaseArm):
    def __init__(self, mean_estimator: mean_estimators.BaseMeanEstimator, T: int = 10_000, R: float = 10.0):
        self.T = T
        f = ArmFunc()
        L = f.L
        x0 = np.array([R])
        R = R
        delta = 1 / T**2
        A = np.log(4 * (T + 1) / delta)
        self.sgd = algorithms.ClippedSGD(x0=x0, func=f, mean_estimator=mean_estimator, L=L, A=A, R=R, delta=delta)

    @property
    def mean(self):
        return self.sgd.x

    @property
    def x(self):
        return self.sgd.x

    @x.setter
    def x(self, x0) -> None:
        self.sgd.x = x0.copy()

    def update(self, reward_list: list[int | float]):
        self.sgd.func.rewards = reward_list
        self.sgd.step()

    @property
    def delta(self):
        return self.sgd.delta

    @property
    def pulls_for_update(self) -> int:
        return self.sgd.mean_estimator.sample_len


# class SSTMArm(BaseArm):
#     def __init__(self, mean_estimator: mean_estimators.BaseMeanEstimator, K: int = 10000, R: float = 10):
#         f = ArmFunc()
#         L = f.L
#         x0 = np.array([R])
#         alpha0 = 0.0
#         delta = 1 / K**2
#         a = 0.1
#         self.sstm = algorithms.ClippedSSTM(
#             x0=x0, func=f, mean_estimator=mean_estimator, a=a, alpha0=alpha0, L=L, R=R, delta=delta
#         )
#         self.sstm.lambd_update_coeff = R / np.log(4 * (K + 1) / delta)

#     def set_x(self, x0):
#         self.sstm.x = x0.copy()
#         self.sstm.y = x0.copy()
#         self.sstm.z = x0.copy()
#     @property
#     def mean(self):
#         return self.sstm.y

#     def update(self, reward_list: list[int | float]):
#         self.sstm.func.rewards = reward_list
#         self.sstm.step()

#     @property
#     def delta(self):
#         return self.sstm.delta

#     @property
#     def pulls_for_update(self) -> int:
#         return self.sstm.mean_estimator.sample_len
