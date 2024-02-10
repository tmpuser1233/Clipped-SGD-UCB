from abc import ABC, abstractmethod, abstractproperty
from typing import Any

import numpy as np


class BaseMeanEstimator(ABC):
    @abstractmethod
    def __call__(self, sample_list) -> Any:
        raise NotImplementedError

    @abstractproperty
    def sample_len(self) -> int:
        """
        should return number of samples to make estimation
        """
        raise NotImplementedError


class SMoM(BaseMeanEstimator):
    """
    Sample Median of Means
    https://arxiv.org/pdf/2311.04161.pdf
    """

    def __init__(self, n=2, m=2, d=1, *, theta) -> None:
        self.n = n
        self.m = m
        self.d = d
        self.theta = theta

    @property
    def sample_len(self):
        return (2 * self.m + 1) * self.n

    def _smoothing(self):
        return self.theta * np.random.multivariate_normal(mean=np.zeros(shape=(self.d)), cov=np.eye(self.d))

    def __call__(self, sample_list: list) -> Any:
        """

        Params:
            sample_list: list[float|vector]

            list of sampled gradients
        """
        # assert len(sample_list) == self.sample_len
        n = len(sample_list) // (2 * self.m + 1)
        # print(n, len(sample_list))
        means = [np.mean(sample_list[j * n : (j + 1) * n], axis=0) + self._smoothing() for j in range(2 * self.m + 1)]
        means = np.stack(means, axis=0)
        medians = np.median(means, axis=0)
        return medians


class SampleAverage(BaseMeanEstimator):
    def __init__(self, sample_len) -> None:
        self._sample_len = sample_len

    @property
    def sample_len(self):
        return self._sample_len

    def __call__(self, sample_list: list) -> Any:
        assert len(sample_list) == self.sample_len
        means = np.mean(sample_list, axis=0)
        return means
