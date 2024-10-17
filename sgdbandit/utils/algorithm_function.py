"""
Function template to use with optimization algorithms provided in algorithms.py
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseFunc(ABC):
    @abstractmethod
    def generate_grad_sample(self, x, n=1) -> list[Any]:
        raise NotImplementedError

    def __call__(self, x) -> float:
        return NotImplementedError


class Func(BaseFunc):
    """
    example of function with noisy gradient
    """

    def __init__(self, d) -> None:
        self.d = d
        H = np.random.rand(d, d)
        u, _, vh = np.linalg.svd(H, full_matrices=False)
        U = u @ vh
        diag = np.diag(np.random.rand(self.d) * 2 + 2)
        mu = np.random.rand(d)
        self.mu = mu
        self.A = U.T @ diag @ U

    def _noise(self):
        return np.random.standard_cauchy(size=self.d)

    @property
    def L(self):
        return np.max(np.linalg.eigvals(self.A))

    def generate_grad(self, x, n=1):
        tmp = self.A @ (x - self.mu)
        rez = [tmp + self._noise for _ in range(n)]
        return rez

    def __call__(self, x) -> float:
        return 0.5 * (x - self.mu).T @ self.A @ (x - self.mu)
