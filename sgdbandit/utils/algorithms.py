"""
Optimization algorithms from paper
https://arxiv.org/pdf/2311.04161.pdf
"""

import numpy as np
import numpy.typing as npt

from .algorithm_function import BaseFunc
from .mean_estimators import BaseMeanEstimator


def _clip(vect: npt.ArrayLike, lambd):
    norm = np.linalg.norm(
        vect,
    )
    if norm == 0:
        return np.zeros_like(vect, dtype=float)
    else:
        return min(1, lambd / norm) * vect


class ClippedSSTM:
    def __init__(
        self,
        *,
        x0: npt.ArrayLike,
        func: BaseFunc,
        mean_estimator: BaseMeanEstimator,
        a: float,
        alpha0: float,
        L: float,
        R: float,
        delta: float,
    ) -> None:
        self.func = func
        self.mean_estimator = mean_estimator
        self.x = x0.copy()
        self.y = x0.copy()
        self.z = x0.copy()
        self.a = a
        self.L = L
        self.R = R
        self.delta = delta

        self.k = 0
        self.alpha_prev = alpha0
        self.A_prev = alpha0

        self.alpha_next = None
        self.A_next = None
        self.lambd_next = None

    def param_next_update(self):
        self.alpha_next = (self.k + 2) / (2 * self.a * self.L)
        self.A_next = self.A_prev + self.alpha_next
        self.lambd_next = self.lambd_update_coeff / self.alpha_next

    def param_prev_set(self):
        self.alpha_prev = self.alpha_next
        self.A_prev = self.A_next
        self.lambd_prev = self.lambd_next

    def step(self):
        self.param_next_update()
        self.x = (self.A_prev * self.y + self.alpha_next * self.z) / self.A_next

        grads_sample = self.func.generate_grad_sample(self.x, self.mean_estimator.sample_len)
        x_meaned = self.mean_estimator(sample_list=grads_sample)

        grad_clipped = _clip(x_meaned, self.lambd_next)
        self.z = self.z - self.alpha_next * grad_clipped
        self.y = (self.A_prev * self.y + self.alpha_next * self.z) / self.A_next
        self.param_prev_set()
        self.k += 1

    def run(self, K: int, lambd_update_coeff: float | None = None, verbose=False):
        if lambd_update_coeff is None:
            self.lambd_update_coeff = self.R / np.log(4 * (K + 1) / self.delta)
        else:
            self.lambd_update_coeff = lambd_update_coeff

        if verbose:
            x_s = []
            rez = []
        for _ in range(K):
            self.step()
            if verbose:
                x_s.append(self.y)
                rez.append(self.func(self.y))
        if verbose:
            return rez, x_s
        else:
            return


class ClippedSGD:
    def __init__(
        self,
        *,
        x0: npt.ArrayLike,
        func: BaseFunc,
        mean_estimator: BaseMeanEstimator,
        L: float,
        A: float,
        R: float,
        delta: float,
    ) -> None:
        self.func = func
        self.mean_estimator = mean_estimator
        self.x = x0.copy()
        self.L = L
        self.A = A
        self.gamma = 1 / (L * A)
        self.lambd = R / (self.gamma * A)
        self.delta = delta
        self.R = R

        self.k = 0

    def step(self):
        grads_sample = self.func.generate_grad_sample(self.x, self.mean_estimator.sample_len)
        x_meaned = self.mean_estimator(sample_list=grads_sample)

        grad_clipped = _clip(x_meaned, self.lambd)
        self.x = self.x - self.gamma * grad_clipped
        self.k += 1

    def run(self, K: int, verbose=False):
        if verbose:
            x_s = []
            rez = []
        for _ in range(K):
            self.step()
            if verbose:
                x_s.append(self.y)
                rez.append(self.func(self.y))
        if verbose:
            return rez, x_s
        else:
            return


# class RestartsClippedSSTM:
#     def __init__(self, **kwargs) -> None:
#         self.kwargs = kwargs
#         self.x = kwargs['x0']
#         self.func = self.kwargs['func']

#     def run(self, iterations, verbose = False):
#         rez = []
#         for t in range(iterations):
#             R_t = self.R / (2 ** ((t)/ 2.))
#             R_t_m1 = self.R / (2 ** ((t - 1)/ 2.))
#             eps_t = 0
#             K_t = max( )
#             lambd_update_coeff = R_t /(30 *
# np.log(4 * K_t * iterations / self.delta))
#             a_t = max (...)
#             self.kwargs['a'] = a_t
#             self.kwargs['x0'] = self.x
#             sstm_t = ClippedSSTM(**self.kwargs)
#             sstm_t.run(K_t, lambd_update_coeff, verbose=verbose)
#             self.x = sstm_t.y
