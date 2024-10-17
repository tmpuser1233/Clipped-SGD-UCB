"""
there implementations of robust ucb agents
"""

import numpy as np

from .abstract_agent import AbstractAgent



class ClassicUCB(AbstractAgent):
    def reset(self):
        super().__init__(self.n_actions)

    def __init__(self, n_actions=5, R=0.99):
        super().__init__(n_actions=n_actions)
        self.R = R
        self.mu = np.zeros(self.n_actions)

    def update(self, action, reward):
        # update mean for all actions
        pulls_before = self._history_pull[action]
        self.mu[action] = (self.mu[action] * pulls_before + reward) / (pulls_before + 1)
        # classic update
        super().update(action, reward)

    def _ucb(self, t):
        ar1 = self.mu  # calculate mu_i
        ar2 = self.R * (2 * np.log(t) / self._history_pull) ** 0.5  # interval for ucb
        ucb = ar1 + ar2
        return np.argmax(ucb)

    def get_action(self):
        t = self._total_pulls
        if t < self.n_actions:
            return t
        else:
            return self._ucb(t)


class RobustUCBTruncated(AbstractAgent):
    def reset(self):
        super().__init__(self.n_actions)

    def __init__(self, n_actions=5, eps=0.25, u=0.25):
        super().__init__(n_actions=n_actions)
        self.eps = eps
        self.u = u

    def _truncated_mean_comp(self, delta):
        ar_1 = self.u / np.log(1 / delta)
        ar_2 = 1 / (1 + self.eps)
        ans = np.zeros(self.n_actions)
        for i in range(self.n_actions):
            if len(self._rewards[i]) == 0:
                ans[i] = float("inf")
            else:
                for j, x_tmp in enumerate(self._rewards[i]):
                    if abs(x_tmp) <= (ar_1 * (j + 1)) ** (ar_2):
                        ans[i] += x_tmp
                assert self._history_pull[i] == len(self._rewards[i])
                ans[i] /= len(self._rewards[i])
        return ans

    def _tr_mean(self, t):
        ar_0 = 4 * self.u ** (1.0 / (1 + self.eps))
        ar_1 = np.log((t + 1) ** 2) / (self._history_pull)
        ar_2 = self.eps / (1 + self.eps)
        Bs = self._truncated_mean_comp(1 / ((t + 1) ** 2)) + ar_0 * ((ar_1) ** (ar_2))
        # print(self._rewards)
        # print(Bs)
        return np.argmax(Bs)

    def get_action(self):
        # Место для Вашей реализации
        t = self._total_pulls
        if t < self.n_actions:
            return t
        else:
            return self._tr_mean(t)


class RobustUCBMedian(AbstractAgent):
    def reset(self):
        super().__init__(self.n_actions)

    def __init__(self, n_actions=5, eps=0.25, v=0.25):
        """
        eps - distributions have moments of order 1 + eps, eps in (0; 1]
        v - “variance factor"
        """
        super().__init__(n_actions=n_actions)
        self.eps = eps
        self.v = v

    def _median_of_means_comp(self, log_delta):
        ans = np.zeros(self.n_actions)
        for i in range(self.n_actions):
            n = self._history_pull[i]
            x_s = self._rewards[i]
            k = int(min(1 - 8 * log_delta, n / 2))
            # k = int(min(8 * np.log(1 / 8 - log_delta), n / 2))
            N = int(n / k)
            means = [np.mean(x_s[j * N : (j + 1) * N]) for j in range(k)]
            est = np.median(means)
            ans[i] = est
        return ans

    def _med_mean(self, t):
        log_delta = np.log(1 / (1 + t) ** 2)
        ar_0 = (12 * self.v) ** (1 / (1.0 + self.eps))
        ar_1 = 16 * (1 / 8.0 - log_delta) / self._history_pull
        ar_2 = self.eps / (1.0 + self.eps)
        Bs = self._median_of_means_comp(log_delta) + ar_0 * (ar_1**ar_2)
        return np.argmax(Bs)

    def get_action(self):
        # Место для Вашей реализации
        t = self._total_pulls
        if t < self.n_actions * 2:
            return t % self.n_actions
        else:
            return self._med_mean(t)


class RobustUCBCatoni(AbstractAgent):
    pass

    # def reset(self):
    #     super().__init__(
    #         self.n_actions,
    #     )

    # def __init__(self, n_actions=5, eps=0.25, v=0.25, tol=0.25):
    #     # eps - distributions have moments of order 1 + eps, eps \in (0; 1]
    #     # v - “variance factor"
    #     # u
    #     super().__init__(n_actions=n_actions)
    #     self.eps = eps
    #     self.v = v
    #     self.tol = tol
    #     self._last_catoni_mean = np.zeros(n_actions)

    # def _catonialpha(self, v, intercount, _size):
    #     lg4t = 4.0 * np.log(intercount)
    #     return (lg4t / (_size * (v + v * lg4t / (_size - lg4t)))) ** 0.5

    # def _psi(self, x):
    #     if x < 0:
    #         return -self._psi(-x)
    #     elif x > 1:
    #         return np.log(2 * x - 1) / 4 + 5.0 / 6
    #     else:
    #         return x - x * x * x / 6

    # def _dpsi(self, x):
    #     if x < 0:
    #         return self._dpsi(-x)
    #     elif x > 1:
    #         return 1 / (4 * x - 2)
    #     else:
    #         return 1.0 - x * x / 2

    # def _sumpsi(self, v, intercount, guess, arr):
    #     ans = 0
    #     a_d = self._catonialpha(v, intercount=intercount, _size=len(arr))
    #     for i in range(len(arr)):
    #         ans += self._psi(a_d * (arr[i] - guess))
    #     return ans

    # def _dsumpsi(self, v, intercount, guess, arr):
    #     ans = 0
    #     a_d = self._catonialpha(v, intercount=intercount, _size=len(arr))
    #     for i in range(len(arr)):
    #         ans += self._dpsi(a_d * (arr[i] - guess))
    #     return -a_d * ans

    # def _nt_iter(self, v, intercount, guess, arr, fguess):
    #     return guess - fguess / self._dsumpsi(v, intercount, guess, arr=arr)

    # def _catoni_mean_estimate(self, i, v, intercount, guess, arr, tol):
    #     a = self._sumpsi(v, intercount, guess, arr)
    #     nt_intercount = 0
    #     a_d = self._catonialpha(v, intercount, len(arr))
    #     realtol = tol * a_d * a_d
    #     while (a > realtol or a < -realtol) and nt_intercount < 50:
    #         guess = self._nt_iter(v, intercount, guess, arr, a)
    #         a = self._sumpsi(v, intercount, guess, arr)
    #         nt_intercount += 1

    #     self._last_catoni_mean[i] = guess
    #     return (guess, nt_intercount)

    # def _catoni_mean(self, t):
    #     lgt_4 = 4 * np.log(t + 1)
    #     ans = np.array(
    #         [
    #             self._catoni_mean_estimate(
    #                 i,
    #                 self.v,
    #                 t + 1,
    #                 self._last_catoni_mean[i],
    #                 self._rewards[i],
    #                 self.tol,
    #             )[0]
    #             + np.sqrt(2 * self.v * lgt_4 / len(self._rewards[i]))
    #             for i in range(self.n_actions)
    #         ]
    #     )
    #     return np.argmax(ans)

    # def get_action(self):
    #     # Место для Вашей реализации
    #     t = self._total_pulls
    #     if t < self.n_actions * 2:
    #         return t % self.n_actions
    #     else:
    #         return self._catoni_mean(t)
