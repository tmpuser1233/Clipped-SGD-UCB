"""
APE algorithm from https://arxiv.org/pdf/2010.12866
"""

from typing import Any
import numpy as np
import scipy.stats as sps # for perturbation

from .abstract_agent import AbstractAgent

class psi_p:
    def __init__(self, p) -> None:
        self.p = p
        if p < 1 + 1e-6:
            self.b = 1
        else:
            self.b = ( 2 * ((2-p)/(p-1))**(1 - 2/p) + ((2-p)/(p-1))**(2 - 2/p) )**(-p/2)
    
    def __call__(self, x) -> Any:
        """
        there x may be a vector of values.
        in this case we get sum of them before return
        """
        if  not isinstance(x, np.ndarray):
            x = np.array([x])
        positives = (x >= 0)
        psi = np.sum(np.log(self.b * np.power(np.abs(x[positives]), self.p) + x[positives] + 1)) +\
            np.sum(np.log(self.b * np.power(np.abs(x[~positives]), self.p) - x[~positives] + 1))
        
        return psi


class APE(AbstractAgent):
    def __init__(self, n_actions, c, p, F_inv = None):
        super().__init__(n_actions,)
        if F_inv is None:
            F_inv = lambda x: sps.chi2.ppf(x, df = 1, scale = 2)
        self.F_inv = F_inv # for perturbation generation
        self.c = c
        self.p = p
        self.psi = psi_p(p)
        self._initial_exploration = np.random.permutation(n_actions)

    def reset(self):
        self.__init__(self.n_actions, self.c, self.p, self.F_inv)

    @property
    def get_perturbation(self):
        u = np.random.rand(self.n_actions)
        return self.F_inv(u)
    
    def p_robust_estimation(self):
        """
        here we cmpute estimation of reward for all arms        
        """
        arm_est = self.get_perturbation * \
                    self.c / (self._history_pull ** (1 - 1/self.p))
        for i, (n_i, r_i) in enumerate(zip(self._history_pull, self._rewards)):
            arm_est[i] += self.c / (n_i ** (1 - 1/self.p)) * \
                self.psi(np.array(r_i)/(self.c * (n_i**(1/self.p))))
        return np.argmax(arm_est)


    def get_action(self):
        # Место для Вашей реализации
        t = self._total_pulls
        if t < self.n_actions:
            return self._initial_exploration[t]
        else:
            return self.p_robust_estimation()
    