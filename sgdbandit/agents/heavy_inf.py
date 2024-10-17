from .abstract_agent import AbstractAgent

import numpy as np
import numpy.random as rn
import scipy.optimize as opt
from scipy.optimize import Bounds, LinearConstraint, minimize

ALPHA = 1.
SIGMA = 1.

class HeavyInf(AbstractAgent):
    def __init__(self, n_actions, alpha=ALPHA, sigma = SIGMA, remember_reward_history=True):
        super().__init__(n_actions, remember_reward_history)

        self.alpha = alpha  #: Store the constant :math:`\alpha` used by the Online-Mirror-Descent step using :math:`\alpha` Tsallis entropy.
        self.alpha_inv = 1./ alpha
        self.sigma = sigma

        if alpha >= 2:
            self.theta = 1 - 2** (-1./3)
        else:
            self.theta = min(1 - 2**(-(alpha - 1)/(2 * alpha - 1)), (2 - 2/alpha)**(1/(2 - alpha)))
        # self.theta = 1

        # self.inverse_exponent = 1.0 / (self.alpha - 1.0)  #: Store :math:`\frac{1}{\alpha-1}` to only compute it once.
        self.cumulative_losses = np.zeros((n_actions,), dtype = float)  #: Keep in memory the vector :math:`\hat{L}_t` of cumulative (unbiased estimates) of losses.
        self.thrashed = np.zeros((n_actions,))
        self.arms_stat = np.zeros((n_actions,))
        self.t = 1
        self.weights = np.ones(n_actions, dtype = float)/n_actions

        self._initial_exploration = rn.permutation(n_actions)
    
    def reset(self):
        # super().reset()
        self.__init__(self.n_actions, self.alpha, self.sigma, self.remember_reward_history)
    
    def get_action(self):
        if self.t < self.n_actions:
            # DONE we could use a random permutation instead of deterministic order!
            return self._initial_exploration[self.t]
        else:
            return rn.choice(self.n_actions, p=self.weights)
    
    @property
    def eta(self):
        r""" Decreasing learning rate, :math:`\eta_t = \frac{1}{\sqrt{t}}`."""
        return 1./(self.sigma * (self.t) ** (self.alpha_inv))
    
    def update(self, action, reward):
        super().update(action,reward)
        self.t += 1
        self.arms_stat[action] += 1

        loss = - reward
        eta_t = self.eta        

        # считаем трешхолд для вырезания лосса

        threshold = self.theta * (eta_t**-1) * (self.weights[action] ** (self.alpha_inv)) 

        # print(threshold, reward)

        if (abs(loss) > threshold): 
            self.thrashed[action] += 1
            loss = 0.

        unbiased_loss = loss / self.weights[action]

        self.cumulative_losses[action] += unbiased_loss

# 
#  optimizer
# 
        def f(x):
            return eta_t * self.cumulative_losses @ x - self.alpha * np.sum(x**(1/self.alpha))

        constraints = [LinearConstraint(np.ones((self.n_actions,)).tolist(),lb = 1., ub = 1.)]
        rez = minimize(f, np.ones((self.n_actions))/self.n_actions, bounds= Bounds(np.zeros(self.n_actions).tolist()),
                            constraints=constraints)
                
        new_weights = rez.x    

        # 3. Renormalize weights at each step
        new_weights /= np.sum(new_weights)

        # 4. store weights
        self.weights =  new_weights

