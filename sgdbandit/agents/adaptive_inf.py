from .abstract_agent import AbstractAgent

import numpy as np
import numpy.random as rn
import scipy.optimize as opt
from scipy.optimize import Bounds, LinearConstraint, minimize

ALPHA = 1.
SIGMA = 1.

class AdaptiveInf(AbstractAgent):
    def __init__(self, n_actions, T, remember_reward_history=True):
        super().__init__(n_actions, remember_reward_history)

        self.T = T

        self.S = 0
        self.J = 0


        # self.inverse_exponent = 1.0 / (self.alpha - 1.0)  #: Store :math:`\frac{1}{\alpha-1}` to only compute it once.
        self.cumulative_losses = np.zeros((n_actions,), dtype = float)  #: Keep in memory the vector :math:`\hat{L}_t` of cumulative (unbiased estimates) of losses.
        self.thrashed = np.zeros((n_actions,))
        self.arms_stat = np.zeros((n_actions,))

        self.t = 1
        self.weights = np.ones(n_actions, dtype = float)/n_actions

        self._initial_exploration = rn.permutation(n_actions)
    
    def reset(self):
        # super().reset()
        self.__init__(self.n_actions, self.T, self.remember_reward_history)
    
    def get_action(self):
        if self.t < self.n_actions:
            # DONE we could use a random permutation instead of deterministic order!
            return self._initial_exploration[self.t]
        else:
            return rn.choice(self.n_actions, p=self.weights)
    
    # @property/
    # def name(self):
        # return str(self.__class__.__name__)
    
    @property
    def lambd(self):
        return 2 ** self.J
    
    @property
    def eta(self):
        r""" Decreasing learning rate, :math:`\eta_t = \frac{1}{\sqrt{t}}`."""
        return 1./(self.lambd * self.t ** 0.5)
    
    def update(self, arm, reward):
        self.t += 1
        self.arms_stat[arm] += 1
        # super(TsallisInf, self).getReward(arm, reward)  # XXX Call to Exp3
        # normalize reward to [0,1]

        loss = - reward
        # reward = (reward - self.lower) / self.amplitude
        # for one reward in [0,1], loss = 1 - reward
        # biased_loss = 1.0 - reward
        # unbiased estimate, from the weights of the previous step
        # unbiased_loss = biased_loss / self.weights[arm]
        eta_t = self.eta        

        # считаем трешхолд для вырезания лосса
        threshold =  (1 - 2**(-1/3))/eta_t * (self.weights[arm] ** 0.5)
        if abs(loss) > threshold:
            self.thrashed[arm] += 1
            c_t = loss
            loss = 0
        else:
            unbiased_loss = loss / self.weights[arm]
            c_t = 2 * eta_t * (self.weights[arm] ** -0.5) * (loss ** 2)
            self.cumulative_losses[arm] += unbiased_loss

        self.S = self.S + c_t # тут поставил минус, потому что лосс должен накопиться если мы неправильно оцениваем
        tmp = (self.n_actions * (self.T  + 1))
        # print(self.J, (2 ** self.J) * tmp ** 0.5, self.S, threshold, loss, c_t)
        if (2 ** self.J) * tmp ** 0.5 < self.S:
            self.J = max(self.J + 1, np.ceil(np.log2(c_t/tmp)) + 1)
            self.S = c_t 
# 
#  optimizer
# 
        def f(x):
            return eta_t * self.cumulative_losses @ x - 2 * np.sum(x**(1/2))
        # LinearConstraint([[1, 2], [2, 1]], [-np.inf, 1], [1, 1])
        # {'type': 'ineq', 'fun': constraint1}

        constraints = [LinearConstraint(np.ones((self.n_actions,)).tolist(),lb = 1., ub = 1.)]
        rez = minimize(f, np.ones((self.n_actions))/self.n_actions, bounds= Bounds(np.zeros(self.n_actions).tolist()),
                            constraints=constraints)
                
        new_weights = rez.x
        # print(new_weights)
        # 
        # 1. solve f(x)=1 to get an approximation of the (unique) Lagrange multiplier x
        # def objective_function(x):
        #     return (np.sum( (eta_t * (self.cumulative_losses - x)) ** self.inverse_exponent ) - 1) ** 2

        # result_of_minimization = opt.minimize_scalar(objective_function)
        # # result_of_minimization = opt.minimize(objective_function, 0.0)  # XXX is it not faster?
        # x = result_of_minimization.x

        # # 2. use x to compute the new weights
        # new_weights = ( eta_t * (self.cumulative_losses - x) ) ** self.inverse_exponent

        # print("DEBUG: {} at time {} (seeing reward {} on arm {}), compute slack variable x = {}, \n    and new_weights = {}...".format(self, self.t, reward, arm, x, new_weights))  # DEBUG

        # XXX Handle weird cases, slow down everything but safer!
        if not np.all(np.isfinite(new_weights)):
            new_weights[~np.isfinite(new_weights)] = 0  # set bad values to 0
        # Bad case, where the sum is so small that it's only rounding errors
        # or where all values where bad and forced to 0, start with new_weights=[1/K...]
        if np.isclose(np.sum(new_weights), 0):
            # Normalize it!
            new_weights[:] = 1.0

        # 3. Renormalize weights at each step
        new_weights /= np.sum(new_weights)

        # 4. store weights
        self.weights =  new_weights
