from ..utils import arms, mean_estimators
from . import arm_agent

# def SSTM_SMoM(n_actions, coeff,n=2, m=2, K:int = 10_000, R: float = 10.0, theta: float = 0.001,init_steps: int = 0):
#     mean_estimator = mean_estimators.SMoM(n, m, theta=theta)
#     arm = arms.SSTMArm(mean_estimator, K, R)
#     agent = arm_agent.ArmAgent(n_actions=n_actions, coeff=coeff, arm=arm, init_steps=init_steps)
#     agent.name = "SSTM_SMoM"
#     return agent


def SGD_SMoM(n_actions, coeff, n=1, m=0, T: int = 10_000, R: float = 10.0, theta: float = 0.001, init_steps: int = 0):
    mean_estimator = mean_estimators.SMoM(n, m, theta=theta)
    arm = arms.SGDArm(mean_estimator, T, R)

    agent = arm_agent.ArmAgent(n_actions=n_actions, coeff=coeff, arm=arm, init_steps=init_steps)
    agent.name = "SGD_SMoM"
    return agent
