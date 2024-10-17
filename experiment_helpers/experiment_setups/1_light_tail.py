import inspect
import sys

import numpy as np

from experiment_helpers.utils import Experiment
from sgdbandit import agents, environments

RANDOM_SEED = 123

eps = 0.3
agents_partly = [
    lambda n_actions, R, K: agents.SGD_SMoM(n_actions=n_actions, R=R, T=K, m=0, n=1, coeff=0.07, init_steps=1),
    lambda n_actions, R, K: agents.SGD_SMoM(n_actions=n_actions, R=R, T=K, m=0, n=1, coeff=0.1, init_steps=1),
    lambda n_actions, R, K: agents.SGD_SMoM(n_actions=n_actions, R=R, T=K, m=0, n=1, coeff=0.12, init_steps=1),
    lambda n_actions: agents.ClassicUCB(n_actions=n_actions, R=0.2),
    lambda n_actions: agents.ClassicUCB(n_actions=n_actions, R=0.5),
    lambda n_actions: agents.ClassicUCB(n_actions=n_actions, R=1.0),  # так как показывали разные результаты
    lambda n_actions, eps, R: agents.RobustUCBMedian(n_actions=n_actions, eps=0.0, v=R),
    lambda n_actions, eps, T: agents.APE(n_actions, c = 0.1, p = 2),
]

agent_names = [
    'SGD-SMOM 0.07',
    'SGD-SMOM 0.1',
    'SGD-SMOM 0.12',
    'UCB 0.2',
    'UCB 0.5',
    'UCB 1.0',
    "RucbMedian"
    "APE 2"    
    ]

# assert len(agents_partly) == len(agent_names)


env_partly = [lambda reward_arr: environments.NormalDistributionEnv(reward_arr=reward_arr, sigma=1)]

reward_arr_lists = [
    np.array(list(range(10))) / 50,
    np.array(list(range(100))) / 50,
]


def experiment_from_rewards_list(reward_arr, env_partly, agents_partly, K=5000, n_trials=100, exp_name=None, exp_num=0):
    assert exp_name is not None
    R = np.max(np.abs(reward_arr))
    n_actions = len(reward_arr)
    env = env_partly(reward_arr)
    eps = 1
    agent_list = []
    T = K

    for i, agent in enumerate(agents_partly):
        initargs = inspect.signature(agent).parameters.keys()
        argvals = {}
        for elem in initargs:
            argvals[elem] = locals()[elem]
        agent = agent(**argvals)
        agent.name = f"ag_num_{i}"
        agent_list.append(agent)
    name = f"{exp_name}_{exp_num}"
    description = {"rewards": reward_arr, "n_trials": n_trials, "K": K, "env": str(env.__class__.__name__)}
    experiment = Experiment(
        agent_list=agent_list, environment=env, n_steps=K, n_trials=n_trials, name=name, description=description
    )
    return experiment


def init_experiments_list(name: str = "experiments_name"):
    experiments = []
    for i, reward_arr in enumerate(reward_arr_lists):
        for j, env in enumerate(env_partly):
            exp_num = f"_reward_{i}_env_{j}"
            exp = experiment_from_rewards_list(
                reward_arr=reward_arr,
                env_partly=env,
                agents_partly=agents_partly,
                K=3_000,
                n_trials=150,
                exp_name=name,
                exp_num=exp_num,
            )
            experiments.append(exp)
    return experiments


if __name__ == "__main__":
    # print(sys.argv)
    np.random.seed(RANDOM_SEED)
    assert len(sys.argv) > 1, "pls provide experiment name"
    name = "1_exp_light_tail"
    # path = Path(name)
    experiments = init_experiments_list(name)
    for exp in experiments:
        try:
            exp.run()
            # exp.plot()
            exp.save()
        except Exception as e:
            print(e)
            continue
        del exp
