import inspect
import sys

import numpy as np

from experiment_helpers.utils import Experiment
from sgdbandit import agents, environments

RANDOM_SEED = 123

# less than 17 algorithms
agents_partly = [
    lambda n_actions: agents.ClassicUCB(n_actions, R=0.1),
    lambda n_actions: agents.ClassicUCB(n_actions, R=1.0),
    lambda n_actions, R, K: agents.SGD_SMoM(n_actions, m=0, n=1, coeff=0.1, K=K, init_steps=3, R=R),
    lambda n_actions, R, K: agents.SGD_SMoM(n_actions, m=0, n=1, coeff=0.2, K=K, init_steps=3, R=R),
    lambda n_actions, R, K: agents.SGD_SMoM(n_actions, m=1, n=1, coeff=0.1, K=K, init_steps=3, R=R),
    lambda n_actions, R, K: agents.SGD_SMoM(n_actions, m=1, n=1, coeff=0.2, K=K, init_steps=3, R=R),
    lambda n_actions, R, K: agents.SGD_SMoM(n_actions, m=1, n=2, coeff=0.1, K=K, init_steps=3, R=R),
    lambda n_actions, R, K: agents.SGD_SMoM(n_actions, m=1, n=2, coeff=0.2, K=K, init_steps=3, R=R),
    # lambda n_actions,eps, R: agents.RobustUCBTruncated(n_actions=n_actions, eps = eps, u = R),
    lambda n_actions, eps, R: agents.RobustUCBMedian(
        n_actions=n_actions, eps=0.0, v=R
    ),  # because we found that it works
]

agent_names = [
    "UCB 0.1",
    "UCB 1.0",
    "SGD-UCB 0.1",
    "SGD-UCB 0.2",
    "SGD-UCB-Median 0.1",
    "SGD-UCB-Median 0.2",
    "SGD-UCB-SMoM 0.1",
    "SGD-UCB-SMoM 0.2",
    # "RUCB-Truncated",
    "RUCB-Median",
]

assert len(agent_names) == len(agents_partly)

env_partly = [
    lambda reward_arr: environments.CauchyDistributionEnv(reward_arr=reward_arr, gamma=3),
    lambda reward_arr: environments.CauchyDistributionEnv(reward_arr=reward_arr, gamma=1),
    lambda reward_arr: environments.FrechetDistribution(reward_arr=reward_arr, alpha=1.0),
    lambda reward_arr: environments.FrechetDistribution(reward_arr=reward_arr, alpha=1.1),
    lambda reward_arr: environments.FrechetDistribution(reward_arr=reward_arr, alpha=1.25),
    lambda reward_arr: environments.CauchyPlusExpDistributionEnv(reward_arr=reward_arr, gamma=1),
    lambda reward_arr: environments.CauchyPlusParetoEnv(reward_arr=reward_arr, gamma=1),
]

epsilons_list = [0.0, 0.0, 0.0, 0.1, 0.25, 0.0, 0.0]

assert len(epsilons_list) == len(env_partly)

reward_arr_lists = [
    np.array(list(range(10))),
    np.array(list(range(10))) / 10,
    np.array(list(range(100))) / 50,
]

num_iter_list = [
    10_000,
    10_000,
    15_000,
]

assert len(reward_arr_lists) == len(num_iter_list)


def experiment_from_rewards_list(
    reward_arr, env_partly, agents_partly, K=5000, n_trials=100, eps=0, exp_name=None, exp_num=0
):
    assert exp_name is not None
    R = np.max(np.abs(reward_arr))
    n_actions = len(reward_arr)
    env = env_partly(reward_arr)

    agent_list = []

    for ag_name, agent in zip(agent_names, agents_partly):
        initargs = inspect.signature(agent).parameters.keys()
        argvals = {}
        for elem in initargs:
            argvals[elem] = locals()[elem]
        agent = agent(**argvals)
        agent.name = ag_name
        agent_list.append(agent)
    name = f"{exp_name}_{exp_num}"
    description = {"rewards": reward_arr, "n_trials": n_trials, "K": K, "env": str(env.__class__.__name__)}
    experiment = Experiment(
        agent_list=agent_list, environment=env, n_steps=K, n_trials=n_trials, name=name, description=description
    )
    return experiment


def init_experiments_list(name: str = "experiments_name"):
    experiments = []
    for i, (reward_arr, num_iters) in enumerate(zip(reward_arr_lists, num_iter_list)):
        for j, (env, eps) in enumerate(zip(env_partly, epsilons_list)):
            exp_num = f"_reward_{i}_env_{j}"
            exp = experiment_from_rewards_list(
                reward_arr=reward_arr,
                env_partly=env,
                agents_partly=agents_partly,
                K=num_iters,
                n_trials=120,
                eps=eps,
                exp_name=name,
                exp_num=exp_num,
            )
            experiments.append(exp)
    return experiments


if __name__ == "__main__":
    # print(sys.argv)
    np.random.seed(RANDOM_SEED)
    assert len(sys.argv) > 1, "pls provide experiment name"
    name = "4_all_arms_and_envs"
    # path = Path(name)
    experiments = init_experiments_list(name)
    for exp in experiments:
        try:
            exp.run()
            exp.plot()
            exp.save()
        except Exception as e:
            print(e)
            continue
        del exp
