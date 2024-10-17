import json

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from pathlib import Path
import sys
import time

from sgdbandit import agents, environments


def get_trials_regret(
    env,
    agents,
    n_steps=5000,
    n_trials=100,
    n_jobs=10,
):
    
    thresholds = [0.1, 0.05]
    num_thresholds = len(thresholds)
    def exp_trial(agent):
        # run experiment for one algorithm and measure its working time
        scores = {"thresholds_times": [[] for i in range(num_thresholds)], "num_failures": [0 for i in range(num_thresholds)]}
        for _ in range(n_trials):
            agent.reset()

            regret = 0.
            treshold_pos = 0
            t_start = time.time()
            for t in range(1, n_steps + 1):
                optimal_reward = env.optimal_reward()
                action = agent.get_action()
                reward = env.pull(action)
                agent.update(action, reward)
                regret += optimal_reward - env.action_reward(action)

                if t > n_steps /50:
                    # чтобы алгоритм точно проработал некоторое время
                    if regret/t < thresholds[treshold_pos]:
                        scores["thresholds_times"][treshold_pos].append(time.time() - t_start)
                        treshold_pos += 1
                        if treshold_pos >= len(thresholds):
                            break # end this trial
            for j in range(treshold_pos, len(thresholds)):
                scores['num_failures'][j] += 1
        return agent.name, scores

    result = {}
    delayed_exp_trial = delayed(exp_trial)
    parallel = Parallel(n_jobs=n_jobs, return_as="generator")(delayed_exp_trial(agent) for agent in agents)
    for i, trial_rez in tqdm(enumerate(parallel)):
        ag_name, rezz = trial_rez
        result[ag_name] = rezz
        
    return result


def main():
    """
    select the necessary parameters to run and the 
    necessary algorithms here
    """
    K = 10_000  # n_steps, budget
    n_trials = 100
    reward_arr = np.array(list(range(10)))
    n_actions =len(reward_arr)
    R = 10
    env = environments.CauchyDistributionEnv(reward_arr=reward_arr, gamma=1)    

    rez = {}

    agent_list = [
        agents.SGD_SMoM(n_actions, m=0, n=1, coeff=0.1, T=K, init_steps=3, R=R),
        # agents.SGD_SMoM(n_actions, m=0, n=1, coeff=0.2, T=K, init_steps=3, R=R),
        agents.SGD_SMoM(n_actions, m=1, n=1, coeff=0.1, T=K, init_steps=3, R=R),
        # agents.SGD_SMoM(n_actions, m=1, n=1, coeff=0.2, T=K, init_steps=3, R=R),
        agents.SGD_SMoM(n_actions, m=1, n=2, coeff=0.1, T=K, init_steps=3, R=R),
        # agents.SGD_SMoM(n_actions, m=1, n=2, coeff=0.2, T=K, init_steps=3, R=R),
        agents.RobustUCBMedian(n_actions=n_actions, eps=0.0, v=R),
        agents.APE(n_actions, c = 1., p = 1 + 0.,),
        agents.APE(n_actions, c = 1., p = 1 + 0.25,),
        agents.APE(n_actions, c = 1., p = 1 + 1.,),
        # agents.HeavyInf(n_actions, alpha=1 + 0., sigma=40)

    ]
    agent_names = [
        "SGD-UCB 0.1",
        # "SGD-UCB 0.2",
        "SGD-UCB-Median 0.1",
        # "SGD-UCB-Median 0.2",
        "SGD-UCB-SMoM 0.1",
        # "SGD-UCB-SMoM 0.2",
        "RUCB",
        "APE",
        "APE +0.25",
        "APE 2",
        # "Heavy-Inf"
    ]
    assert len(agent_list) == len(agent_names)


    for name, ag in zip(agent_names, agent_list):
        ag.name = name
    
    result = get_trials_regret(env, agent_list, n_steps=K, n_trials=n_trials)    
    return result 


if __name__ == "__main__":
    assert len(sys.argv) > 1, "Please, provide experiment name."
    exp_name = sys.argv[1]
    assert exp_name.endswith('.json'), "Experiment should be saved as json."
    exp_name = Path(exp_name)

    if exp_name.exists():
        raise NameError(f"{exp_name} already exists.")    

    rez = main()
    with open(exp_name, "w") as f:
        json.dump(rez, f)
