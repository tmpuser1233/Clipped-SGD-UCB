import json

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import sys
from pathlib import Path
from sgdbandit import agents, environments


def get_trials_regret(
    env,
    agents,
    n_steps=5000,
    n_trials=100,
    n_jobs=31,
):
    scores = {agent.name: 0.0 for agent in agents}

    def exp_trial(env, agents):
        scores = {agent.name: 0.0 for agent in agents}

        for agent in agents:
            agent.reset()

        for i in range(n_steps):
            optimal_reward = env.optimal_reward()

            for agent in agents:
                action = agent.get_action()
                reward = env.pull(action)
                agent.update(action, reward)
                scores[agent.name] += optimal_reward - env.action_reward(action)
        return scores

    delayed_exp_trial = delayed(exp_trial)
    parallel = Parallel(n_jobs=n_jobs, return_as="generator")(delayed_exp_trial(env, agents) for _ in range(n_trials))
    for i, trial_rez in tqdm(enumerate(parallel)):
        for key, val in scores.items():
            scores[key] += trial_rez[key]
    for key, val in scores.items():
        scores[key] /= n_trials
    return scores


def main():
    """
    select the necessary parameters to run and the 
    necessary algorithms here
    """
    K = 2_000  # n_steps, budget for pulls
    n_trials = 300 
    n_actions = 5
    deltas = np.linspace(0.0, 10.0, 25)

    rez = {}
    for delta in deltas:
        reward_arr = np.array([0.0] * (n_actions - 1) + [delta])
        env = environments.CauchyDistributionEnv(reward_arr=reward_arr, gamma=2)
        R = delta
        agent_list = [
            # agents.ClassicUCB(n_actions, R=0.1),
            # agents.ClassicUCB(n_actions, R=1.0),
            agents.SGD_SMoM(n_actions, m=0, n=1, coeff=0.1, T=K, init_steps=3, R=R),
            agents.SGD_SMoM(n_actions, m=0, n=1, coeff=0.2, T=K, init_steps=3, R=R),
            agents.SGD_SMoM(n_actions, m=1, n=1, coeff=0.1, T=K, init_steps=3, R=R),
            agents.SGD_SMoM(n_actions, m=1, n=1, coeff=0.2, T=K, init_steps=3, R=R),
            agents.SGD_SMoM(n_actions, m=1, n=2, coeff=0.1, T=K, init_steps=3, R=R),
            agents.SGD_SMoM(n_actions, m=1, n=2, coeff=0.2, T=K, init_steps=3, R=R),
            agents.RobustUCBMedian(n_actions=n_actions, eps=0.0, v=R),
            agents.APE(n_actions, c = 1, p = 1 + 0.,),
            agents.APE(n_actions, c = 1, p = 1 + 0.25,),
            agents.APE(n_actions, c = 1, p = 1 + 1.,),
            agents.HeavyInf(n_actions, alpha=1 + 0., sigma=40)

        ]
        agent_names = [
            # "UCB c=0.1",
            # "UCB c=1.0",
            "SGD-UCB 0.1",
            "SGD-UCB 0.2",
            "SGD-UCB-Median 0.1",
            "SGD-UCB-Median 0.2",
            "SGD-UCB-SMoM 0.1",
            "SGD-UCB-SMoM 0.2",
            "RUCB",
            "APE",
            "APE +0.25",
            "APE 2",
            "Heavy-Inf"
        ]
        assert len(agent_list) == len(agent_names)
        
        for name, ag in zip(agent_names, agent_list):
            ag.name = name
        scores = get_trials_regret(env, agent_list, n_steps=K, n_trials=n_trials)
        rez[delta] = scores
    rez["rewards_list"] = deltas.tolist()
    return rez


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
