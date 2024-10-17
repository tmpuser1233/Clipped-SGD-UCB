import json
import pathlib
import shutil
from copy import deepcopy
from datetime import datetime

import dill
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from . import drawer

SAVEDIR = pathlib.Path("experiments")

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def get_trials_regret(
    env,
    agents,
    n_steps=5000,
    n_trials=100,
    n_jobs=8,
    verbose = False
):
    scores = {agent.name: [None for i in range(n_trials)] for agent in agents}
    def exp_trial(env, agents):
        scores = {agent.name: [0 for i in range(n_steps)] for agent in agents}

        for agent in agents:
            agent.reset()

        for i in range(n_steps):
            optimal_reward = env.optimal_reward()

            for agent in agents:
                action = agent.get_action()
                reward = env.pull(action)
                agent.update(action, reward)
                scores[agent.name][i] += optimal_reward - env.action_reward(action)
        
        if verbose:
            for agent in agents:        
                if hasattr(agent, "arms_stat"):
                    print(f"{agent.name}:\nlosses  {agent.cumulative_losses}\n\
                    pulls          : {agent.arms_stat}\n\
                    thrashed pulls : {agent.thrashed}")
                elif hasattr(agent, "_history_pull"):
                    print(f"{agent.name}:\n             pulls: {agent._history_pull}")

        return scores

    delayed_exp_trial = delayed(exp_trial)
    parallel = Parallel(n_jobs=n_jobs, return_as="generator")(delayed_exp_trial(env, agents) for _ in range(n_trials))
    for i, trial_rez in tqdm(enumerate(parallel)):
        for key, val in scores.items():
            val[i] = trial_rez[key]    
    return scores


#  experiment setter
class Experiment:
    def __init__(
        self, agent_list, environment, n_steps, n_trials, name: str | None = None, description: str = "", save_rez=False,
        verbose = False, savedir = SAVEDIR
    ):
        self.savedir = savedir
        self.verbose = verbose
        self.save_rez = save_rez
        self.agent_list = agent_list
        self.environment = environment
        self.n_steps = n_steps
        self.n_trials = n_trials
        if name is None:
            name = datetime.now().time().strftime("%d_%H_%M_%S")
        self._name = name
        self._data = {
            "agent_list": agent_list,
            "environment": environment,
            "n_steps": n_steps,
            "n_trials": n_trials,
            "name": self._name,
        }
        self._description = description
        self.can_save_called = False

    @property
    def name(self):
        return self._name

    def thinn_rez(self, compress_steps = 100):
        """
        just get every compress_steps element
        """
        assert hasattr(self, "_rez"), "do an experiment first"
        for agent, rez in self._rez.items():
            self._rez[agent] = [lst[::compress_steps] for lst in rez]
        

    def run(self, n_jobs=8):
        self._rez = get_trials_regret(self.environment, self.agent_list, self.n_steps, self.n_trials, n_jobs, self.verbose)

    def plot(self):
        self._fig, self._fig_data = drawer.plot(self._rez)

    def delete(
        self,
    ):
        if hasattr(self, "_path"):
            assert self._path.parent.name.startswith(
                self.savedir.name
            ), f"{self._path.name} do not start with {self.savedir.name}"
            shutil.rmtree(self._path)

    def _save(self, tmp, path):
        if "data" in tmp:
            with open(path / "data.exp", "wb") as f:
                dill.dump(tmp["data"], f)
        if 'rez' in tmp:
        #     if self.save_rez:
            with open(path /'rez.json', 'w') as f:
                json.dump(tmp['rez'], f, cls=NpEncoder)
        if "fig_data" in tmp:
            with open(path / "fig_data.json", "w") as f:
                json.dump(tmp["fig_data"], f, cls=NpEncoder)
    
    def can_save(self, filename: str | None = None):
        if not self.can_save_called:
            if not self.savedir.exists():
                self.savedir.mkdir()
            if filename is None:
                filename = f"{self._name}"
            path = self.savedir / filename
            print(path)
            assert not path.exists(), "try to rewrite existing file"
            path.mkdir()
            self.can_save_called = True

    def save(self, filename: str | None = None):

        assert hasattr(self, "_rez"), "do an experiment first"

        if filename is None:
            filename = f"{self._name}"
        path = self.savedir / filename

        if not self.can_save_called:
            if not self.savedir.exists():
                self.savedir.mkdir()            
            assert not path.exists(), "try to rewrite existing file"
            path.mkdir()
          
        self._path = deepcopy(path)

        tmp = {"data": self._data, "rez": self._rez}

        if hasattr(self, "_fig_data"):
            tmp["fig_data"] = self._fig_data
        self._save(tmp, path)

        with open(path / "description.txt", "w") as f:
            f.write(str(self._description))

        if hasattr(self, "_fig"):

            path = path / "images"
            path.mkdir()
            for name, fig in self._fig.items():
                fig.tight_layout()
                fig.savefig(str(path / f"{name}_image.png"))
                fig.savefig(str(path / f"{name}_image.pdf"))

                data = np.array(fig.canvas.buffer_rgba())
                weights = [0.2989, 0.5870, 0.1140]
                data = np.dot(data[..., :-1], weights)
                plt.imsave(str(path / f"{name}_image_gray.png"), data, cmap="gray")
                plt.imsave(str(path / f"{name}_image_gray.pdf"), data, cmap="gray")

                plt.close(fig)
        return
