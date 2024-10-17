import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import colors

##################################################################################
# set there style and etc parameters of firuges
##################################################################################
LINESTYLES = [
    ("d", "dashdot"),
    ("d", "dotted"),
    ("d", "solid"),
    ("d", "dashed"),
    ("dashdotdotted", (0, (3, 5, 1, 5, 1, 5))),
]


COLORMAP_NAME = "tab20"
DPI = 500
FIGSIZE = (17, 8)
FONTSIZE = 20


def get_fig_set_style(lines_count):
    cmap = plt.colormaps.get_cmap(COLORMAP_NAME)
    colors_list = [colors.to_hex(cmap(i)) for i in range(lines_count)]
    colors_list = ["blue", "y", "black", "purple", "red", "c", "y", "g"]
    sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
    fig, ax = plt.subplots(dpi=DPI)
    plt.grid(which="both")
    return fig, ax, colors_list


def draw(filename):

    agent_names = [
            "UCB c=0.1",
            "UCB c=1.0",
            "SGD-UCB 0.1",
            "SGD-UCB",
            "SGD-UCB-Median 0.1",
            "SGD-UCB-Median",
            "SGD-UCB-SMoM 0.1",
            "SGD-UCB-SMoM",
            "RUCB",
            "APE",
            r"APE $p = 1.25 + \alpha$",
            r"APE, $p = 2$",
            "Heavy-Inf"
    ]

    ignore = [
            # "UCB c=0.1",
            # "UCB c=1.0",
            # "SGD-UCB 0.1",
            # "SGD-UCB 0.2",
            # "SGD-UCB-Median 0.1",
            # "SGD-UCB-Median 0.2",
            # "SGD-UCB-SMoM 0.1",
            # "SGD-UCB-SMoM 0.2",
            # "RUCB",
            # "APE",
            # "APE +0.25",
            # "APE 2",
            # "Heavy-Inf"
    ]
    with open(filename, "r") as f:
        arr = json.load(f)
    alg_names = arr["0.0"].keys()
    fig, ax, colors = get_fig_set_style(len(alg_names))
    x_s = arr["rewards_list"] #np.linspace(0.0, 10.0, 25)
    del arr["rewards_list"]
    alg_rez = {alg_name: [] for alg_name in alg_names}
    for key, val in arr.items():
        for alg_name, regret in val.items():
            alg_rez[alg_name].append(regret)
    plotted_num = 0
    for new_name, (name, rez) in zip(agent_names, alg_rez.items()):
        if new_name in ignore:
            continue
        # if new_name[-1].isdigit():
        #     new_name = new_name[:-4]
        ax.plot(x_s, rez, label=new_name, color=colors[plotted_num])
        plotted_num += 1
    ax.legend()
    ax.set_xlabel(r"$\Delta$")
    ax.set_ylabel("Expected regret")
    plt.grid(which="both")
    plt.grid()
    plt.legend(loc="upper right")
    return fig
