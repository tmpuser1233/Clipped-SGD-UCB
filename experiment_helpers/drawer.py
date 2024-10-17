import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

##################################################################################
# set there style and etc parameters of firuges
##################################################################################
LINESTYLES = [ 
    ("d", "dashdot"),
    ("d", "dotted"),
    ("d", "solid"),
    ("d", "dashed"),
    ("densely dashdotdotted", (0, (3, 1, 1, 1, 1, 1))),
    ("densely dashed", (0, (5, 1))),
     ("densely dotted", (0, (1, 1))),
    ("long dash with offset", (1, (0, 1))),
    ("loosely dashdotdotted", (0, (3, 2, 1, 5, 1, 5))),
    ("densely dashdotted", (0, (3, 1, 1, 1))),
    ("dashdotdotted", (0, (3, 5, 1, 5, 1, 5))),
    ("long dash with offset", (1, (0, 0))),
    ("dashdotted", (0, (3, 5, 1, 5))),
    ("densely dotted", (0, (1, 1))),
    ("long dash with offset", (5, (10, 3))),
    ("loosely dashdotted", (0, (3, 10, 1, 10))),
    ("densely dashdotdotted", (0, (3, 1, 1, 1, 1, 1))),
    ("loosely dashed", (0, (5, 10))),
    ("dashed", (0, (5, 5))),
    ("dashdotdotted", (0, (3, 5, 1, 5, 1, 5))),
    ("loosely dashdotdotted", (0, (3, 10, 1, 10, 1, 14))),
    ("loosely dashdotdotted", (0, (3, 10, 1, 10, 1, 23))),
    ("loosely dashdotdotted", (0, (3, 10, 1, 10, 1, 1))),
    ("loosely dashdotdotted", (0, (3, 10, 1, 1, 1, 10))),
    ("loosely dashdotdotted", (0, (3, 10, 10, 10, 1, 10))),
]


COLORMAP_NAME = "tab20"
DPI = 500
FIGSIZE = (17, 8)
FONTSIZE = 20


def get_fig_set_style(lines_count):
    # cmap = plt.colormaps.get_cmap(COLORMAP_NAME)
    # colors_list = [colors.to_hex(cmap(i)) for i in range(lines_count)]
    colors_list = ["blue", "black", "m", "red", "#0b5509", "y", "g", "y", "c", "g"]
    sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
    fig, ax = plt.subplots(dpi=DPI)
    plt.grid(which="both")
    return fig, ax, colors_list


##################################################################################
def mean_and_disp_t(arr):
    times = np.array([i for i in range(1, 1 + arr.shape[1])])
    arr = arr.cumsum(axis=1)
    arr = arr / times[None, :]
    mean = arr.mean(axis=0)
    disp = arr.std(axis=0)
    return mean, disp


def mean_and_disp(arr):
    arr = np.array(arr)
    arr = arr.cumsum(axis=1)
    mean = arr.mean(axis=0)
    disp = arr.std(axis=0)
    return mean, disp


def _plot(ax, mean, disp, color, label, linestyle):
    ax.plot(mean, color=color, label=label, linestyle=linestyle)
    ax.fill_between(range(len(mean)), mean - 0.5 * disp, mean + 0.5* disp, color=color, alpha=0.15)


def plot(
    regret_dict,
    with_legend = True
):
    figures = {}
    computed_data = {}
    for plot_type, plot_type_name, ylabel in zip(
        [mean_and_disp, mean_and_disp_t], ["regret_true", "regret_div_t"], ["Regret", "Regret/t"]
    ):
        fig, ax, colors_list = get_fig_set_style(len(regret_dict))
        plt.figure(figsize=FIGSIZE)

        for data, color, (_, linestyle) in zip(regret_dict.items(), colors_list, LINESTYLES):
            agent, regrets = data
            x, y = plot_type(np.array(regrets))
            computed_data[f"{plot_type_name}_{agent}"] = (x.tolist(), y.tolist())
            _plot(ax, x, y, color=color, label=agent, linestyle=linestyle)

            ax.set_ylabel(ylabel, fontsize=FONTSIZE)
            ax.set_xlabel("Steps, t", fontsize=FONTSIZE)
            if plot_type_name == "regret_true" and with_legend:
                ax.legend(loc="upper left")
        figures[plot_type_name] = fig
    return figures, computed_data


def plot_arr(regret_dict, ylabel):

    fig, ax, colors_list = get_fig_set_style(len(regret_dict))
    for (agent, [x, y]), color, (_, linestyle) in zip(regret_dict.items(), colors_list, LINESTYLES):
        x, y = np.array(x), np.array(y)
        _plot(ax, x, y, color=color, label=agent, linestyle=linestyle)
        ax.set_ylabel(ylabel, fontsize=FONTSIZE)
        ax.set_xlabel("Steps, t", fontsize=FONTSIZE)
        ax.legend(loc="upper right")
    return fig


def savefig(fig, path, name):
    fig.tight_layout()
    # fig.savefig(str(path / f"{name}_image.png"))
    fig.savefig(str(path / f"{name}_image.pdf"))

    # data = np.array(fig.canvas.buffer_rgba())
    # weights = [0.2989, 0.5870, 0.1140]
    # data = np.dot(data[..., :-1], weights)
    # plt.imsave(str(path / f"{name}_image_gray.png"), data, cmap="gray")
    # plt.imsave(str(path / f"{name}_image_gray.pdf"), data, cmap="gray")
