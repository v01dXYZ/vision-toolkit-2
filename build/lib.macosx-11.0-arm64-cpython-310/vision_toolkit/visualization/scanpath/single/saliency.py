# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import seaborn as sns

def plot_saliency_map(s_m, path):
    ax = sns.heatmap(
        s_m,
        square=True,
        vmin=0,
        vmax=0.0015,        # optionnel
    )

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.yticks(rotation=0)

    plt.xlabel("Horizontal position", fontsize=12)
    plt.ylabel("Vertical position", fontsize=12)

    if path != None:
        fig = plt.gcf()
        fig.savefig(path + "_scanpath_saliency_map", dpi=200, bbox_inches="tight")
        
    plt.show()
    plt.clf()