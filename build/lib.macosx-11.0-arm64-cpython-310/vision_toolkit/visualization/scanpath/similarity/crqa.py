# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_CRQA(r_m, path):
    plt.style.use("seaborn-v0_8")

    # Figure : fond extérieur blanc
    fig, ax = plt.subplots()
    fig.patch.set_facecolor("white")

    # Fond violet très léger de seaborn
    bg_color = plt.rcParams["axes.facecolor"]
    fg_color = "#0b3c8c"  # bleu foncé pour les récurrences

    # Si r_m n'est pas strictement binaire, on peut le seuiller
    r_bin = (r_m > 0).astype(int)

    # Colormap binaire
    cmap = ListedColormap([bg_color, fg_color])

    ax.pcolormesh(
        r_bin,
        cmap=cmap,
        edgecolors="w",
        linewidth=0.5,
    )

    ax.set_facecolor(bg_color)
    ax.set_aspect("equal")

    ax.set_xlabel("Second sequence fixation index", fontsize=12)
    ax.set_ylabel("First sequence fixation index", fontsize=12)

    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)

    if path is not None:
        fig.savefig(path + "_scanpath_CRQA", dpi=200, bbox_inches="tight")

    plt.show()
    plt.clf()


def plot_CRQA_laminarity(r_m, set_, path):
    plt.style.use("seaborn-v0_8")

    # Matrice de laminarité (0/1)
    l_mat = np.zeros_like(r_m, dtype=int)
    for l_ in set_:
        for i in range(len(l_)):
            l_mat[l_[i, 0], l_[i, 1]] = 1

    l_bin = (l_mat > 0).astype(int)

    fig, ax = plt.subplots()
    fig.patch.set_facecolor("white")

    bg_color = plt.rcParams["axes.facecolor"]
    fg_color = "#0b3c8c"

    cmap = ListedColormap([bg_color, fg_color])

    ax.pcolormesh(
        l_bin,
        cmap=cmap,
        edgecolors="w",
        linewidth=0.5,
    )

    ax.set_facecolor(bg_color)
    ax.set_aspect("equal")

    ax.set_xlabel("Second sequence fixation index", fontsize=12)
    ax.set_ylabel("First sequence fixation index", fontsize=12)

    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)

    if path is not None:
        fig.savefig(path + "_scanpath_CRQA_laminarity", dpi=200, bbox_inches="tight")

    plt.show()
    plt.clf()


def plot_CRQA_determinism(r_m, d_set, path):
    plt.style.use("seaborn-v0_8")

    # Matrice de déterminisme (0/1)
    d_mat = np.zeros_like(r_m, dtype=int)
    for l_ in d_set:
        for i in range(len(l_)):
            d_mat[l_[i, 0], l_[i, 1]] = 1

    d_bin = (d_mat > 0).astype(int)

    fig, ax = plt.subplots()
    fig.patch.set_facecolor("white")

    bg_color = plt.rcParams["axes.facecolor"]
    fg_color = "#0b3c8c"

    cmap = ListedColormap([bg_color, fg_color])

    ax.pcolormesh(
        d_bin,
        cmap=cmap,
        edgecolors="w",
        linewidth=0.5,
    )

    ax.set_facecolor(bg_color)
    ax.set_aspect("equal")

    ax.set_xlabel("Second sequence fixation index", fontsize=12)
    ax.set_ylabel("First sequence fixation index", fontsize=12)

    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)

    if path is not None:
        fig.savefig(path + "_scanpath_CRQA_determinism", dpi=200, bbox_inches="tight")

    plt.show()
    plt.clf()

 