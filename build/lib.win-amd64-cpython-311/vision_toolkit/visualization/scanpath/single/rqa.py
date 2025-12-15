# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


def plot_RQA(r_m, path):
    plt.style.use("seaborn-v0_8")

    # Figure avec fond extérieur blanc
    fig, ax = plt.subplots()
    fig.patch.set_facecolor("white")  # autour des axes = blanc

    # Couleur "violet très léger" de seaborn (fond des axes)
    bg_color = plt.rcParams["axes.facecolor"]  # ex. #eaeaf2 avec seaborn-v0_8

    # Colormap binaire : 0 -> fond violet clair, 1 -> bleu foncé (récurrences)
    fg_color = "#0b3c8c"  # bleu foncé
    cmap = ListedColormap([bg_color, fg_color])

    # Si r_m est binaire (0/1). Sinon, on peut le seuiller avant :
    # r_bin = (r_m > 0).astype(int)
    r_bin = r_m

    mesh = ax.pcolormesh(
        r_bin,
        cmap=cmap,
        edgecolors="w",
        linewidth=0.5,
    )

    # Fond de l'axe = même violet très léger
    ax.set_facecolor(bg_color)

    ax.set_aspect("equal")
    ax.set_xlabel("Fixation index", fontsize=14)
    ax.set_ylabel("Fixation index", fontsize=14)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)

    if path is not None:
        fig.savefig(path + "_scanpath_RQA", dpi=200, bbox_inches="tight")

    plt.show()
    plt.clf()


def plot_RQA_laminarity(r_m, h_set, v_set, path):
    plt.style.use("seaborn-v0_8")

    # Construction de la matrice de laminarité (0/1)
    l_mat = np.zeros_like(r_m, dtype=int)
    for l_ in h_set:
        for i in range(len(l_)):
            l_mat[l_[i, 0], l_[i, 1]] = 1

    for l_ in v_set:
        for i in range(len(l_)):
            l_mat[l_[i, 0], l_[i, 1]] = 1

    # Symétriser et binariser (tout > 0 devient 1)
    l_mat = l_mat + l_mat.T
    l_bin = (l_mat > 0).astype(int)

    # Figure : fond extérieur blanc
    fig, ax = plt.subplots()
    fig.patch.set_facecolor("white")

    # Récupérer le violet très léger seaborn
    bg_color = plt.rcParams["axes.facecolor"]
    fg_color = "#0b3c8c"  # bleu foncé pour les segments laminaires

    # Colormap binaire : 0 -> violet clair, 1 -> bleu foncé
    cmap = ListedColormap([bg_color, fg_color])

    # Affichage de la matrice de laminarité
    ax.pcolormesh(
        l_bin,
        cmap=cmap,
        edgecolors="w",
        linewidth=0.5,
    )

    # Fond de l'axe = même violet clair
    ax.set_facecolor(bg_color)

    ax.set_aspect("equal")
    ax.set_xlabel("Fixation index", fontsize=14)
    ax.set_ylabel("Fixation index", fontsize=14)

    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)

    if path is not None:
        fig.savefig(path + "_scanpath_RQA_laminarity", dpi=200, bbox_inches="tight")

    plt.show()
    plt.clf()


def plot_RQA_determinism(r_m, d_set, path):
    plt.style.use("seaborn-v0_8")

    # Construction de la matrice de déterminisme (0/1)
    d_mat = np.zeros_like(r_m, dtype=int)
    for l_ in d_set:
        for i in range(len(l_)):
            d_mat[l_[i, 0], l_[i, 1]] = 1

    # Symétriser et binariser
    d_mat = d_mat + d_mat.T
    d_bin = (d_mat > 0).astype(int)

    # Figure : fond extérieur blanc
    fig, ax = plt.subplots()
    fig.patch.set_facecolor("white")

    # Couleur de fond "violet clair" fournie par seaborn
    bg_color = plt.rcParams["axes.facecolor"]
    fg_color = "#0b3c8c"  # bleu foncé pour les segments déterministes

    # Colormap binaire : 0 -> violet clair, 1 -> bleu foncé
    cmap = ListedColormap([bg_color, fg_color])

    # Affichage matrice de déterminisme
    ax.pcolormesh(
        d_bin,
        cmap=cmap,
        edgecolors="w",
        linewidth=0.5,
    )

    # Fond de l'axe = violet clair
    ax.set_facecolor(bg_color)

    ax.set_aspect("equal")
    ax.set_xlabel("Fixation index", fontsize=14)
    ax.set_ylabel("Fixation index", fontsize=14)

    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)

    if path is not None:
        fig.savefig(path + "_scanpath_RQA_determinism", dpi=200, bbox_inches="tight")

    plt.show()
    plt.clf()
