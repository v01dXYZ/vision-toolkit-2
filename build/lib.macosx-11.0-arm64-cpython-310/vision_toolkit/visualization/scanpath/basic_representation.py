# -*- coding: utf-8 -*-

import cv2
import matplotlib.pyplot as plt
import numpy as np


def display_scanpath_reference_image(values, config, ref_image):
 
    path = config["display_scanpath_path"]

    # --- Charger l'image si chemin ---
    if isinstance(ref_image, str):
        ref_image = cv2.imread(ref_image, cv2.IMREAD_COLOR)
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)

    # --- Redimensionner à la taille du plan ---
    ref_image = cv2.resize(ref_image, (config["size_plan_x"], config["size_plan_y"]))

    vf_diag = np.linalg.norm(np.array([config["size_plan_x"], config["size_plan_y"]]))
    plt.style.use("seaborn-v0_8")

    s_p = values.T  # s_p[:,0]=x, s_p[:,1]=y, s_p[:,2]=dur

    fig, ax = plt.subplots()

    # --- Fond image ---
    ax.imshow(ref_image, alpha=0.4)
    ax.grid(None)

    # --- FLÈCHES ENTRE FIXATIONS ---
    for i in range(len(s_p) - 1):
        x0, y0 = s_p[i, 0], s_p[i, 1]
        x1, y1 = s_p[i + 1, 0], s_p[i + 1, 1]

        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="-|>",
                facecolor="purple",
                edgecolor="purple",
                lw=0.8,
                mutation_scale=15,
                shrinkA=0,
                shrinkB=0,
            ),
        )

    # --- CERCLES POUR LES FIXATIONS ---
    for i in range(len(s_p)):
        dur = s_p[i, 2]
        circle = plt.Circle(
            (s_p[i, 0], s_p[i, 1]),
            0.05 * dur * vf_diag,
            linewidth=0.8,
            edgecolor="darkblue",
            facecolor="darkblue",
            fill=True,
            alpha=0.2,
        )
        ax.add_patch(circle)

    plt.xlabel("Horizontal position (px)", fontsize=12)
    plt.ylabel("Vertical position (px)", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # mêmes limites/inversion que display_scanpath
    plt.xlim([0, config["size_plan_x"]])
    plt.ylim([0, config["size_plan_y"]])
    plt.gca().invert_yaxis()

    if path is not None:
        fig.savefig(path + "_scanpath_reference_image", dpi=200, bbox_inches="tight")

    plt.show()
    plt.clf()


 
def display_scanpath(values, config):
 
    path = config["display_scanpath_path"]
    vf_diag = np.linalg.norm(np.array([config["size_plan_x"], config["size_plan_y"]]))
    plt.style.use("seaborn-v0_8")
    s_p = values.T   # s_p[:,0] = x, s_p[:,1] = y, s_p[:,2] = dur

    fig, ax = plt.subplots()

    # --- FLÈCHES ENTRE FIXATIONS ---
    for i in range(len(s_p) - 1):
        x0, y0 = s_p[i, 0],  s_p[i, 1]
        x1, y1 = s_p[i+1, 0], s_p[i+1, 1]

        ax.annotate(
            "",
            xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="-|>",      # flèche pleine bien marquée
                facecolor="purple",
                edgecolor="purple",
                lw=0.8,
                mutation_scale=15,
                shrinkA=0, shrinkB=0,
            ),
        )

    # --- CERCLES POUR LES FIXATIONS ---
    for i in range(len(s_p)):
        dur = s_p[i, 2]
        circle = plt.Circle(
            (s_p[i, 0], s_p[i, 1]),
            0.05 * dur * vf_diag,
            linewidth=0.8,
            edgecolor="darkblue",
            facecolor="darkblue",
            fill=True,
            alpha=0.2,     
        )
        ax.add_patch(circle)

    plt.xlabel("Horizontal position (px)", fontsize=12)
    plt.ylabel("Vertical position (px)", fontsize=12)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.xlim([0, config["size_plan_x"]])
    plt.ylim([0, config["size_plan_y"]])
    plt.gca().invert_yaxis()

    if path is not None:
        fig.savefig(path + "_scanpath", dpi=200, bbox_inches="tight")

    plt.show()
    plt.clf()

