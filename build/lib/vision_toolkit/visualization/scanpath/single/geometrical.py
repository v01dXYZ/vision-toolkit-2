# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def plot_BCEA(scanpath, p, path):
    plt.style.use("seaborn-v0_8")

    cov = np.cov(scanpath[0], scanpath[1])

    eigenvalues, eigenvectors = np.linalg.eig(cov)

    i = np.argmax(eigenvalues)
    i_ = np.argmin(eigenvalues)

    ei = eigenvalues[i]
    ei_ = eigenvalues[i_]

    ev = eigenvectors[:, i]
    angle = np.arctan2(ev[1], ev[0])

    if angle < 0:
        angle += 2 * np.pi

    x_m = np.mean(scanpath[0])
    y_m = np.mean(scanpath[1])

    chisquare_val = stats.chi2.ppf(p, df=2)

    a = np.sqrt(chisquare_val * ei)
    b = np.sqrt(chisquare_val * ei_)

    theta_grid = np.linspace(0, 2 * np.pi)

    ellipse_x_r = a * np.cos(theta_grid)
    ellipse_y_r = b * np.sin(theta_grid)

    rot_mat = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )

    r_ellipse = np.matmul(rot_mat, np.vstack((ellipse_x_r, ellipse_y_r)))

    plt.scatter(scanpath[0], scanpath[1], marker="P", s=35, color="darkblue")

    plt.plot(
        r_ellipse[0] + x_m,
        r_ellipse[1] + y_m,
        color="purple",
        linewidth=2,
        linestyle="--",
    )

    plt.title("Confidence Ellipse")

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.xlabel("Horizontal position", fontsize=12)
    plt.ylabel("Vertical position", fontsize=12)

    plt.gca().invert_yaxis()

    if path != None:
        fig = plt.gcf()
        fig.savefig(path + "_scanpath_BCEA", dpi=200, bbox_inches="tight")

    plt.show()
    plt.clf()


def plot_voronoi_cells(scanpath, vertices, path):
    plt.style.use("seaborn-v0_8")

    # scatter des fixations
    plt.scatter(scanpath[0], scanpath[1], marker="P", s=35, color="darkblue")

    # Générer une palette de couleurs
    cmap = plt.cm.get_cmap("tab20")  # ou "tab20b", "tab20c", "hsv", etc.
    n_cells = len(vertices)
    colors = cmap(np.linspace(0, 1, n_cells))

    # Dessiner les cellules avec des couleurs différentes
    for i, poly in enumerate(vertices):
        xs, ys = zip(*poly)
        plt.fill(
            xs,
            ys,
            color=colors[i],   # couleur différente pour chaque polygone
            alpha=0.4,
            edgecolor="none",  # ou "k" si tu veux les contours
        )

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.xlabel("Horizontal position", fontsize=12)
    plt.ylabel("Vertical position", fontsize=12)

    plt.gca().invert_yaxis()
    plt.gca().set_aspect("equal", adjustable="box")  # optionnel mais souvent mieux

    if path is not None:
        fig = plt.gcf()
        fig.savefig(path + "_scanpath_voronoi", dpi=200, bbox_inches="tight")

    plt.show()
    plt.clf()



def plot_HFD(sc_b, dist_, hilbert_points, coefs, x_, l_, path):
    plt.style.use("seaborn-v0_8")

    plt.plot(hilbert_points[:, 0], hilbert_points[:, 1], linewidth=0.8, color="purple")

    plt.plot(
        sc_b[0], sc_b[1], linestyle="", marker="P", markersize=8.0, color="darkblue"
    )

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.xlabel("Horizontal position", fontsize=12)
    plt.ylabel("Vertical position", fontsize=12)
    plt.gca().invert_yaxis()

    # x_left, x_right = plt.xlim()
    # y_low, y_high = plt.ylim()
    # plt.gca().set_aspect(abs((x_right-x_left)/(y_low-y_high))*1.0)
    # fig = plt.gcf()
    # fig.savefig('hfd_hilbert', dpi=200, bbox_inches='tight')

    if path != None:
        fig = plt.gcf()
        fig.savefig(path + "_scanpath_HFD_hilbert", dpi=200, bbox_inches="tight")

    plt.show()
    plt.clf()

    plt.plot(
        dist_,
        linewidth=0.5,
        linestyle="--",
        marker="P",
        markersize=8.0,
        color="purple",
        mfc="darkblue",
    )

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.xlabel("Fixation index", fontsize=12)
    plt.ylabel("Hillbert distance", fontsize=12)

    # x_left, x_right = plt.xlim()
    # y_low, y_high = plt.ylim()
    # plt.gca().set_aspect(abs((x_right-x_left)/(y_low-y_high))*1.0)
    # fig = plt.gcf()
    # fig.savefig('hfd_dist', dpi=200, bbox_inches='tight')

    if path != None:
        fig = plt.gcf()
        fig.savefig(path + "_scanpath_HFD_distances", dpi=200, bbox_inches="tight")

    plt.show()
    plt.clf()

    x = np.linspace(np.min(x_), np.max(x_), 1000)
    y = np.polyval(coefs, x)

    plt.plot(x, y, linewidth=1.8, color="black", linestyle="dashed")

    plt.plot(x_, l_, linewidth=1.2, color="darkblue")

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.xlabel("log lengths", fontsize=12)
    plt.ylabel("log inverse time intervals", fontsize=12)

    if path != None:
        fig = plt.gcf()
        fig.savefig(path + "_scanpath_HFD_regression", dpi=200, bbox_inches="tight")

    plt.show()
    plt.clf()


def plot_convex_hull(scanpath, h_a, path):
    plt.style.use("seaborn-v0_8")

    # On ferme le polygone de la convex hull
    h_a_closed = np.append(h_a, h_a[0, :].reshape(1, 2), axis=0)

    fig, ax = plt.subplots()

    # --- CONVEX HULL REMPLIE EN BLEU ---
    ax.fill(
        h_a_closed[:, 0],
        h_a_closed[:, 1],
        facecolor="royalblue",
        edgecolor="blue",
        alpha=0.25,
        linewidth=2.0,
        zorder=0,
    )

    # --- FLÈCHES ENTRE FIXATIONS (au lieu d'une ligne continue) ---
    n_fix = scanpath.shape[1]  # scanpath : (3, N) -> x, y, dur
    for i in range(n_fix - 1):
        x0, y0 = scanpath[0, i],     scanpath[1, i]
        x1, y1 = scanpath[0, i + 1], scanpath[1, i + 1]

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
            zorder=2,
        )

    # --- CERCLES POUR LES FIXATIONS (légèrement remplis) ---
    for i in range(n_fix):
        dur = scanpath[2, i]
        circle = plt.Circle(
            (scanpath[0, i], scanpath[1, i]),
            dur * 35,                # ton scaling d'origine
            edgecolor="darkblue",
            facecolor="darkblue",
            fill=True,
            alpha=0.2,               # remplissage léger
            linewidth=0.8,
        )
        ax.add_patch(circle)

    plt.xlabel("Horizontal position (px)", fontsize=12)
    plt.ylabel("Vertical position (px)", fontsize=12)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.gca().invert_yaxis()
    plt.gca().set_aspect("equal", adjustable="box")

    if path is not None:
        fig.savefig(path + "_scanpath_convex_hull", dpi=200, bbox_inches="tight")

    plt.show()
    plt.clf()
