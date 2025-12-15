# -*- coding: utf-8 -*-
 


import numpy as np
import matplotlib.pyplot as plt
from string import ascii_uppercase as ABC


def plot_scanpaths_with_grid_arrows( 
    xlim=(0, 1200), ylim=(0, 800),
    x_splits=None, y_splits=None,
    save_path='figures/scanpath_grid_2.png',
    arrow_lw=1.6, arrow_scale=18, arrowstyle="-|>"
):
    if x_splits is None:
        x_splits = np.linspace(xlim[0], xlim[1], 5)  # 4 colonnes
    if y_splits is None:
        y_splits = np.linspace(ylim[0], ylim[1], 5)  # 4 lignes

    blue_xy = np.array([
        [888,701],[610,501],[500,550],[527,701],
        [1020,500],[447,267],[200,51],[217,317],[401,101]
    ], dtype=float)

    purple_xy = np.array([
        [995,760],[405,505],[560,640],[820,660],
        [1060,670],[915,425],[680,415],[342,220],[90,220],
        [265,105],[405,20]
    ], dtype=float)

    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots()

 
    for xv in x_splits[1:-1]:
        ax.axvline(xv, color="black", lw=3)
    for yv in y_splits[1:-1]:
        ax.axhline(yv, color="black", lw=3)

    x0, x1 = x_splits[0], x_splits[-1]
    y0, y1 = y_splits[0], y_splits[-1]
    ax.add_patch(plt.Rectangle(
        (x0, y0), x1 - x0, y1 - y0,
        fill=False, edgecolor="black", lw=4, zorder=10
    ))

 
    letters = list(ABC[:16])  # A..P
    k = 0
    for r in range(len(y_splits) - 1):
        for c in range(len(x_splits) - 1):
            cx = 0.5 * (x_splits[c] + x_splits[c+1])
            cy = 0.5 * (y_splits[r] + y_splits[r+1])
            ax.text(
                cx, cy, letters[k],
                color="dimgray",
                ha="center", va="center",
                fontsize=36, alpha=0.8
            )
            k += 1
 
    def draw_arrows(points, color):
        for i in range(len(points) - 1):
            (x0, y0), (x1, y1) = points[i], points[i+1]
            ax.annotate(
                "",
                xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle=arrowstyle,
                    facecolor=color,
                    edgecolor=color,
                    lw=arrow_lw,
                    mutation_scale=arrow_scale,
                    shrinkA=0,
                    shrinkB=0,
                ),
                zorder=3,
            )
 
 
    draw_arrows(blue_xy,   color="#0b3c8c")  # bleu
    draw_arrows(purple_xy, color="#6a1b9a")  # violet
 
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Horizontal position (px)", fontsize=14)
    ax.set_ylabel("Vertical position (px)", fontsize=14)
    ax.tick_params(labelsize=11)

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()
    plt.clf()
    
    
def plot_scanpath_with_grid_arrows_circle(
    xlim=(0, 1200), ylim=(0, 800),
    x_splits=None, y_splits=None,
    save_path='figures/scanpath_grid.png',
    arrow_lw=1.6, arrow_scale=18, arrowstyle="-|>",
    circle_radii=[50, 70, 28, 22, 25, 52, 27, 34, 31, 48, 31],           # <-- rayon(s) des cercles
    circle_edgecolor="darkblue",               # contour des cercles
    circle_facecolor="darkblue",               # remplissage des cercles
    circle_alpha=0.20,                        # transparence du remplissage
):
    
    # Données (x, y)
    blue_xy = np.array([
       [995,760],[405,505],[560,640],[820,660],
       [1060,670],[915,425],[680,415],[342,220],[90,220], [265, 105], [405, 20]
    ], dtype=float)

    # Grille 4x4 par défaut
    if x_splits is None: x_splits = np.linspace(xlim[0], xlim[1], 5)
    if y_splits is None: y_splits = np.linspace(ylim[0], ylim[1], 5)

    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots()

    # --- Grille noire ---
    for xv in x_splits[1:-1]: ax.axvline(xv, color="black", lw=3)
    for yv in y_splits[1:-1]: ax.axhline(yv, color="black", lw=3)
    x0, x1 = x_splits[0], x_splits[-1]
    y0, y1 = y_splits[0], y_splits[-1]
    ax.add_patch(plt.Rectangle((x0, y0), x1-x0, y1-y0,
                               fill=False, edgecolor="black", lw=4))

    # --- Lettres A–P ---
    letters = list(ABC[:16])
    k = 0
    for r in range(len(y_splits)-1):
        for c in range(len(x_splits)-1):
            cx = 0.5*(x_splits[c] + x_splits[c+1])
            cy = 0.5*(y_splits[r] + y_splits[r+1])
            ax.text(cx, cy, letters[k], color="dimgray",
                    ha="center", va="center", fontsize=36, alpha=0.8)
            k += 1

    # --- Flèches pleines entre points (aucun dot) ---
    def draw_arrows(points, color):
        for i in range(len(points) - 1):
            (x0, y0), (x1, y1) = points[i], points[i+1]
            ax.annotate(
                "", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle=arrowstyle,
                    facecolor=color, edgecolor=color,
                    lw=arrow_lw, mutation_scale=arrow_scale,
                    shrinkA=0, shrinkB=0
                ),
                zorder=3,
            )

    draw_arrows(blue_xy, color="#6a1b9a")  # violet

    # --- Cercles légèrement remplis sur les points ---
    # rayon : scalaire ou liste/array par point
    radii = np.asarray(circle_radii, dtype=float)
    if radii.size != len(blue_xy):
        raise ValueError("circle_radii doit avoir la même longueur que blue_xy.")

    for (x, y), r in zip(blue_xy, radii):
        circ = plt.Circle(
            (x, y), r,
            edgecolor=circle_edgecolor,
            facecolor=circle_facecolor,
            fill=True, alpha=circle_alpha,
            linewidth=1.2, zorder=4
        )
        ax.add_patch(circ)

    # --- Axes & mise en forme ---
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.invert_yaxis()                     # 0 en haut
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Horizontal position (px)", fontsize=14)
    ax.set_ylabel("Vertical position (px)", fontsize=14)
    ax.tick_params(labelsize=11)

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show(); plt.clf()
    
 
    
def dtw_figure_with_arrows():
    # --- Données ---
    blue_xy = np.array([
        [640, 420],
        [460, 610],
        [800, 630],
        [1040, 590],
        [985, 290],
        [798, 355],
        [580, 404],
        [597, 320],
        [605, 255],
        [655, 260],
        [703, 250],
        [1050, 85],
    ], dtype=float)

    purple_xy = np.array([
        [505, 403],
        [410, 560],
        [480, 535],
        [585, 515],
        [600, 580],
        [858, 430],
        [1020, 405],
        [1100, 290],
        [940, 190],
        [790, 295],
        [685, 395],
        [640, 315],
        [1055, 201],
    ], dtype=float)

    # --- DTW (coût cumulé + backtracking) ---
    def dtw(P, Q):
        n, m = len(P), len(Q)
        C = np.full((n, m), np.inf, float)
        parent = np.full((n, m, 2), -1, int)

        def d(i, j):
            return np.linalg.norm(P[i] - Q[j])

        C[0, 0] = d(0, 0)
        parent[0, 0] = [-1, -1]

        for i in range(1, n):
            C[i, 0] = C[i-1, 0] + d(i, 0)
            parent[i, 0] = [i-1, 0]
        for j in range(1, m):
            C[0, j] = C[0, j-1] + d(0, j)
            parent[0, j] = [0, j-1]

        for i in range(1, n):
            for j in range(1, m):
                cost = d(i, j)
                # insertion, suppression, match
                opts = [
                    (C[i-1, j],   (i-1, j)),
                    (C[i,   j-1], (i,   j-1)),
                    (C[i-1, j-1], (i-1, j-1)),
                ]
                prev_cost, prev_idx = min(opts, key=lambda t: t[0])
                C[i, j] = cost + prev_cost
                parent[i, j] = prev_idx

        # Chemin optimal
        path = []
        i, j = n-1, m-1
        while i >= 0 and j >= 0:
            path.append((i, j))
            if parent[i, j][0] == -1:
                break
            i, j = parent[i, j]
        path = path[::-1]

        # Plus grande distance locale le long du chemin (pour le segment rouge)
        local_dists = [np.linalg.norm(P[i] - Q[j]) for (i, j) in path]
        imax = int(np.argmax(local_dists))
        return C[n-1, m-1], path, imax

    dtw_cost, coupling, imax = dtw(blue_xy, purple_xy)

    # --- Figure ---
    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots()

    # Flèches entre points (comme display_scanpath)
    def draw_arrows(points, color):
        for k in range(len(points) - 1):
            (x0, y0), (x1, y1) = points[k], points[k+1]
            ax.annotate(
                "",
                xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="-|>",
                    facecolor=color,
                    edgecolor=color,
                    lw=1.2,
                    mutation_scale=16,
                    shrinkA=0, shrinkB=0,
                ),
                zorder=3,
            )

    draw_arrows(blue_xy,   color="#0b3c8c")  # BLEU
    draw_arrows(purple_xy, color="#6a1b9a")  # VIOLET

    # Couplage DTW en pointillés
    for (i, j) in coupling:
        ax.plot([blue_xy[i, 0], purple_xy[j, 0]],
                [blue_xy[i, 1], purple_xy[j, 1]],
                "k--", lw=1, zorder=2)


    # Les annotations n'autoscalent pas -> fixer limites depuis les points
    all_pts = np.vstack([blue_xy, purple_xy])
    pad = 40
    ax.set_xlim(all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad)
    ax.set_ylim(all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)

    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()  # décommente si tu veux l'origine en haut
    ax.set_xlabel("Horizontal position (px)")
    ax.set_ylabel("Vertical position (px)") 
    
    save_path = 'figures/DTW.png'
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    
    plt.show()



def frechet_figure_with_arrows():
    # --- Données ---
    blue_xy = np.array([
        [640, 420],
        [460, 610],
        [800, 630],
        [1040, 590],
        [985, 290],
        [798, 355],
        [580, 404],
        [597, 320],
        [605, 255],
        [655, 260],
        [703, 250],
        [1050, 85],
    ], dtype=float)

    purple_xy = np.array([
        [505, 403],
        [410, 560],
        [480, 535],
        [585, 515],
        [600, 580],
        [858, 430],
        [1020, 405],
        [1100, 290],
        [940, 190],
        [790, 295],
        [685, 395],
        [640, 315],
        [1055, 201],
    ], dtype=float)

    # --- Fréchet discret (Eiter & Mannila) + backtracking ---
    def discrete_frechet(P, Q):
        n, m = len(P), len(Q)
        ca = np.full((n, m), np.inf, float)
        parent = np.full((n, m, 2), -1, int)

        def d(i, j):
            return np.linalg.norm(P[i] - Q[j])

        for i in range(n):
            for j in range(m):
                dist = d(i, j)
                if i == 0 and j == 0:
                    ca[i, j] = dist; parent[i, j] = [-1, -1]
                elif i > 0 and j == 0:
                    ca[i, 0] = max(ca[i-1, 0], dist); parent[i, 0] = [i-1, 0]
                elif i == 0 and j > 0:
                    ca[0, j] = max(ca[0, j-1], dist); parent[0, j] = [0, j-1]
                else:
                    # d[i,j] = max( dist, min(ca[i-1,j], ca[i-1,j-1], ca[i,j-1]) )
                    opts = [
                        (max(ca[i-1, j-1], dist), (i-1, j-1)),
                        (max(ca[i-1, j],   dist), (i-1, j)),  
                        (max(ca[i,   j-1], dist), (i,   j-1)),
                    ]
                    ca[i, j], parent[i, j] = min(opts, key=lambda t: t[0])

        # backtrack du couplage optimal
        path = []
        i, j = n-1, m-1
        while i >= 0 and j >= 0:
            path.append((i, j))
            if parent[i, j][0] == -1:
                break
            i, j = parent[i, j]
        path = path[::-1]

        # paire qui réalise la distance (max le long du chemin)
        dists = [np.linalg.norm(P[i] - Q[j]) for (i, j) in path]
        imax = int(np.argmax(dists))
        return ca[n-1, m-1], path, imax

    frechet_val, coupling, imax = discrete_frechet(blue_xy, purple_xy)

    # --- Figure ---
    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots()

    # Flèches pleines entre points (comme display_scanpath)
    def draw_arrows(points, color):
        for k in range(len(points) - 1):
            (x0, y0), (x1, y1) = points[k], points[k+1]
            ax.annotate(
                "",
                xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="-|>",      # flèche pleine
                    facecolor=color,
                    edgecolor=color,
                    lw=1.2,
                    mutation_scale=16,
                    shrinkA=0, shrinkB=0,
                ),
                zorder=3,
            )

    draw_arrows(blue_xy,   color="#0b3c8c")  # BLEU
    draw_arrows(purple_xy, color="#6a1b9a")  # VIOLET

    # Couplage Fréchet optimal en pointillés
    for (i, j) in coupling:
        ax.plot([blue_xy[i, 0], purple_xy[j, 0]],
                [blue_xy[i, 1], purple_xy[j, 1]],
                "k--", lw=1, zorder=2)

    # Segment rouge = paire responsable de la distance de Fréchet
    i, j = coupling[imax]
    ax.plot([blue_xy[i, 0], purple_xy[j, 0]],
            [blue_xy[i, 1], purple_xy[j, 1]],
            color="red", lw=3, zorder=4)

    # Les annotations n'autoscalent pas -> fixer limites depuis les points
    all_pts = np.vstack([blue_xy, purple_xy])
    pad = 40
    ax.set_xlim(all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad)
    ax.set_ylim(all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)

    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()  # décommente si ton repère a l'origine en haut
    ax.set_xlabel("Horizontal position (px)")
    ax.set_ylabel("Vertical position (px)") 
    
    save_path = 'figures/frechet.png'
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    
    plt.show()

# Lance
dtw_figure_with_arrows()
