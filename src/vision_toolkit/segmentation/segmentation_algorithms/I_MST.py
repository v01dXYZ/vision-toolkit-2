# -*- coding: utf-8 -*-
import time

import networkx as nx
import numpy as np
from networkx.algorithms import tree
from scipy.spatial.distance import cdist

from vision_toolkit.utils.segmentation_utils import (
    centroids_from_ints,
    interval_merging)


def process_IMST(data_set, config):
    """
    

    Parameters
    ----------
    data_set : TYPE
        DESCRIPTION.
    config : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    assert config["distance_type"] == "euclidean", "'distance_type' must be set to 'euclidean'"

    if config.get("verbose", False):
        print("Processing MST Identification...")
        start_time = time.time()

    x_a = data_set["x_array"]
    y_a = data_set["y_array"]

    n_s = int(config["nb_samples"])
    s_f = float(config["sampling_frequency"])

    # (n_s, 2) array of gaze points
    g_p = np.column_stack((x_a.reshape(n_s), y_a.reshape(n_s)))

    vareps = float(config["IMST_distance_threshold"])

    # Window length in samples
    t_du = int(np.ceil(float(config["IMST_window_duration"]) * s_f))
    t_du = max(2, t_du)  # need at least 2 points

    # Overlap / stride (B)
    # If user provides IMST_step_samples, use it; otherwise default to 50% overlap.
    step = config.get("IMST_step_samples", None)
    if step is None:
        step = max(1, t_du // 2)
    else:
        step = max(1, int(step))

    # Minimum cluster size to accept as fixation-like (A)
    # Default: at least the minimum fixation duration in samples, capped by window length.
    min_pts = config.get("IMST_min_cluster_size", None)
    if min_pts is None:
        min_pts = int(np.ceil(float(config["min_fix_duration"]) * s_f))
    min_pts = max(2, min(min_pts, t_du))

    # Build fixation mask via voting across overlapping windows
    fix_votes = np.zeros(n_s, dtype=np.int32)
    cover_votes = np.zeros(n_s, dtype=np.int32)

    i = 0
    while i < n_s:
        j = min(i + t_du, n_s)
        # If the remaining tail is too small, stop (nothing meaningful to MST)
        if (j - i) < 2:
            break

        w_gp = g_p[i:j]

        # Mark coverage for later normalisation
        cover_votes[i:j] += 1

        # Compute pairwise distances for this window
        d_m = cdist(w_gp, w_gp, metric="euclidean")

        # MST on dense graph
        g = nx.from_numpy_array(d_m, create_using=nx.Graph())
        mst = tree.minimum_spanning_tree(g, algorithm="prim")
        edgelist = mst.edges(data=True)

        # (A) Build thresholded graph from MST edges, then take connected components
        G_thr = nx.Graph()
        G_thr.add_nodes_from(range(j - i))
        for u, v, attr in edgelist:
            if attr["weight"] < vareps:
                G_thr.add_edge(u, v)

        # Mark nodes belonging to sufficiently large components as fixation-like
        for comp in nx.connected_components(G_thr):
            if len(comp) >= min_pts:
                idx = np.fromiter(comp, dtype=int)
                fix_votes[i + idx] += 1

        i += step

    # Convert votes to fixation mask:
    # - robust default: a sample is fixation if it was classified fixation-like
    #   in at least half of the windows that covered it.
    # If a sample was never covered (unlikely), it stays False.
    i_fix = np.zeros(n_s, dtype=bool)
    covered = cover_votes > 0
    i_fix[covered] = fix_votes[covered] >= np.ceil(0.5 * cover_votes[covered])


    if config["verbose"]:
        print("Done")


    i_sac = i_fix == False
    wi_sac = np.where(i_sac == True)[0]

    s_ints = interval_merging(
        wi_sac,
        min_int_size=np.ceil(config["min_sac_duration"] * s_f),
    )

    if config["verbose"]:
        print(
            "   Saccadic intervals identified with minimum duration: {s_du} sec".format(
                s_du=config["min_sac_duration"]
            )
        )

    # i_sac events not retained as intervals are relabeled as fix events
    i_fix = np.array([True] * config["nb_samples"])

    for s_int in s_ints:
        i_fix[s_int[0] : s_int[1] + 1] = False

    # second pass to merge saccade separated by short fixations
    fix_dur_t = int(np.ceil(config["min_fix_duration"] * s_f))

    for i in range(1, len(s_ints)):
        s_int = s_ints[i]
        o_s_int = s_ints[i - 1]

        gap = s_int[0] - o_s_int[1] - 1
        if 0 <= gap < fix_dur_t:
            i_fix[o_s_int[1] + 1 : s_int[0]] = False

    if config["verbose"]:
        print(
            "   Close saccadic intervals merged with duration threshold: {f_du} sec".format(
                f_du=config["min_fix_duration"]
            )
        )

    # Recompute fixation intervals
    wi_fix = np.where(i_fix == True)[0]

    f_ints = interval_merging(
        wi_fix,
        min_int_size=np.ceil(config["min_fix_duration"] * s_f),
        max_int_size=np.ceil(config["max_fix_duration"] * s_f),
        status=data_set["status"],
        proportion=config["status_threshold"],
    )

    # Compute fixation centroids
    ctrds = centroids_from_ints(f_ints, x_a, y_a)

    # Recompute saccadic intervals
    i_sac = i_fix == False
    wi_sac = np.where(i_sac == True)[0]

    s_ints = interval_merging(
        wi_sac,
        min_int_size=np.ceil(config["min_sac_duration"] * s_f),
        status=data_set["status"],
        proportion=config["status_threshold"],
    )

    if config["verbose"]:
        print(
            "   Fixations ans saccades identified using availability status threshold: {s_th}".format(
                s_th=config["status_threshold"]
            )
        )

    assert len(f_ints) == len(
        ctrds
    ), "Interval set and centroid set have different lengths"

    if config["verbose"]:
        print("\n...MST Identification done\n")
        print("--- Execution time: %s seconds ---" % (time.time() - start_time))

    # Keep track of index that were effectively labeled
    i_lab = np.array([False] * config["nb_samples"])

    for f_int in f_ints:
        i_lab[f_int[0] : f_int[1] + 1] = True

    for s_int in s_ints:
        i_lab[s_int[0] : s_int[1] + 1] = True

    return dict(
        {
            "is_labeled": i_lab,
            "fixation_intervals": f_ints,
            "saccade_intervals": s_ints,
            "centroids": ctrds,
        }
    )
