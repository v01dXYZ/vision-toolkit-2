# -*- coding: utf-8 -*-

import time

import matplotlib.pyplot as plt
import numpy as np

from vision_toolkit.utils.segmentation_utils import interval_merging


def process_IVMP(data_set, config):
    """
    

    Parameters
    ----------
    data_set : TYPE
        DESCRIPTION.
    config : TYPE
        DESCRIPTION.

    Returns
    -------
    dict
        DESCRIPTION.

    """

    if config["verbose"]:
        print("Processing VMP Identification...")
        start_time = time.time()

    a_sp = data_set["absolute_speed"]
    n_s = config["nb_samples"]
    s_f = config["sampling_frequency"]

    x_array = data_set["x_array"]
    y_array = data_set["y_array"]

    t_s = config["IVMP_saccade_threshold"]
    t_du = int(np.ceil(config["IVMP_window_duration"] * s_f))
    t_du = max(2, t_du)
    t_r = config["IVMP_rayleigh_threshold"]

    # Base labeling from velocity threshold
    is_sac = a_sp > t_s
    is_fix = ~is_sac          # intersaccades = fixation by default
    is_purs = np.zeros(n_s, dtype=bool)

    # Intersaccadic intervals
    wi_intersac = np.where(~is_sac)[0]
    inter_ints = interval_merging(wi_intersac)

    # Compute successive direction angles
    dx = np.empty(n_s)
    dy = np.empty(n_s)
    dx[:-1] = x_array[1:] - x_array[:-1]
    dy[:-1] = y_array[1:] - y_array[:-1]
    dx[-1] = dx[-2]
    dy[-1] = dy[-2]

    # angle in [0, 2pi)
    suc_dir = np.mod(np.arctan2(dy, dx), 2 * np.pi)

    for a, b in inter_ints:
        # work in blocks inside [a, b] inclusive
        i = a
        while i <= b:
            j = min(i + t_du, b + 1)  # j is exclusive
            if (j - i) < 2:
                break

            pos_unitary_circle = np.array([np.cos(suc_dir[i:j]), np.sin(suc_dir[i:j])])
            rm_vec = np.sum(pos_unitary_circle, axis=1) / (j - i)
            z_score = np.linalg.norm(rm_vec) ** 2

            if z_score > t_r:
                is_purs[i:j] = True
                is_fix[i:j] = False

            i = j

    # enforce exclusivity: pursuit overrides fixation only inside intersaccades
    is_purs = is_purs & (~is_sac)
    is_fix = (~is_sac) & (~is_purs)

    if config["verbose"]:
        print("\n...VMP Identification done\n")
        print("--- Execution time: %s seconds ---" % (time.time() - start_time))

    return {
        "is_saccade": is_sac,
        "saccade_intervals": interval_merging(np.where(is_sac)[0]),
        "is_pursuit": is_purs,
        "pursuit_intervals": interval_merging(np.where(is_purs)[0]),
        "is_fixation": is_fix,
        "fixation_intervals": interval_merging(np.where(is_fix)[0]),
    }
