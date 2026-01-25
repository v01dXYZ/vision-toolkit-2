# -*- coding: utf-8 -*-

import time

import numpy as np

from vision_toolkit.utils.segmentation_utils import dispersion_metric, interval_merging


def process_IVDT(data_set, config):
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
        print("Processing VDT Identification...")
        start_time = time.time()

    a_sp = data_set["absolute_speed"]
    s_f = config["sampling_frequency"]

    if config["distance_type"] == "euclidean":
        x_a = data_set["x_array"]
        y_a = data_set["y_array"]
    elif config["distance_type"] == "angular":
        theta_coord = data_set["theta_coord"]
        x_a = theta_coord[0, :]
        y_a = theta_coord[1, :]

    t_s = config["IVDT_saccade_threshold"]
    t_du = int(np.ceil(config["IVDT_window_duration"] * s_f))
    t_du = max(2, t_du)
    t_di = config["IVDT_dispersion_threshold"]

    # base: saccades by velocity threshold
    i_sac = (a_sp > t_s).astype(int)
    i_purs = np.zeros_like(i_sac)
    i_fix = np.zeros_like(i_sac)

    wi_intersac = np.where(a_sp <= t_s)[0]
    _ints = interval_merging(wi_intersac)

    for _int in _ints:
        a, b = _int[0], _int[1]  # inclusive bounds
        if (b - a + 1) <= t_du:
            # too short to apply dispersion window robustly: mark as pursuit by default (as in your logic)
            i_purs[a:b+1] = 1
            i_sac[a:b+1] = 0
            continue

        i = a
        while (i + t_du) <= (b + 1):
            j = i + t_du  # exclusive

            d = dispersion_metric(x_a[i:j], y_a[i:j])

            if d < t_di:
                while d < t_di and j < (b + 1):
                    j += 1
                    d = dispersion_metric(x_a[i:j], y_a[i:j])

                i_fix[i:j-1] = 1
                i_sac[i:j-1] = 0
                i = j
            else:
                i_purs[i] = 1
                i_sac[i] = 0
                i += 1

        # remaining tail
        i_purs[i:b+1] = 1
        i_sac[i:b+1] = 0

    if config["verbose"]:
        print("\n...VDT Identification done\n")
        print("--- Execution time: %s seconds ---" % (time.time() - start_time))

    return {
        "is_saccade": i_sac == 1,
        "saccade_intervals": interval_merging(np.where(i_sac == 1)[0]),
        "is_pursuit": i_purs == 1,
        "pursuit_intervals": interval_merging(np.where(i_purs == 1)[0]),
        "is_fixation": i_fix == 1,
        "fixation_intervals": interval_merging(np.where(i_fix == 1)[0]),
    }
