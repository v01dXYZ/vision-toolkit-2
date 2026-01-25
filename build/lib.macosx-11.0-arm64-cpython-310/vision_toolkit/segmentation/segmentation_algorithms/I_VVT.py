# -*- coding: utf-8 -*-

import time

import numpy as np

from vision_toolkit.utils.segmentation_utils import interval_merging


def process_IVVT(data_set, config):
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
        print("Processing VVT Identification...")
        start_time = time.time()

    a_sp = data_set["absolute_speed"]
    T_s = config["IVVT_saccade_threshold"]
    T_p = config["IVVT_pursuit_threshold"]

    # handle invalid speeds 
    valid = np.isfinite(a_sp)

    is_saccade  = (~valid) | (a_sp > T_s)
    is_pursuit  = valid & (a_sp > T_p) & (a_sp <= T_s)
    is_fixation = valid & (a_sp <= T_p)

    if config["verbose"]:
        print("\n...VVT Identification done\n")
        print("--- Execution time: %s seconds ---" % (time.time() - start_time))

    return {
        "is_saccade": is_saccade,
        "saccade_intervals": interval_merging(np.where(is_saccade)[0]),
        "is_pursuit": is_pursuit,
        "pursuit_intervals": interval_merging(np.where(is_pursuit)[0]),
        "is_fixation": is_fixation,
        "fixation_intervals": interval_merging(np.where(is_fixation)[0]),
    }

