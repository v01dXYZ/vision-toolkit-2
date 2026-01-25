# -*- coding: utf-8 -*-

import copy
import time

import numpy as np
from scipy.stats import norm

from vision_toolkit.utils.segmentation_utils import interval_merging


def process_IBDT(data_set, config):
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
        print("Processing BDT Identification...")
        start_time = time.time()

    n_s = int(config["nb_samples"])
    s_f = float(config["sampling_frequency"])

    d_t = max(1, int(np.ceil(float(config["IBDT_duration_threshold"]) * s_f)))

    fix_t = float(config["IBDT_fixation_threshold"])
    sac_t = float(config["IBDT_saccade_threshold"])
    pur_t = float(config["IBDT_pursuit_threshold"])

    fix_sd = max(1e-9, float(config["IBDT_fixation_sd"]))
    sac_sd = max(1e-9, float(config["IBDT_saccade_sd"]))

    a_s = data_set["absolute_speed"]

    priors = {"fix": np.zeros(n_s), "sac": np.zeros(n_s), "pur": np.zeros(n_s)}
    likelihoods = {"fix": None, "sac": None, "pur": np.zeros(n_s)}
    posteriors = {"fix": None, "sac": None, "pur": None}

    # i < d_t
    for i in range(min(d_t, n_s)):
        likelihoods["pur"][i] = np.sum(a_s[: i + 1] > pur_t) / (i + 1)
        priors["pur"][i] = np.mean(likelihoods["pur"][: i + 1])
        priors["fix"][i] = priors["sac"][i] = (1.0 - priors["pur"][i]) / 2.0

    # i >= d_t
    for i in range(d_t, n_s):
        likelihoods["pur"][i] = np.sum(a_s[i - d_t + 1 : i + 1] > pur_t) / d_t
        priors["pur"][i] = np.mean(likelihoods["pur"][i - d_t + 1 : i + 1])
        priors["fix"][i] = priors["sac"][i] = (1.0 - priors["pur"][i]) / 2.0

    # Likelihoods fix/sac
    lk_f = a_s.copy()
    lk_f[lk_f < fix_t] = fix_t
    likelihoods["fix"] = norm.pdf(lk_f, loc=fix_t, scale=fix_sd)

    lk_s = a_s.copy()
    lk_s[lk_s > sac_t] = sac_t
    likelihoods["sac"] = norm.pdf(lk_s, loc=sac_t, scale=sac_sd)

    # Pur "pseudo-likelihood" stabilisée
    eps = 1e-6
    likelihoods["pur"] = np.clip(likelihoods["pur"], eps, 1 - eps)

    for ev in ("fix", "sac", "pur"):
        posteriors[ev] = priors[ev] * likelihoods[ev]

    a_m = np.argmax(
        np.vstack((posteriors["fix"], posteriors["sac"], posteriors["pur"])),
        axis=0,
    )

    if config["verbose"]:
        print("\n...BDT Identification done\n")
        print("--- Execution time: %s seconds ---" % (time.time() - start_time))

    return {
        "is_saccade": a_m == 1,
        "saccade_intervals": interval_merging(np.where(a_m == 1)[0], min_int_size=config["min_int_size"]),
        "is_pursuit": a_m == 2,
        "pursuit_intervals": interval_merging(np.where(a_m == 2)[0], min_int_size=config["min_int_size"]),
        "is_fixation": a_m == 0,
        "fixation_intervals": interval_merging(np.where(a_m == 0)[0], min_int_size=config["min_int_size"]),
    }
