# -*- coding: utf-8 -*-

import itertools
from itertools import groupby
from operator import itemgetter

import numpy as np


def dispersion_metric(x_array, y_array):
    """

    Parameters
    ----------
    x_array : TYPE
        DESCRIPTION.
    y_array : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return max(x_array) - min(x_array) + max(y_array) - min(y_array)


def centroids_from_ints(f_ints, x_coords, y_coords):
    """

    Parameters
    ----------
    w_i : TYPE
        DESCRIPTION.
    x_coords : TYPE
        DESCRIPTION.
    y_coords : TYPE
        DESCRIPTION.
    p_o : TYPE, optional
        DESCRIPTION. The default is 0.
    s_o : TYPE, optional
        DESCRIPTION. The default is 0.
    min_int_size : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    centroids : TYPE
        DESCRIPTION.

    """
    centroids = list()

    for f_int in f_ints:
        l_ctrd = comp_centroid(
            x_coords[f_int[0] : f_int[-1] + 1], y_coords[f_int[0] : f_int[-1] + 1]
        )

        centroids.append(l_ctrd)

    return centroids


def comp_centroid(x_coords, y_coords):
    """

    Parameters
    ----------
    x_coords : TYPE
        DESCRIPTION.
    y_coords : TYPE
        DESCRIPTION.

    Returns
    -------
    list
        DESCRIPTION.

    """
    _len = len(x_coords)
    centroid_x = sum(x_coords) / _len
    centroid_y = sum(y_coords) / _len

    return [centroid_x, centroid_y]


def interval_merging(
    w_i, min_int_size=0, max_int_size=np.inf, p_o=0, s_o=0, status=None, proportion=0.95
):
    """

    Parameters
    ----------
    w_i : TYPE
        DESCRIPTION.
    min_int_size : TYPE, optional
        DESCRIPTION. The default is 0.
    p_o : TYPE, optional
        DESCRIPTION. The default is 0.
    s_o : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    intervals : TYPE
        DESCRIPTION.

    """
    intervals = list()

    for k, g in groupby(enumerate(w_i), lambda ix: ix[0] - ix[1]):
        i_l = list(map(itemgetter(1), g))
        if (i_l[-1] + s_o - (i_l[0] - p_o)) >= min_int_size and (
            i_l[-1] + s_o - (i_l[0] - p_o)
        ) < max_int_size:
            ends_local = [i_l[0] - p_o, i_l[-1] + s_o]

            if not status is None:
                if (
                    np.sum(status[i_l[0] - p_o : i_l[-1] + s_o + 1])
                    / (i_l[-1] + s_o - i_l[0] - p_o)
                    > proportion
                ):
                    intervals.append(ends_local)

            else:
                intervals.append(ends_local)

    return intervals


def dict_vectorize(dict_list):
    """

    Parameters
    ----------
    dict_list : TYPE
        DESCRIPTION.

    Returns
    -------
    res_ : TYPE
        DESCRIPTION.

    """
    res_ = []

    for k in dict_list[0].keys():
        res_.append(
            np.array(
                list(itertools.chain.from_iterable([list(d[k]) for d in dict_list]))
            )
        )

    res_ = np.array(res_).T

    return res_


def filter_binary_intervals_by_duration(results,
                                        sampling_frequency,
                                        min_fix_duration,
                                        max_fix_duration):
    """
    

    Parameters
    ----------
    results : TYPE
        DESCRIPTION.
    sampling_frequency : TYPE
        DESCRIPTION.
    min_fix_duration : TYPE
        DESCRIPTION.
    max_fix_duration : TYPE
        DESCRIPTION.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    def _dur_samples(intv):
        return int(intv[1] - intv[0] + 1)

    def _keep_by_duration(intervals, min_s, max_s, fs):
        min_n = int(np.ceil(min_s * fs))
        max_n = int(np.floor(max_s * fs))

        min_n = max(1, min_n)
        max_n = max(min_n, max_n)

        kept, rejected = [], []
        for itv in intervals:
            d = _dur_samples(itv)
            if (d >= min_n) and (d <= max_n):
                kept.append(itv)
            else:
                rejected.append(itv)
        return kept, rejected

    # infer n_s
    n_s = None
    for k in ("is_fixation", "is_saccade"):
        if k in results and results[k] is not None:
            n_s = len(results[k])
            break
    if n_s is None:
        raise ValueError("Cannot infer number of samples from results masks.")

    fs = float(sampling_frequency)

    fix_ints = results.get("fixation_intervals", [])
    fix_kept, _ = _keep_by_duration(fix_ints, min_fix_duration, max_fix_duration, fs)

    is_sac = np.ones(n_s, dtype=bool)
    is_fix = np.zeros(n_s, dtype=bool)

    for a, b in fix_kept:
        is_fix[a:b+1] = True
        is_sac[a:b+1] = False

    fix_out = interval_merging(np.where(is_fix)[0])
    sac_out = interval_merging(np.where(is_sac)[0])

    return {
        "is_saccade": is_sac,
        "saccade_intervals": sac_out,
        "is_fixation": is_fix,
        "fixation_intervals": fix_out,
    }


def filter_ternary_intervals_by_duration(results,
                                     sampling_frequency,
                                     min_fix_duration,
                                     max_fix_duration,
                                     min_pursuit_duration,
                                     max_pursuit_duration):
    """
    

    Parameters
    ----------
    results : TYPE
        DESCRIPTION.
    sampling_frequency : TYPE
        DESCRIPTION.
    min_fix_duration : TYPE
        DESCRIPTION.
    max_fix_duration : TYPE
        DESCRIPTION.
    min_pursuit_duration : TYPE
        DESCRIPTION.
    max_pursuit_duration : TYPE
        DESCRIPTION.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    def _dur_samples(intv): 
        return intv[1] - intv[0] + 1

    def _keep_by_duration(intervals, min_s, max_s, fs):
        min_n = int(np.ceil(min_s * fs))
        max_n = int(np.floor(max_s * fs))
      
        min_n = max(1, min_n)
        max_n = max(min_n, max_n)
        kept, rejected = [], []
        for itv in intervals:
            d = _dur_samples(itv)
            if (d >= min_n) and (d <= max_n):
                kept.append(itv)
            else:
                rejected.append(itv)
        return kept, rejected
 
    n_s = None
    for k in ("is_saccade", "is_pursuit", "is_fixation"):
        if k in results and results[k] is not None:
            n_s = len(results[k])
            break
    if n_s is None:
        raise ValueError("Cannot infer number of samples from results masks.")

    fs = float(sampling_frequency)
 
    fix_ints = results["fixation_intervals"]
    purs_ints = results["pursuit_intervals"]
 
    fix_kept, fix_bad = _keep_by_duration(fix_ints, min_fix_duration, max_fix_duration, fs)
    purs_kept, purs_bad = _keep_by_duration(purs_ints, min_pursuit_duration, max_pursuit_duration, fs)
 
    is_sac = np.ones(n_s, dtype=bool)
    is_fix = np.zeros(n_s, dtype=bool)
    is_purs = np.zeros(n_s, dtype=bool)
    
    for a, b in fix_kept:
        is_fix[a:b+1] = True
        is_sac[a:b+1] = False
    
    for a, b in purs_kept:
        is_purs[a:b+1] = True
        is_sac[a:b+1] = False
    
    # enforce exclusivity 
    is_purs[is_fix] = False
 
    fix_out = interval_merging(np.where(is_fix)[0])
    purs_out = interval_merging(np.where(is_purs)[0])
    sac_out = interval_merging(np.where(is_sac)[0])

    return {
        "is_saccade": is_sac,
        "saccade_intervals": sac_out,
        "is_pursuit": is_purs,
        "pursuit_intervals": purs_out,
        "is_fixation": is_fix,
        "fixation_intervals": fix_out,
    }


def standard_normalization(features, mu=None, sigma=None):
    """

    Parameters
    ----------
    features : TYPE
        DESCRIPTION.

    Returns
    -------
    features : TYPE
        DESCRIPTION.

    """
    if not isinstance(mu, np.ndarray):
        mu = np.mean(features, axis=0)

    if not isinstance(sigma, np.ndarray):
        sigma = np.std(features, axis=0)
        sigma[sigma == 0] = features[0][sigma == 0]

    features = (features - mu) / sigma

    return features, mu, sigma
