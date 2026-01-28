# -*- coding: utf-8 -*-

import numpy as np


def modified_k_coefficient(scanpath):
    """
    Modified k-coefficient using the *departure fixation* convention:
    each saccade i -> i+1 is paired with the duration of fixation i.

    values shape: (3, n) with rows [x, y, duration]
    """
    values = np.asarray(scanpath.values)

    # Basic shape check
    if values.ndim != 2 or values.shape[0] < 3:
        raise ValueError(
            "modified_k_coefficient requires scanpath.values with at least 3 rows (x, y, duration)."
        )

    n_ = values.shape[1]
    if n_ < 2:
        return np.nan

    # Saccade amplitudes: i -> i+1 (length n-1)
    a_s = np.linalg.norm(values[0:2, 1:] - values[0:2, :-1], axis=0)

    # Departure fixation durations: fixation i (length n-1)
    d = values[2, :-1]

    # Need at least 2 samples for ddof=1
    if d.size < 2 or a_s.size < 2:
        return np.nan

    mu_d = np.mean(d)
    std_d = np.std(d, ddof=1)

    mu_a = np.mean(a_s)
    std_a = np.std(a_s, ddof=1)

    if std_d == 0 or std_a == 0 or not np.isfinite(std_d) or not np.isfinite(std_a):
        return np.nan

    # k_j for each transition i -> i+1 uses duration of departure fixation (i)
    k_j = (d - mu_d) / std_d - (a_s - mu_a) / std_a

    denom = float(n_ - 1)
    if denom <= 0:
        return np.nan

    return float(np.sum(k_j) / denom)

