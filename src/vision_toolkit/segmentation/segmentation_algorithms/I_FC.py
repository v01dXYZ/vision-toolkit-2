# -*- coding: utf-8 -*-
import time
import numpy as np
from scipy.cluster.vq import kmeans2


 
def _dict_to_matrix(features):

    keys = sorted(features.keys())
    X = np.stack([np.asarray(features[k]).reshape(-1) for k in keys], axis=1)
    
    return X, keys

 

def _safe_norm_rows(M, eps=1e-12):
    
    n = np.linalg.norm(M, axis=1, keepdims=True)
    n = np.where(n < eps, 1.0, n)
    
    return M / n


def _safe_arccos(x):
    return np.arccos(np.clip(x, -1.0, 1.0))


 
def bcea(theta_coord, p, std_xy):
    """
    

    Parameters
    ----------
    theta_coord : TYPE
        DESCRIPTION.
    p : TYPE
        DESCRIPTION.
    std_xy : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    theta_x = theta_coord[0]
    theta_y = theta_coord[1]
    k = -np.log(1 - p)

    def pearson_corr(t_x, t_y):
        
        mx = np.mean(t_x)
        my = np.mean(t_y)
        xm, ym = t_x - mx, t_y - my
        _num = np.sum(xm * ym)
        _den = np.sqrt(np.sum(xm ** 2) * np.sum(ym ** 2))
        
        if _den == 0:
            return 0.0
        p_c = _num / _den
        
        return float(np.clip(p_c, -1.0, 1.0))

    p_c = pearson_corr(theta_x, theta_y)
    sd_x = float(std_xy[0])
    sd_y = float(std_xy[1])
    
    return 2 * np.pi * k * sd_x * sd_y * np.sqrt(max(0.0, 1 - p_c ** 2))


 
def pre_process_IFC(data_set, config):
    """
    

    Parameters
    ----------
    data_set : TYPE
        DESCRIPTION.
    config : TYPE
        DESCRIPTION.

    Raises
    ------
    AssertionError
        DESCRIPTION.
    KeyError
        DESCRIPTION.
    ValueError
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if config.get("distance_type") != "angular":
        raise AssertionError("'distance_type' must be set to 'angular' for I_FC.")
 
    required = ("x_array", "y_array", "theta_coord", "unitary_gaze_vectors", "absolute_speed")
    missing = [k for k in required if k not in data_set]
    if missing:
        raise KeyError(
            f"pre_process_IFC: missing keys in data_set: {missing}. "
            "Did you run OcculomotorSeries.generate(...) / Basic_Processing first?"
        )

    if config.get("verbose", False):
        print("Processing feature extraction...")

    start_time = time.time()

    s_f = float(config["sampling_frequency"])

    x_array = np.asarray(data_set["x_array"], dtype=float)
    y_array = np.asarray(data_set["y_array"], dtype=float)
    n_s = int(x_array.shape[0])

    # keep config consistent, but don't rely on it
    config["nb_samples"] = n_s

    theta_coord = np.asarray(data_set["theta_coord"], dtype=float)              # (2, n_s)
    unitary_gaze_vectors = np.asarray(data_set["unitary_gaze_vectors"], dtype=float)  # (3, n_s)
    a_sp = np.asarray(data_set["absolute_speed"], dtype=float)                  # (n_s,)

    if theta_coord.shape[1] != n_s or unitary_gaze_vectors.shape[1] != n_s or a_sp.shape[0] != n_s:
        raise ValueError("pre_process_IFC: length mismatch between arrays and nb_samples.")

    bcea_p = float(config.get("IFC_bcea_prob", 0.68))
    i2mc = bool(config.get("IFC_i2mc", True))

    if n_s < 2:
        # degenerate case: return empty but well-formed
        return np.zeros((n_s, 1), dtype=float)
 
    diff_ = np.zeros((2, n_s), dtype=float)
    diff_[0, :-1] = x_array[1:] - x_array[:-1]
    diff_[1, :-1] = y_array[1:] - y_array[:-1]
    diff_[:, -1] = diff_[:, -2]

    acc = np.zeros_like(a_sp)
    acc[:-1] = a_sp[1:] - a_sp[:-1]
    acc[-1] = acc[-2]
 
    dot_ = np.sum(unitary_gaze_vectors[:, :-1] * unitary_gaze_vectors[:, 1:], axis=0)
    rad_suc_dist = _safe_arccos(dot_)  # (n_s-1,)
    ang_suc_dist = np.zeros_like(a_sp)
    ang_suc_dist[:-1] = np.abs(rad_suc_dist / np.pi * 180.0)
    ang_suc_dist[-1] = ang_suc_dist[-2]  # FIX: last value should not remain 0
 
    suc_dir = np.mod(np.arctan2(diff_[1, :] + 1e-12, diff_[0, :] + 1e-12), 2.0 * np.pi)

    # windows in samples
    n_50 = max(1, int(np.ceil(0.050 * s_f)))
    n_11 = max(1, int(np.ceil(0.011 * s_f)))

 
    features = {
        "sf": np.ones(n_s) * s_f,
        "rms": np.zeros(n_s),
        "delta_rms": np.zeros(n_s),
        "std_x": np.zeros(n_s),
        "std_y": np.zeros(n_s),
        "delta_std_x": np.zeros(n_s),
        "delta_std_y": np.zeros(n_s),
        "bcea": np.zeros(n_s),
        "delta_bcea": np.zeros(n_s),
        "dispersion": np.zeros(n_s),
        "velocity": a_sp,
        "acceleration": acc,
        "med_dist": np.zeros(n_s),
        "mean_dist": np.zeros(n_s),
        "z_ray": np.zeros(n_s),
    }

 
    if i2mc:
        features["i2mc"] = np.zeros(n_s)
        t_du = int(np.floor(float(config.get("IFC_i2mc_window_duration", 0.2)) * s_f / 2))
        t_du = max(1, t_du)
        t_mo = int(np.ceil(float(config.get("IFC_i2mc_moving_threshold", 0.02)) * s_f))
        t_mo = max(1, t_mo)

        n_wc = int(2 * t_du + 1)
        n_wc_2 = int(np.ceil(n_wc / 2))
        n_wc_4 = int(np.ceil(n_wc / 4))

        f_w = np.zeros(n_s, dtype=float)
        norm_ = np.zeros(n_s, dtype=float)

   
    w_c_100_m = np.array([
        theta_coord[:, np.clip(np.arange(i - n_50, i + n_50 + 1, dtype=int), 0, n_s - 1)]
        for i in range(n_s)
    ])  # (n_s, 2, win)

    w_c_100_asd = np.array([
        ang_suc_dist[np.clip(np.arange(i - n_50, i + n_50 + 1, dtype=int), 0, n_s - 1)]
        for i in range(n_s)
    ])  # (n_s, win)

    w_c_100_ugv = np.array([
        unitary_gaze_vectors[:, np.clip(np.arange(i - n_50, i + n_50 + 1, dtype=int), 0, n_s - 1)]
        for i in range(n_s)
    ])  # (n_s, 3, win)

    w_c_22_sd = np.array([
        suc_dir[np.clip(np.arange(i - n_11, i + n_11 + 1, dtype=int), 0, n_s - 1)]
        for i in range(n_s)
    ])  # (n_s, win)

    # mean / median positions (unit vectors) per window
    mean_pos_v = np.mean(w_c_100_ugv, axis=2)   # (n_s, 3)
    mean_pos_v = _safe_norm_rows(mean_pos_v)

    med_pos_v = np.median(w_c_100_ugv, axis=2)  # (n_s, 3)
    med_pos_v = _safe_norm_rows(med_pos_v)

    # dispersion (theta space)
    disp_v = (
        (np.max(w_c_100_m[:, 0, :], axis=1) - np.min(w_c_100_m[:, 0, :], axis=1))
        + (np.max(w_c_100_m[:, 1, :], axis=1) - np.min(w_c_100_m[:, 1, :], axis=1))
    )

    std_v = np.std(w_c_100_m, axis=2)  # (n_s, 2)
    rms_v = np.sqrt(np.sum(w_c_100_asd ** 2, axis=1) / w_c_100_asd.shape[1])

    # Rayleigh z: based on suc_dir in short window
    r_mat = np.array([np.cos(w_c_22_sd), np.sin(w_c_22_sd)])  # (2, n_s, win)
    n_22 = r_mat.shape[2]
    rm_mat = np.sum(r_mat, axis=2) / float(n_22)              # (2, n_s)
    z_ray_v = (np.linalg.norm(rm_mat, axis=0) ** 2) * float(n_22)

    # BCEA per sample
    bcea_v = np.zeros(n_s, dtype=float)
    for i in range(n_s):
        w_idx = np.clip(np.arange(i - n_50, i + n_50 + 1, dtype=int), 0, n_s - 1)
        bcea_v[i] = bcea(theta_coord[:, w_idx], bcea_p, std_v[i])

    # i2mc-like transitions feature
    if i2mc:
        t_x = theta_coord[0, :]
        t_y = theta_coord[1, :]
        i = 0
        while i + t_mo < n_s:
            i += t_mo
            w_c = np.clip(np.arange(i - t_du, i + t_du + 1, dtype=int), 0, n_s - 1)

            X_1 = np.c_[t_x[w_c], t_y[w_c]]
            _, lbl_1 = kmeans2(X_1, 2, iter=15, minit="++")
            trs_1 = np.where(np.diff(lbl_1) != 0)[0] + 1
            w_1 = np.zeros(n_wc)
            if len(trs_1) > 0:
                w_1[trs_1] = 1.0 / len(trs_1)
            f_w[w_c] += w_1

            X_2 = np.c_[t_x[w_c][::2], t_y[w_c][::2]]
            _, lbl_2 = kmeans2(X_2, 2, iter=15, minit="++")
            trs_2 = np.where(np.diff(lbl_2) != 0)[0] + 1
            w_2 = np.zeros(n_wc_2)
            if len(trs_2) > 0:
                w_2[trs_2] = 1.0 / len(trs_2)
            f_w[w_c] += np.repeat(w_2, 2)[:n_wc]

            X_4 = np.c_[t_x[w_c][::4], t_y[w_c][::4]]
            _, lbl_4 = kmeans2(X_4, 2, iter=15, minit="++")
            trs_4 = np.where(np.diff(lbl_4) != 0)[0] + 1
            w_4 = np.zeros(n_wc_4)
            if len(trs_4) > 0:
                w_4[trs_4] = 1.0 / len(trs_4)
            f_w[w_c] += np.repeat(w_4, 4)[:n_wc]

            norm_[w_c] += 1.0

        norm_[norm_ == 0] = 1.0
        f_w /= norm_
        features["i2mc"] = f_w

    # fill computed features
    features["std_x"] = std_v[:, 0]
    features["std_y"] = std_v[:, 1]
    features["rms"] = rms_v
    features["dispersion"] = disp_v
    features["bcea"] = bcea_v
    features["z_ray"] = z_ray_v

    # deltas and mean/median distances
    for i in range(n_s):
        i_b = int(max(0, i - n_50))
        i_a = int(min(n_s - 1, i + n_50))

        features["delta_rms"][i] = rms_v[i_a] - rms_v[i_b]

        d_std = std_v[i_a] - std_v[i_b]
        features["delta_std_x"][i] = d_std[0]
        features["delta_std_y"][i] = d_std[1]

        features["delta_bcea"][i] = bcea_v[i_a] - bcea_v[i_b]

        # angle between mean unit vectors
        mean_dot = float(np.dot(mean_pos_v[i_a], mean_pos_v[i_b]))
        rad_mean_dist = _safe_arccos(mean_dot)
        features["mean_dist"][i] = abs(rad_mean_dist / np.pi * 180.0)

        # angle between median unit vectors
        med_dot = float(np.dot(med_pos_v[i_a], med_pos_v[i_b]))
        rad_med_dist = _safe_arccos(med_dot)
        features["med_dist"][i] = abs(rad_med_dist / np.pi * 180.0)

    if config.get("verbose", False):
        print("--- feature extraction: %.3f seconds ---" % (time.time() - start_time))

    # to matrix
    X, _keys = _dict_to_matrix(features)
  
    return X

