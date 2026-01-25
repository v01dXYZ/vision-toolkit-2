# -*- coding: utf-8 -*-
import numpy as np


def pre_process_IHOV(data_set, config):
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

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if config.get("distance_type", None) != "euclidean":
        raise AssertionError("'distance_type' must be set to 'euclidean' for I_HOV.")

    s_f = float(config["sampling_frequency"])

    x_array = np.asarray(data_set["x_array"], dtype=float)
    y_array = np.asarray(data_set["y_array"], dtype=float)
    n_s = int(x_array.shape[0])

    # keep config consistent, but don't rely on it
    config["nb_samples"] = n_s

    coord = np.stack([x_array, y_array], axis=0)  # (2, n_s)

    # durations (seconds) -> samples
    t_du = int(np.ceil(float(config.get("IHOV_duration_threshold", 0.2)) * s_f))
    t_du = max(1, t_du)

    t_av = int(np.ceil(float(config.get("IHOV_averaging_threshold", 0.2)) * s_f))
    t_av = max(1, t_av)
    if (t_av % 2) == 0:
        t_av += 1

    nb_ang_bin = int(config.get("IHOV_angular_bin_nbr", 36))
    nb_ang_bin = max(4, nb_ang_bin)
    ang_bin = 360.0 / nb_ang_bin


    def averaging(v_, t_av_):
        """
        

        Parameters
        ----------
        v_ : TYPE
            DESCRIPTION.
        t_av_ : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        n_ = len(v_)
        h = int(t_av_ / 2)
        conv = np.convolve(v_, np.ones(t_av_, dtype=float), mode="full") / float(t_av_)
        
        return conv[h : n_ + h]


    def comp_hist_from_vectors(vect_2xn):
        """
        

        Parameters
        ----------
        vect_2xn : TYPE
            DESCRIPTION.

        Returns
        -------
        hist : TYPE
            DESCRIPTION.

        """
        # magnitudes
        dist_ = np.linalg.norm(vect_2xn, axis=0)

        # angles in degrees in [0, 360)
        ang = np.degrees(np.mod(np.arctan2(vect_2xn[1, :], vect_2xn[0, :]), 2.0 * np.pi))

        # bins in [0, nb_ang_bin-1] 
        b = np.floor(ang / ang_bin).astype(int)
        b = np.clip(b, 0, nb_ang_bin - 1)

        hist = np.zeros(nb_ang_bin, dtype=float)
        # accumulate magnitudes per bin
        # (np.add.at handles repeated indices)
        np.add.at(hist, b, dist_)

        # rotate so max bin is at 0
        kmax = int(np.argmax(hist))
        if kmax != 0:
            hist = np.concatenate([hist[kmax:], hist[:kmax]], axis=0)

        s = float(np.sum(hist))
        if s > 0:
            hist /= s
        return hist

    # build per-sample histograms (3 windows)  
    hists_w_1 = np.zeros((n_s, nb_ang_bin), dtype=float)
    hists_w_2 = np.zeros((n_s, nb_ang_bin), dtype=float)
    hists_w_3 = np.zeros((n_s, nb_ang_bin), dtype=float)

    # window indices per i (clamped)
    for i in range(n_s):
        # past: vectors from (i) to points in [i-t_du, ..., i-1]
        idx_p = np.arange(i - t_du, i, dtype=int)
        idx_p = np.clip(idx_p, 0, n_s - 1)
        v1 = coord[:, i].reshape(2, 1) - coord[:, idx_p]  # (2, t_du)
        hists_w_1[i, :] = comp_hist_from_vectors(v1)

        # future: vectors from (i) to points in [i+1, ..., i+t_du]
        idx_f = np.arange(i + 1, i + t_du + 1, dtype=int)
        idx_f = np.clip(idx_f, 0, n_s - 1)
        v2 = coord[:, idx_f] - coord[:, i].reshape(2, 1)  # (2, t_du)
        hists_w_2[i, :] = comp_hist_from_vectors(v2)

        # across: vectors from past points to future points (paired)
        # matches your original: future reversed - past
        idx_f_rev = np.flip(idx_f)
        v3 = coord[:, idx_f_rev] - coord[:, idx_p]  # (2, t_du)
        hists_w_3[i, :] = comp_hist_from_vectors(v3)

    # temporal averaging per bin 
    a_hists_w_1 = np.zeros_like(hists_w_1)
    a_hists_w_2 = np.zeros_like(hists_w_2)
    a_hists_w_3 = np.zeros_like(hists_w_3)

    for k in range(nb_ang_bin):
        a_hists_w_1[:, k] = averaging(hists_w_1[:, k], t_av)
        a_hists_w_2[:, k] = averaging(hists_w_2[:, k], t_av)
        a_hists_w_3[:, k] = averaging(hists_w_3[:, k], t_av)

    # renormalize per row after averaging (safety)
    def _renorm(mat):
        s = np.sum(mat, axis=1, keepdims=True)
        s[s == 0] = 1.0
        return mat / s

    a_hists_w_1 = _renorm(a_hists_w_1)
    a_hists_w_2 = _renorm(a_hists_w_2)
    a_hists_w_3 = _renorm(a_hists_w_3)

    feat_ = np.concatenate((a_hists_w_1, a_hists_w_2, a_hists_w_3), axis=1)
    
    return feat_
