# -*- coding: utf-8 -*-


import numpy as np
from sklearn.cluster import AffinityPropagation

from vision_toolkit.utils.identification_utils import compute_aoi_sequence
from vision_toolkit.visualization.aoi.basic_representation import (
    display_aoi_identification, display_aoi_identification_reference_image)


def process_IAP(values, config, ref_image=None):
    pos_ = values[0:2]
    dur_ = values[2]

    center_method = config["AoI_IAP_centers"]

    ap = AffinityPropagation(max_iter=1000, damping=0.8).fit(pos_.T)
    seq_ = ap.labels_

    centers_ = dict({})
    clus_ = dict({})
    to_relabel = []
    i_ = 0

    # To avoid single point clusters
    for i in range(list(set(seq_))[-1] + 1):
        vals_ = np.argwhere(np.array(seq_) == int(i)).T[0]
        if len(vals_) >= 2:
            clus_.update({chr(i_ + 65): vals_})
            seq_[vals_] = i_
            if center_method == "mean":
                centers_.update({chr(i_ + 65): np.mean(pos_[:, vals_], axis=1)})
            elif center_method == "raw_IAP":
                # ap.cluster_centers_ contient 1 centre par cluster conservé par AP.
                # Ici i_ est le nouvel index compact; i est l'ancien label AP.
                # Pour éviter mismatch, on préfère toujours "mean" si possible.
                centers_.update({chr(i_ + 65): np.mean(pos_[:, vals_], axis=1)})
            i_ += 1
        else:
            to_relabel += list(vals_)

    # --- Fallback: no valid centers (all clusters are singletons) ---
    if len(centers_) == 0:
        clus_ = {'A': np.arange(pos_.shape[1], dtype=int)}
        centers_ = {'A': np.mean(pos_, axis=1)}
        seq_ = np.zeros(pos_.shape[1], dtype=int)
        to_relabel = []

    # Relabel single points
    if to_relabel != []:
        centers_array = np.array([centers_[k_] for k_ in sorted(list(centers_.keys()))])

        for val in to_relabel:
            pos_l = pos_[:, val]
            d_ = np.sum((centers_array - pos_l) ** 2, axis=1)
            c_val = np.argmin(d_)
            seq_[val] = c_val
            old_clus = sorted(list(clus_[chr(c_val + 65)]) + [val])
            clus_[chr(c_val + 65)] = np.array(old_clus)

    seq_, seq_dur = compute_aoi_sequence(seq_, dur_, config)

    if config["display_AoI_identification"]:
        if ref_image is None:
            display_aoi_identification(pos_, clus_, config)
        else:
            display_aoi_identification_reference_image(pos_, clus_, config, ref_image)

    return {
        "AoI_sequence": seq_,
        "AoI_durations": seq_dur,
        "centers": centers_,
        "clustered_fixations": clus_,
    }
