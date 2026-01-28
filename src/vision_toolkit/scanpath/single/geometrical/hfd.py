# -*- coding: utf-8 -*-
 
import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve

from vision_toolkit.utils.binning import spatial_bin
from vision_toolkit.visualization.scanpath.single.geometrical import plot_HFD


class HiguchiFractalDimension:
    def __init__(self, scanpath, HFD_hilbert_iterations, HFD_k_max):
     
        h_i = HFD_hilbert_iterations
        if h_i is None or h_i < 1:
            raise ValueError("HFD_hilbert_iterations (h_i) must be >= 1.")

        if HFD_k_max is None or HFD_k_max < 1:
            raise ValueError("HFD_k_max must be >= 1.")

        self.k_m = HFD_k_max

        x_grid_size = y_grid_size = 2**h_i
 
        x_size = scanpath.config["size_plan_x"]
        y_size = scanpath.config["size_plan_y"]
 
        self.sc_b = spatial_bin(
            scanpath.values[0:2], x_grid_size, y_grid_size, x_size, y_size
        )
 
        h_c = HilbertCurve(h_i, 2)
 
        self.dist_ = h_c.distances_from_points(self.sc_b.T)
 
        s_, x_, l_ = self.compute_hfd(np.array(self.dist_))
 
        if s_ is None or len(s_) == 0 or not np.isfinite(s_[0]):
            fd = float("nan")
        else:
            fd = float(s_[0])

        self.results = dict(
            {
                "fractal_dimension": fd,
                "log_lengths": l_,
                "log_inverse_time_intervals": x_,
            }
        )

        if scanpath.config.get("display_results", False):
            dist_h = list(range(0, int(2 ** (h_i * 2) - 1)))
            h_pts = h_c.points_from_distances(dist_h)

            plot_HFD(
                self.sc_b,
                self.dist_,
                np.array(h_pts),
                s_,
                x_,
                l_,
                scanpath.config.get("display_path"),
            )

    def compute_hfd(self, dist_):
        
        dist_ = np.asarray(dist_, dtype=float)
        n = len(dist_)
    
        if n < 2:
            return np.array([np.nan, np.nan]), np.array([]), np.array([])
    
        l_, x_ = [], []
        k_max = min(self.k_m, n // 2)
    
        for k in range(1, k_max + 1):
            Lk = []
            for m in range(k):
                idxs = np.arange(1, int(np.floor((n - m - 1) / k)) + 1)
                if idxs.size == 0:
                    continue
    
                diffs = np.abs(dist_[m + idxs * k] - dist_[m + (idxs - 1) * k])
                lm = np.sum(diffs)
                norm = (n - 1) / (idxs.size * k)
                Lk.append(lm * norm)
    
            if len(Lk) == 0:
                l_.append(np.nan)
            else:
                l_.append(np.log(np.mean(Lk)))
    
            x_.append(np.log(1.0 / k))
    
        x_ = np.array(x_)
        l_ = np.array(l_)
    
        idx = np.isfinite(x_) & np.isfinite(l_)
        if np.sum(idx) < 2:
            s_ = np.array([np.nan, np.nan])
        else:
            s_ = np.polyfit(x_[idx], l_[idx], 1)
    
        return s_, x_, l_
