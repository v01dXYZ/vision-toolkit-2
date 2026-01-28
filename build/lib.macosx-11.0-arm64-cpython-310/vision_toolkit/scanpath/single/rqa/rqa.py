# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial.distance import cdist

from vision_toolkit.scanpath.scanpath_base import Scanpath
from vision_toolkit.scanpath.single.rqa.rqa_base import RecurrenceBase
from vision_toolkit.segmentation.processing.binary_segmentation import BinarySegmentation
from vision_toolkit.visualization.scanpath.single.rqa import (
    plot_RQA,
    plot_RQA_determinism,
    plot_RQA_laminarity)

 
## See https://drive.google.com/file/d/14mKqSnkGl08jzpgX_aRl5qsT4TqKDjk6/view

class RQAAnalysis(RecurrenceBase):
    def __init__(self, input, **kwargs):
      
        verbose = kwargs.get("verbose", True)
        display_results = kwargs.get("display_results", True)
        display_path = kwargs.get("display_path", None)

        if verbose:
            print("Processing RQA Analysis...\n")

        if isinstance(input, str):
            self.scanpath = Scanpath.generate(input, **kwargs)

        elif isinstance(input, BinarySegmentation):
            self.scanpath = Scanpath.generate(input, **kwargs)

        elif isinstance(input, Scanpath):
            self.scanpath = input

        else:
            raise ValueError(
                "Input must be a csv, a BinarySegmentation or a Scanpath object"
            )

        d_thrs = (
            np.linalg.norm(
                np.array(
                    [
                        self.scanpath.config["size_plan_x"],
                        self.scanpath.config["size_plan_y"],
                    ]
                )
            )
            * 0.1
        )
    
        self.scanpath.config.update(
            {
                "scanpath_RQA_distance_threshold": kwargs.get(
                    "scanpath_RQA_distance_threshold", d_thrs
                ),
                "scanpath_RQA_minimum_length": kwargs.get(
                    "scanpath_RQA_minimum_length", 2
                ),
                "verbose": verbose,
                "display_results": display_results,
                "display_path": display_path
            }
        )

        super().__init__(self.scanpath.values)

        ## Compute the recurrence matrix r_m and the sum r_p of
        ## recurrences in the upper triangle of the recurrence matrix
        self.r_m, self.r_p = self.comp_recurrence_matrix()

        ## Compute the upper trinagle matrix
        self.r_u = np.zeros((self.n, self.n))
        r_c = np.triu_indices(self.n, 1)
        self.r_u[r_c] = self.r_m[r_c]

        L = self.scanpath.config["scanpath_RQA_minimum_length"]

        # verticales dans r_u (OK)
        self.v_set = self.find_lines(self.r_u, L, "vertical")
        
        # horizontales = verticales dans r_u.T, puis remap coords (i,j) -> (j,i)
        h_set_T = self.find_lines(self.r_u.T, L, "vertical")
        self.h_set = [line[:, [1, 0]] for line in h_set_T]

        ## Compute the set of diagonal lines
        self.d_set = self.find_diags(
            self.r_u, self.scanpath.config["scanpath_RQA_minimum_length"]
        )

        if display_results:
            plot_RQA(self.r_m, display_path)
 
        if verbose:
            print("...RQA Analysis done\n")


    def comp_recurrence_matrix(self):
     
        s_ = self.s_[0:2]
        n = self.n

        d_m = cdist(s_.T, s_.T, metric="euclidean")

        r_m = (d_m < self.scanpath.config["scanpath_RQA_distance_threshold"]).astype(
            int
        )
        r_p = (np.sum(r_m) - n) / 2

        return r_m, r_p


    def scanpath_RQA_recurrence_rate(self):
        
        denom = (self.n - 1) * self.n
        if denom <= 0 or self.r_p <= 0:
            r_r = 0.0
        else:
            r_r = (200.0 * self.r_p) / denom
    
        results = {"RQA_recurrence_rate": float(r_r)}
        self.scanpath.verbose()
        
        return results
    
    
    def scanpath_RQA_laminarity(self, display_results=True, display_path=None):
        
        self.scanpath.config.update({"display_results": display_results, "display_path": display_path})
        s_l = sum(len(v) for v in self.v_set) + sum(len(h) for h in self.h_set)
    
        if self.r_p <= 0 or s_l == 0:
            lam = 0.0
        else:
            lam = (50.0 * s_l) / self.r_p
    
        if self.scanpath.config["display_results"]:
            plot_RQA_laminarity(self.r_m, self.h_set, self.v_set, self.scanpath.config['display_path'])
    
        results = {"RQA_laminarity": float(lam)}
        
        self.scanpath.verbose()
        return results
    
    
    def scanpath_RQA_determinism(self, display_results=True, display_path=None):
        
        self.scanpath.config.update({"display_results": display_results, "display_path": display_path})
        s_l = sum(len(d) for d in self.d_set)
    
        if self.r_p <= 0 or s_l == 0:
            det = 0.0
        else:
            det = (100.0 * s_l) / self.r_p
    
        if self.scanpath.config["display_results"]:
            plot_RQA_determinism(self.r_m, self.d_set, self.scanpath.config['display_path'])
    
        results = {"RQA_determinism": float(det)}
        self.scanpath.verbose()
        
        return results
    
    
    def scanpath_RQA_CORM(self):
        
        n = self.n
        r_m = self.r_m
        denom = (n - 1) * self.r_p
    
        if denom <= 0:
            corm = 0.0
        else:
            r_ = 0
            for i in range(n - 1):
                for j in range(i + 1, n):
                    r_ += r_m[i, j] * (j - i)
            corm = 100.0 * r_ / denom
    
        results = {"RQA_CORM": float(corm)}
        self.scanpath.verbose()
        
        return results
    
    
    def scanpath_RQA_entropy(self):
        
        d_set = self.d_set
    
        if len(d_set) == 0:
            entropy = 0.0
        else:
            l_s = np.array([len(d) for d in d_set], dtype=int)
            l_s = l_s[l_s > 0]
            if l_s.size == 0:
                entropy = 0.0
            else:
                _, c_ = np.unique(l_s, return_counts=True)
                p_ = c_ / np.sum(c_)
                entropy = 0.0
                for p in p_:
                    if p > 0:
                        entropy -= p * np.log(p)
    
        results = {"RQA_entropy": float(entropy)}
        self.scanpath.verbose()
        
        return results


def scanpath_RQA_recurrence_rate(input, **kwargs):
 
    if isinstance(input, RQAAnalysis):
        results = input.scanpath_RQA_recurrence_rate()

    else:
        geometrical_analysis = RQAAnalysis(input, **kwargs)
        results = geometrical_analysis.scanpath_RQA_recurrence_rate()

    return results


def scanpath_RQA_laminarity(input, **kwargs):
    display_results = kwargs.get("display_results", True)
    display_path = kwargs.get("display_path", None)

    if isinstance(input, RQAAnalysis):
        results = input.scanpath_RQA_laminarity(display_results, display_path)

    else:
        geometrical_analysis = RQAAnalysis(input, **kwargs)
        results = geometrical_analysis.scanpath_RQA_laminarity(display_results, display_path)

    return results


def scanpath_RQA_determinism(input, **kwargs):
    display_results = kwargs.get("display_results", True)
    display_path = kwargs.get("display_path", None)

    if isinstance(input, RQAAnalysis):
        results = input.scanpath_RQA_determinism(display_results, display_path)

    else:
        geometrical_analysis = RQAAnalysis(input, **kwargs)
        results = geometrical_analysis.scanpath_RQA_determinism(display_results, display_path)

    return results


def scanpath_RQA_CORM(input, **kwargs):

    if isinstance(input, RQAAnalysis):
        results = input.scanpath_RQA_CORM()

    else:
        geometrical_analysis = RQAAnalysis(input, **kwargs)
        results = geometrical_analysis.scanpath_RQA_CORM()

    return results


def scanpath_RQA_entropy(input, **kwargs):
  
    if isinstance(input, RQAAnalysis):
        results = input.scanpath_RQA_entropy()

    else:
        geometrical_analysis = RQAAnalysis(input, **kwargs)
        results = geometrical_analysis.scanpath_RQA_entropy()

    return results
