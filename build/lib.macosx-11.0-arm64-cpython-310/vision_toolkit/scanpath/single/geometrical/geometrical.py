# -*- coding: utf-8 -*-

import numpy as np

from vision_toolkit.scanpath.scanpath_base import Scanpath
from vision_toolkit.scanpath.single.geometrical.bcea import bcea
from vision_toolkit.scanpath.single.geometrical.convex_hull import ConvexHull
from vision_toolkit.scanpath.single.geometrical.hfd import HiguchiFractalDimension
from vision_toolkit.scanpath.single.geometrical.k_coefficient import modified_k_coefficient
from vision_toolkit.scanpath.single.geometrical.voronoi import VoronoiCells
from vision_toolkit.segmentation.processing.binary_segmentation import BinarySegmentation


class GeometricalAnalysis:
    def __init__(self, input, **kwargs):
       
        verbose = kwargs.get("verbose", True)

        if verbose:
            print("Processing Geometrical Analysis...\n")

        if isinstance(input, str):
            self.scanpath = Scanpath.generate(input, **kwargs)

        elif isinstance(input, BinarySegmentation):
            self.scanpath = Scanpath.generate(input, **kwargs)

        elif isinstance(input, Scanpath):
            self.scanpath = input

        else:
            raise ValueError(
                "Input must be a csv, or a BinarySegmentation, or a Scanpath object"
            )

        if verbose:
            print("...Geometrical Analysis done\n")
 
    
    def _n_points(self):
        v = self.scanpath.values
        return int(v.shape[1])


    def _has_min_points(self, n_min):
        return self._n_points() >= n_min

    
    def scanpath_length(self):
        
        n = self._n_points()
        if n < 2:
            return {"length": 0.0}

        x_ = self.scanpath.values[:2]
        d_ = float(np.sum(np.linalg.norm(x_[:, 1:] - x_[:, :-1], axis=0)))
        results = dict({"length": d_})
        
        return results


    def scanpath_BCEA(self, BCEA_probability, display_results=True, display_path=None):
        
        if not (0.0 < BCEA_probability < 1.0):
            raise ValueError("BCEA_probability must be in (0, 1).")
        
        if not self._has_min_points(2):
            return {"BCEA": float("nan")}

        self.scanpath.config.update(
            {"display_results": display_results, "display_path": display_path}
        )

        # Compute BCEA (guard against degenerate correlation/std)
        try:
            bcea_ = float(bcea(self.scanpath, BCEA_probability))
        except Exception:
            bcea_ = float("nan")

        results = dict({"BCEA": bcea_})
        self.scanpath.verbose(dict({"scanpath_BCEA_probability": BCEA_probability}))
        
        return results


    def scanpath_k_coefficient(self, display_results=True, display_path=None):
    
        if not self._has_min_points(2):
            return {"k_coefficient": float("nan")}
    
        self.scanpath.config.update(
            {"display_results": display_results, "display_path": display_path}
        )
    
        v = self.scanpath.values
        if v.ndim != 2 or v.shape[0] < 3:
            raise ValueError(
                "scanpath.values must be a 2D array with at least 3 rows (x, y, duration)."
            )
    
        # Departure durations (i) aligned with saccades i->i+1
        dur = v[2, :-1]
        a_s = np.linalg.norm(v[0:2, 1:] - v[0:2, :-1], axis=0)
    
        if dur.size < 2 or a_s.size < 2:
            k_c = float("nan")
        else:
            std_d = np.std(dur, ddof=1)
            std_a = np.std(a_s, ddof=1)
    
            if std_d == 0 or std_a == 0 or (not np.isfinite(std_d)) or (not np.isfinite(std_a)):
                k_c = float("nan")
            else:
                try:
                    k_c = float(modified_k_coefficient(self.scanpath))
                except Exception:
                    k_c = float("nan")
    
        results = {"k_coefficient": k_c}
        self.scanpath.verbose()
        
        return results


    def scanpath_voronoi_cells(self, display_results=True, display_path=None, get_raw=True):
        
        if not self._has_min_points(3):
            return dict(
                {"skewness": float("nan"), "gamma_parameter": float("nan"), "voronoi_areas": []})

        self.scanpath.config.update(
            {"display_results": display_results, "display_path": display_path}
        )

        # Voronoi can fail on duplicates/colinear points (Qhull)
        try:
            v_a = VoronoiCells(self.scanpath)
            results = v_a.results  # keep original behavior (no copy)
        except Exception:
            results = dict(
                {"skewness": float("nan"), "gamma_parameter": float("nan"), "voronoi_areas": []}
            )

        self.scanpath.verbose(dict({"get_raw": get_raw}))

        if not get_raw and "voronoi_areas" in results:
            del results["voronoi_areas"]

        return results


    def scanpath_convex_hull(self, display_results=True, display_path=None, get_raw=True):
        
        if not self._has_min_points(3):
            return dict({"hull_area": float("nan"), "hull_apex": None})

        self.scanpath.config.update(
            {"display_results": display_results, "display_path": display_path}
        )

        try:
            c_h = ConvexHull(self.scanpath)
            results = c_h.results  # keep original behavior (no copy)
        except Exception:
            results = dict({"hull_area": float("nan"), "hull_apex": None})

        self.scanpath.verbose(dict({"get_raw": get_raw}))

        if not get_raw and "hull_apex" in results:
            del results["hull_apex"]

        return results


    def scanpath_HFD(
        self, HFD_hilbert_iterations, HFD_k_max,
        display_results=True, display_path=None, get_raw=True
    ):
        
        if not self._has_min_points(3):
            return dict(
                {
                    "fractal_dimension": float("nan"),
                    "log_lengths": np.array([]),
                    "log_inverse_time_intervals": np.array([]),
                }
            )

        if HFD_hilbert_iterations is None or HFD_hilbert_iterations < 1:
            raise ValueError("HFD_hilbert_iterations must be >= 1.")
        if HFD_k_max is None or HFD_k_max < 1:
            raise ValueError("HFD_k_max must be >= 1.")

        self.scanpath.config.update(
            {"display_results": display_results, "display_path": display_path}
        )

        try:
            h_fd = HiguchiFractalDimension(self.scanpath, HFD_hilbert_iterations, HFD_k_max)
            results = h_fd.results  # keep original behavior (no copy)
        except Exception:
            results = dict(
                {
                    "fractal_dimension": float("nan"),
                    "log_lengths": np.array([]),
                    "log_inverse_time_intervals": np.array([]),
                }
            )

        self.scanpath.verbose(
            dict(
                {
                    "scanpath_HFD_hilbert_iterations": HFD_hilbert_iterations,
                    "scanpath_HFD_k_max": HFD_k_max,
                    "get_raw": get_raw,
                }
            )
        )

        if not get_raw:
            if "log_lengths" in results:
                del results["log_lengths"]
            if "log_inverse_time_intervals" in results:
                del results["log_inverse_time_intervals"]

        return results


def scanpath_length(input, **kwargs):
    if isinstance(input, GeometricalAnalysis):
        results = input.scanpath_length()

    else:
        geometrical_analysis = GeometricalAnalysis(input, **kwargs)
        results = geometrical_analysis.scanpath_length()

    return results


def scanpath_BCEA(input, **kwargs):
    BCEA_probability = kwargs.get("scanpath_BCEA_probability", 0.68)
    display_results = kwargs.get("display_results", True)
    display_path = kwargs.get("display_path", None)

    if isinstance(input, GeometricalAnalysis):
        results = input.scanpath_BCEA(BCEA_probability, display_results, display_path)

    else:
        geometrical_analysis = GeometricalAnalysis(input, **kwargs)
        results = geometrical_analysis.scanpath_BCEA(
            BCEA_probability, display_results, display_path
        )

    return results


def scanpath_k_coefficient(input, **kwargs):
    display_results = kwargs.get("display_results", True)
    display_path = kwargs.get("display_path", None)

    if isinstance(input, GeometricalAnalysis):
        results = input.scanpath_k_coefficient(display_results, display_path)

    else:
        geometrical_analysis = GeometricalAnalysis(input, **kwargs)
        results = geometrical_analysis.scanpath_k_coefficient(
            display_results, display_path
        )

    return results


def scanpath_voronoi_cells(input, **kwargs):
    display_results = kwargs.get("display_results", True)
    display_path = kwargs.get("display_path", None)
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, GeometricalAnalysis):
        results = input.scanpath_voronoi_cells(display_results, display_path, get_raw)

    else:
        geometrical_analysis = GeometricalAnalysis(input, **kwargs)
        results = geometrical_analysis.scanpath_voronoi_cells(
            display_results, display_path, get_raw
        )

    return results


def scanpath_convex_hull(input, **kwargs):
    display_results = kwargs.get("display_results", True)
    display_path = kwargs.get("display_path", None)
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, GeometricalAnalysis):
        results = input.scanpath_convex_hull(display_results, display_path, get_raw)

    else:
        geometrical_analysis = GeometricalAnalysis(input, **kwargs)
        results = geometrical_analysis.scanpath_convex_hull(
            display_results, display_path, get_raw
        )

    return results


def scanpath_HFD(input, **kwargs):
    HFD_hilbert_iterations = kwargs.get("scanpath_HFD_hilbert_iterations", 4)
    HFD_k_max = kwargs.get("scanpath_HFD_k_max", 10)

    display_results = kwargs.get("display_results", True)
    display_path = kwargs.get("display_path", None)
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, GeometricalAnalysis):
        results = input.scanpath_HFD(
            HFD_hilbert_iterations, HFD_k_max, display_results, display_path, get_raw
        )

    else:
        geometrical_analysis = GeometricalAnalysis(input, **kwargs)
        results = geometrical_analysis.scanpath_HFD(
            HFD_hilbert_iterations, HFD_k_max, display_results, display_path, get_raw
        )

    return results
