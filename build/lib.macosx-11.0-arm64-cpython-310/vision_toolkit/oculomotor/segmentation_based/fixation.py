# -*- coding: utf-8 -*-

 

from vision_toolkit.segmentation.processing.binary_segmentation import BinarySegmentation

import numpy as np
from numpy.linalg import norm
 


class FixationAnalysis(BinarySegmentation):
    """
    For a fixation [start,end]:
        * positions:      start .. end        (n_samples = end-start+1)
        * speeds:       start .. end-1      (n_vel = n_samples-1)  => slice a_sp[start:end]
    """

    def __init__(self, input, **kwargs):
        verbose = kwargs.get("verbose", True)

        if verbose:
            print("Processing Fixation Analysis...\n")

        if isinstance(input, BinarySegmentation):
            self.__dict__ = input.__dict__.copy()
            self.config.update({"verbose": verbose})
        else:
            sampling_frequency = kwargs.get("sampling_frequency", None)
            assert sampling_frequency is not None, "Sampling frequency must be specified"
            super().__init__(input, **kwargs)

        self.s_f = float(self.config["sampling_frequency"])
        assert len(self.segmentation_results["fixation_intervals"]) > 0, "No fixation identified"

        if verbose:
            print("...Fixation Analysis done\n")

    # @classmethod
    # def generate(cls, input, **kwargs):
        
    #     if isinstance(input, FixationAnalysis):
    #         return input
    #     return cls(input, **kwargs)


    def _intervals(self):
        return self.segmentation_results["fixation_intervals"]

    def _n_samples_per_interval(self, intervals):
        
        a_i = np.asarray(intervals, dtype=np.int64)
        return (a_i[:, 1] - a_i[:, 0] + 1).astype(np.float64)

    def _speed_array(self):
        return np.asarray(self.data_set["absolute_speed"], dtype=np.float64)

    def _speed_segment(self, start, end):
        
        if end <= start:
            return np.array([], dtype=np.float64)
        a_sp = self._speed_array()
        return a_sp[start:end]

    def _safe_sd(self, x):
        
        x = np.asarray(x, dtype=np.float64)
        return float(np.nanstd(x, ddof=1)) if np.sum(np.isfinite(x)) >= 2 else 0.0

    
    def fixation_count(self):
        return {"count": int(len(self._intervals()))}


    def fixation_frequency(self):
        
        ct = len(self._intervals())
        denom = (self.config["nb_samples"] / self.s_f)
        f = ct / denom if denom > 0 else np.nan
        
        return {"frequency": float(f)}


    def fixation_frequency_wrt_labels(self):
        
        ct = len(self._intervals())
        labeled = float(np.sum(self.segmentation_results["is_labeled"]))
        denom = labeled / self.s_f
        f = ct / denom if denom > 0 else np.nan
        
        return {"frequency": float(f)}


    def fixation_durations(self, get_raw=True):
        
        a_i = np.asarray(self._intervals(), dtype=np.int64)
        a_d = (a_i[:, 1] - a_i[:, 0] + 1) / self.s_f  # inclusive

        results = {
            "duration_mean": float(np.nanmean(a_d)),
            "duration_sd": self._safe_sd(a_d),
            "raw": a_d,
        }
        
        if not get_raw:
            del results["raw"]
            
        return results


    def fixation_centroids(self):
        return {"centroids": np.asarray(self.segmentation_results["centroids"])}

    
    def fixation_mean_velocities(self):
        
        m_sp, sd_sp = [], []
        for start, end in self._intervals():
            seg = self._speed_segment(start, end)  # start..end-1
            m_sp.append(float(np.nanmean(seg)) if seg.size else np.nan)
            sd_sp.append(self._safe_sd(seg) if seg.size else np.nan)

        return {
            "velocity_means": np.asarray(m_sp, dtype=np.float64),
            "velocity_sd": np.asarray(sd_sp, dtype=np.float64),
        }


    def fixation_average_velocity_means(self, weighted=False, get_raw=True, weight_mode="diffs"):
        
        m_sp = self.fixation_mean_velocities()["velocity_means"]

        if not weighted:
            results = {"average_velocity_means": float(np.nanmean(m_sp)), "raw": m_sp}
            
            if not get_raw:
                del results["raw"]
                
            return results

        n_samples = self._n_samples_per_interval(self._intervals())
        if weight_mode == "samples":
            w = np.maximum(n_samples, 0.0)
        else:  # diffs
            w = np.maximum(n_samples - 1.0, 0.0)

        denom = np.nansum(w)
        w_v = float(np.nansum(w * m_sp) / denom) if denom > 0 else np.nan

        results = {"weighted_average_velocity_means": w_v, "raw": m_sp}
       
        if not get_raw:
            del results["raw"]
            
        return results


    def fixation_average_velocity_deviations(self, get_raw=True, weight_mode="diffs"):
       
        sd_sp = self.fixation_mean_velocities()["velocity_sd"]

        n_samples = self._n_samples_per_interval(self._intervals())
        if weight_mode == "samples":
            w = np.maximum(n_samples, 0.0)
        else:
            w = np.maximum(n_samples - 1.0, 0.0)

        denom = np.nansum(w)
        a_sd = float(np.sqrt(np.nansum(w * (sd_sp ** 2)) / denom)) if denom > 0 else np.nan

        results = {"average_velocity_sd": a_sd, "raw": sd_sp}
        
        if not get_raw:
            del results["raw"]
            
        return results


    def fixation_drift_displacements(self, get_raw=True):
        x_a = self.data_set["x_array"]
        y_a = self.data_set["y_array"]
        z_a = self.data_set["z_array"]

        dist_ = self.distances[self.config["distance_type"]]

        dsp = []
        for start, end in self._intervals():
            dsp.append(
                dist_(
                    np.array([x_a[start], y_a[start], z_a[start]]),
                    np.array([x_a[end], y_a[end], z_a[end]]),
                )
            )

        dsp = np.asarray(dsp, dtype=np.float64)
        results = {
            "drift_displacement_mean": float(np.nanmean(dsp)),
            "drift_displacement_sd": self._safe_sd(dsp),
            "raw": dsp,
        }
        
        if not get_raw:
            del results["raw"]
            
        return results


    def fixation_drift_distances(self, get_raw=True):
        x_a = self.data_set["x_array"]
        y_a = self.data_set["y_array"]
        z_a = self.data_set["z_array"]
        n_s = len(x_a)

        dist_ = self.distances[self.config["distance_type"]]

        t_cum = []
        stack_ = np.concatenate(
            (
                x_a.reshape((1, n_s)),
                y_a.reshape((1, n_s)),
                z_a.reshape((1, n_s)),
            ),
            axis=0,
        )

        for start, end in self._intervals():
            if end <= start:
                t_cum.append(np.nan)
                continue

            if self.config["distance_type"] == "euclidean":
                l_a = stack_[:, start : end + 1]
                l_c = float(np.nansum(norm(l_a[:, 1:] - l_a[:, :-1], axis=0)))
            else:
                l_c = float(
                    np.nansum(
                        [
                            dist_(
                                np.array([x_a[k], y_a[k], z_a[k]]),
                                np.array([x_a[k + 1], y_a[k + 1], z_a[k + 1]]),
                            )
                            for k in range(start, end)
                        ]
                    )
                )

            t_cum.append(l_c)

        t_cum = np.asarray(t_cum, dtype=np.float64)
        results = {
            "drift_cumul_distance_mean": float(np.nanmean(t_cum)),
            "drift_cumul_distance_sd": self._safe_sd(t_cum),
            "raw": t_cum,
        }
        
        if not get_raw:
            del results["raw"]
            
        return results


    def fixation_drift_velocities(self, get_raw=True, duration_mode="diffs"):
  
        d_d = np.asarray(self.fixation_drift_displacements(get_raw=True)["raw"], dtype=np.float64)
        n_samples = self._n_samples_per_interval(self._intervals())

        if duration_mode == "samples":
            dur_s = n_samples / self.s_f
        else:
            dur_s = np.maximum(n_samples - 1.0, 0.0) / self.s_f

        d_vel = np.full_like(d_d, np.nan, dtype=np.float64)
        mask = np.isfinite(d_d) & np.isfinite(dur_s) & (dur_s > 0)
        d_vel[mask] = d_d[mask] / dur_s[mask]

        results = {
            "drift_velocity_mean": float(np.nanmean(d_vel)),
            "drift_velocity_sd": self._safe_sd(d_vel),
            "raw": d_vel,
        }
        
        if not get_raw:
            del results["raw"]
            
        return results


    def fixation_BCEA(self, BCEA_probability=0.68, get_raw=True):
        
        def pearson_corr_(x, y):
            x = np.asarray(x, dtype=np.float64)
            y = np.asarray(y, dtype=np.float64)

            mask = np.isfinite(x) & np.isfinite(y)
            x = x[mask]
            y = y[mask]
            if x.size < 2:
                return 0.0

            xm = x - np.mean(x)
            ym = y - np.mean(y)

            den = np.sqrt(np.sum(xm ** 2) * np.sum(ym ** 2))
            if den <= 0:
                return 0.0

            r = float(np.sum(xm * ym) / den)
            
            return max(min(r, 1.0), -1.0)

        p = float(BCEA_probability)
        p = min(max(p, 1e-12), 1.0 - 1e-12)  # évite log(0)
        k = -np.log(1.0 - p)

        x_a = np.asarray(self.data_set["x_array"], dtype=np.float64)
        y_a = np.asarray(self.data_set["y_array"], dtype=np.float64)

        bcea_s = []
        for start, end in self._intervals():
            l_x = x_a[start : end + 1]
            l_y = y_a[start : end + 1]

            r = pearson_corr_(l_x, l_y)
            sd_x = float(np.nanstd(l_x, ddof=1)) if np.sum(np.isfinite(l_x)) >= 2 else 0.0
            sd_y = float(np.nanstd(l_y, ddof=1)) if np.sum(np.isfinite(l_y)) >= 2 else 0.0

            inside = max(0.0, 1.0 - r ** 2)
            l_bcea = 2.0 * np.pi * k * sd_x * sd_y * np.sqrt(inside)
            bcea_s.append(l_bcea)

        bcea_s = np.asarray(bcea_s, dtype=np.float64)
        results = {
            "average_BCEA": float(np.nanmean(bcea_s)),
            "raw": bcea_s,
        }
        
        if not get_raw:
            del results["raw"]
            
        return results


## Some access functions
def fixation_count(input, **kwargs):
    if isinstance(input, FixationAnalysis):
        results = input.fixation_count()
        input.verbose()

    else:
        fixation_analysis = FixationAnalysis.generate(input, **kwargs)
        results = fixation_analysis.fixation_count()
        fixation_analysis.verbose()

    return results


def fixation_frequency(input, **kwargs):
    if isinstance(input, FixationAnalysis):
        results = input.fixation_frequency()
        input.verbose()

    else:
        fixation_analysis = FixationAnalysis.generate(input, **kwargs)
        results = fixation_analysis.fixation_frequency()
        fixation_analysis.verbose()

    return results


def fixation_frequency_wrt_labels(input, **kwargs):
    if isinstance(input, FixationAnalysis):
        results = input.fixation_frequency_wrt_labels()
        input.verbose()

    else:
        fixation_analysis = FixationAnalysis.generate(input, **kwargs)
        results = fixation_analysis.fixation_frequency_wrt_labels()
        fixation_analysis.verbose()

    return results


def fixation_durations(input, **kwargs):
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, FixationAnalysis):
        results = input.fixation_durations(get_raw)
        input.verbose()

    else:
        fixation_analysis = FixationAnalysis.generate(input, **kwargs)
        results = fixation_analysis.fixation_durations(get_raw)
        fixation_analysis.verbose(dict({"get_raw": get_raw}))

    return results


def fixation_centroids(input, **kwargs):
    if isinstance(input, FixationAnalysis):
        results = input.fixation_centroids()
        input.verbose()

    else:
        fixation_analysis = FixationAnalysis.generate(input, **kwargs)
        results = fixation_analysis.fixation_centroids()
        fixation_analysis.verbose()

    return results


def fixation_mean_velocities(input, **kwargs):
    if isinstance(input, FixationAnalysis):
        results = input.fixation_mean_velocities()
        input.verbose()

    else:
        fixation_analysis = FixationAnalysis.generate(input, **kwargs)
        results = fixation_analysis.fixation_mean_velocities()
        fixation_analysis.verbose()

    return results


def fixation_average_velocity_means(input, **kwargs):
    weighted = kwargs.get("fixation_weighted_average_velocity_means", False)
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, FixationAnalysis):
        results = input.fixation_average_velocity_means(weighted, get_raw)
        input.verbose(
            dict(
                {
                    "get_raw": get_raw,
                    "fixation_weighted_average_velocity_means": weighted,
                }
            )
        )

    else:
        fixation_analysis = FixationAnalysis.generate(input, **kwargs)
        results = fixation_analysis.fixation_average_velocity_means(weighted, get_raw)
        fixation_analysis.verbose(
            dict(
                {
                    "get_raw": get_raw,
                    "fixation_weighted_average_velocity_means": weighted,
                }
            )
        )

    return results


def fixation_average_velocity_deviations(input, **kwargs):
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, FixationAnalysis):
        results = input.fixation_average_velocity_deviations(get_raw)
        input.verbose(dict({"get_raw": get_raw}))

    else:
        fixation_analysis = FixationAnalysis.generate(input, **kwargs)
        results = fixation_analysis.fixation_average_velocity_deviations(get_raw)
        fixation_analysis.verbose(dict({"get_raw": get_raw}))

    return results


def fixation_drift_displacements(input, **kwargs):
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, FixationAnalysis):
        results = input.fixation_drift_displacements(get_raw)
        input.verbose(dict({"get_raw": get_raw}))

    else:
        fixation_analysis = FixationAnalysis.generate(input, **kwargs)
        results = fixation_analysis.fixation_drift_displacements(get_raw)
        fixation_analysis.verbose(dict({"get_raw": get_raw}))

    return results


def fixation_drift_distances(input, **kwargs):
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, FixationAnalysis):
        results = input.fixation_drift_distances(get_raw)
        input.verbose(dict({"get_raw": get_raw}))

    else:
        fixation_analysis = FixationAnalysis.generate(input, **kwargs)
        results = fixation_analysis.fixation_drift_distances(get_raw)
        fixation_analysis.verbose(dict({"get_raw": get_raw}))

    return results


def fixation_drift_velocities(input, **kwargs):
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, FixationAnalysis):
        results = input.fixation_drift_velocities(get_raw)
        input.verbose(dict({"get_raw": get_raw}))

    else:
        fixation_analysis = FixationAnalysis.generate(input, **kwargs)
        results = fixation_analysis.fixation_drift_velocities(get_raw)
        fixation_analysis.verbose(dict({"get_raw": get_raw}))

    return results


def fixation_BCEA(input, **kwargs):
    BCEA_probability = kwargs.get("fixation_BCEA_probability", 0.68)
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, FixationAnalysis):
        results = input.fixation_BCEA(BCEA_probability, get_raw)
        input.verbose(
            dict({"get_raw": get_raw, "fixation_BCEA_probability": BCEA_probability})
        )

    else:
        fixation_analysis = FixationAnalysis.generate(input, **kwargs)
        results = fixation_analysis.fixation_BCEA(BCEA_probability, get_raw)
        fixation_analysis.verbose(
            dict({"get_raw": get_raw, "fixation_BCEA_probability": BCEA_probability})
        )

    return results
