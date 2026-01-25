# -*- coding: utf-8 -*-
 
import numpy as np

from vision_toolkit.segmentation.processing.ternary_segmentation import TernarySegmentation


class PursuitAnalysis(TernarySegmentation):
    """
    For a pursuit [start,end]:
        * positions:  start .. end        (n_samples = end-start+1)
        * speeds:     start .. end-1      (n_vel = n_samples-1) => slice a_sp[start:end]
    """

    def __init__(self, input, **kwargs):
        verbose = kwargs.get("verbose", True)

        if verbose:
            print("Processing Pursuit Analysis...\n")

        # Reuse an already computed segmentation
        if isinstance(input, TernarySegmentation):
            # (PursuitAnalysis is also a TernarySegmentation, so this covers both)
            self.__dict__ = input.__dict__.copy()
            self.config.update({"verbose": verbose})
        else:
            sampling_frequency = kwargs.get("sampling_frequency", None)
            assert sampling_frequency is not None, "Sampling frequency must be specified"
            super().__init__(input, **kwargs)

        self.s_f = float(self.config["sampling_frequency"])
        assert (
            len(self.segmentation_results.get("pursuit_intervals", [])) > 0
        ), "No pursuit identified"

        if verbose:
            print("...Pursuit Analysis done\n")

 
    # @classmethod
    # def generate(cls, input, **kwargs):
        
    #     if isinstance(input, PursuitAnalysis):
    #         return input
    #     return cls(input, **kwargs)

        
    def _intervals(self):
        return self.segmentation_results["pursuit_intervals"]

    def _n_samples_per_interval(self, intervals):
        a_i = np.asarray(intervals, dtype=np.int64)
        return (a_i[:, 1] - a_i[:, 0] + 1).astype(np.float64)

    def _speed_array(self):
        return np.asarray(self.data_set["absolute_speed"], dtype=np.float64)

    def _speed_segment(self, start, end):
        # speeds are defined on transitions => start..end-1
        if end <= start:
            return np.array([], dtype=np.float64)
        a_sp = self._speed_array()
        return a_sp[start:end]

    def _safe_sd(self, x):
        x = np.asarray(x, dtype=np.float64)
        return float(np.nanstd(x, ddof=1)) if np.sum(np.isfinite(x)) >= 2 else 0.0

 
    def pursuit_count(self):
        return {"count": int(len(self._intervals()))}


    def pursuit_frequency(self):
        ct = len(self._intervals())
        denom = (self.config["nb_samples"] / self.s_f)
        f = ct / denom if denom > 0 else np.nan
        
        return {"frequency": float(f)}


    def pursuit_durations(self, get_raw=True):
        
        a_i = np.asarray(self._intervals(), dtype=np.int64)
        a_d = (a_i[:, 1] - a_i[:, 0] + 1) / self.s_f   

        results = {
            "duration_mean": float(np.nanmean(a_d)),
            "duration_sd": self._safe_sd(a_d),
            "raw": a_d,
        }
        if not get_raw:
            del results["raw"]
        return results


    def pursuit_proportion(self):
        # proportion of samples labeled as pursuit
        a_i = np.asarray(self._intervals(), dtype=np.int64)
        if a_i.size == 0:
            return 0.0
        n_p = float(np.sum(a_i[:, 1] - a_i[:, 0] + 1))
        denom = float(self.config["nb_samples"])
        
        return float(n_p / denom) if denom > 0 else np.nan


    def pursuit_velocity(self, get_raw=True):
        
        # pooled speeds across all pursuit segments (start..end-1)
        l_sp = []
        for start, end in self._intervals():
            seg = self._speed_segment(start, end)
            if seg.size:
                l_sp.append(seg)

        all_sp = np.concatenate(l_sp) if len(l_sp) else np.array([], dtype=np.float64)

        results = {
            "velocity_mean": float(np.nanmean(all_sp)) if all_sp.size else np.nan,
            "velocity_sd": self._safe_sd(all_sp) if all_sp.size else np.nan,
            "raw": all_sp,
        }
        if not get_raw:
            del results["raw"]
        return results


    def pursuit_velocity_means(self, get_raw=True):
        # mean speed per pursuit interval (start..end-1)
        m_sp = []
        for start, end in self._intervals():
            seg = self._speed_segment(start, end)
            m_sp.append(float(np.nanmean(seg)) if seg.size else np.nan)

        m_sp = np.asarray(m_sp, dtype=np.float64)

        results = {
            "velocity_mean_mean": float(np.nanmean(m_sp)),
            "velocity_mean_sd": self._safe_sd(m_sp),
            "raw": m_sp,
        }
        if not get_raw:
            del results["raw"]
        return results


    def pursuit_peak_velocity(self, get_raw=True):
        # peak speed per pursuit interval (start..end-1)
        p_sp = []
        for start, end in self._intervals():
            seg = self._speed_segment(start, end)
            p_sp.append(float(np.nanmax(seg)) if seg.size else np.nan)

        p_sp = np.asarray(p_sp, dtype=np.float64)

        results = {
            "velocity_peak_mean": float(np.nanmean(p_sp)),
            "velocity_peak_sd": self._safe_sd(p_sp),
            "raw": p_sp,
        }
        if not get_raw:
            del results["raw"]
        return results


    def pursuit_amplitude(self, get_raw=True):
        x_a = self.data_set["x_array"]
        y_a = self.data_set["y_array"]
        z_a = self.data_set["z_array"]

        dist_ = self.distances[self.config["distance_type"]]

        dsp = []
        for start, end in self._intervals():
            s_p = np.array([x_a[start], y_a[start], z_a[start]])
            e_p = np.array([x_a[end], y_a[end], z_a[end]])
            dsp.append(dist_(s_p, e_p))

        dsp = np.asarray(dsp, dtype=np.float64)

        results = {
            "pursuit_amplitude_mean": float(np.nanmean(dsp)),
            "pursuit_amplitude_sd": self._safe_sd(dsp),
            "raw": dsp,
        }
        if not get_raw:
            del results["raw"]
        return results


    def pursuit_distance(self, get_raw=True):
        x_a = self.data_set["x_array"]
        y_a = self.data_set["y_array"]
        z_a = self.data_set["z_array"]

        dist_ = self.distances[self.config["distance_type"]]

        t_cum = []
        for start, end in self._intervals():
            if end <= start:
                t_cum.append(np.nan)
                continue

            l_cum = 0.0
            for k in range(start, end):
                s_p = np.array([x_a[k], y_a[k], z_a[k]])
                e_p = np.array([x_a[k + 1], y_a[k + 1], z_a[k + 1]])
                l_cum += dist_(s_p, e_p)

            t_cum.append(l_cum)

        t_cum = np.asarray(t_cum, dtype=np.float64)

        results = {
            "pursuit_cumul_distance_mean": float(np.nanmean(t_cum)),
            "pursuit_cumul_distance_sd": self._safe_sd(t_cum),
            "raw": t_cum,
        }
        if not get_raw:
            del results["raw"]
        return results


    def pursuit_efficiency(self, get_raw=True):
        # efficiency = straight amplitude / cumulative distance
        x_a = self.data_set["x_array"]
        y_a = self.data_set["y_array"]
        z_a = self.data_set["z_array"]

        dist_ = self.distances[self.config["distance_type"]]

        eff = []
        for start, end in self._intervals():
            if end <= start:
                eff.append(np.nan)
                continue

            s_p = np.array([x_a[start], y_a[start], z_a[start]])
            e_p = np.array([x_a[end], y_a[end], z_a[end]])
            amp = dist_(s_p, e_p)

            l_cum = 0.0
            for k in range(start, end):
                p0 = np.array([x_a[k], y_a[k], z_a[k]])
                p1 = np.array([x_a[k + 1], y_a[k + 1], z_a[k + 1]])
                l_cum += dist_(p0, p1)

            eff.append(amp / l_cum if l_cum > 0 else np.nan)

        eff = np.asarray(eff, dtype=np.float64)

        results = {
            "pursuit_efficiency_mean": float(np.nanmean(eff)),
            "pursuit_efficiency_sd": self._safe_sd(eff),
            "raw": eff,
        }
        if not get_raw:
            del results["raw"]
        return results

   

        
        
def pursuit_count(input, **kwargs):
    if isinstance(input, PursuitAnalysis):
        results = input.pursuit_count()
        input.verbose()
    else:
        pursuit_analysis = PursuitAnalysis.generate(input, **kwargs)
        results = pursuit_analysis.pursuit_count()
        pursuit_analysis.verbose()
    return results

 
def pursuit_frequency(input, **kwargs):
    if isinstance(input, PursuitAnalysis):
        results = input.pursuit_frequency()
        input.verbose()
    else:
        pursuit_analysis = PursuitAnalysis.generate(input, **kwargs)
        results = pursuit_analysis.pursuit_frequency()
        pursuit_analysis.verbose()
    return results


def pursuit_durations(input, **kwargs):
    get_raw = kwargs.get("get_raw", True)
    
    if isinstance(input, PursuitAnalysis):
        results = input.pursuit_durations(get_raw)
        input.verbose(dict({"get_raw": get_raw}))
    else:
        pursuit_analysis = PursuitAnalysis.generate(input, **kwargs)
        results = pursuit_analysis.pursuit_durations(get_raw)
        pursuit_analysis.verbose(dict({"get_raw": get_raw}))
    return results


def pursuit_proportion(input, **kwargs):
    if isinstance(input, PursuitAnalysis):
        results = input.pursuit_proportion()
        input.verbose()
    else:
        pursuit_analysis = PursuitAnalysis.generate(input, **kwargs)
        results = pursuit_analysis.pursuit_proportion()
        pursuit_analysis.verbose()
    return results


def pursuit_velocity(input, **kwargs):
    get_raw = kwargs.get("get_raw", True)
    
    if isinstance(input, PursuitAnalysis):
        results = input.pursuit_velocity(get_raw)
        input.verbose(dict({"get_raw": get_raw}))
    else:
        pursuit_analysis = PursuitAnalysis.generate(input, **kwargs)
        results = pursuit_analysis.pursuit_velocity(get_raw)
        pursuit_analysis.verbose(dict({"get_raw": get_raw}))
    return results



def pursuit_velocity_means(input, **kwargs):
    get_raw = kwargs.get("get_raw", True)
    
    if isinstance(input, PursuitAnalysis):
        results = input.pursuit_velocity_means(get_raw)
        input.verbose(dict({"get_raw": get_raw}))
    else:
        pursuit_analysis = PursuitAnalysis.generate(input, **kwargs)
        results = pursuit_analysis.pursuit_velocity_means(get_raw)
        pursuit_analysis.verbose(dict({"get_raw": get_raw}))
    return results



def pursuit_peak_velocity(input, **kwargs):
    get_raw = kwargs.get("get_raw", True)
    
    if isinstance(input, PursuitAnalysis):
        results = input.pursuit_peak_velocity(get_raw)
        input.verbose(dict({"get_raw": get_raw}))
    else:
        pursuit_analysis = PursuitAnalysis.generate(input, **kwargs)
        results = pursuit_analysis.pursuit_peak_velocity(get_raw)
        pursuit_analysis.verbose(dict({"get_raw": get_raw}))
    return results


def pursuit_amplitude(input, **kwargs):
    get_raw = kwargs.get("get_raw", True)
    
    if isinstance(input, PursuitAnalysis):
        results = input.pursuit_amplitude(get_raw)
        input.verbose(dict({"get_raw": get_raw}))
    else:
        pursuit_analysis = PursuitAnalysis.generate(input, **kwargs)
        results = pursuit_analysis.pursuit_amplitude(get_raw)
        pursuit_analysis.verbose(dict({"get_raw": get_raw}))
    return results


def pursuit_distance(input, **kwargs):
    get_raw = kwargs.get("get_raw", True)
    
    if isinstance(input, PursuitAnalysis):
        results = input.pursuit_distance(get_raw)
        input.verbose(dict({"get_raw": get_raw}))
    else:
        pursuit_analysis = PursuitAnalysis.generate(input, **kwargs)
        results = pursuit_analysis.pursuit_distance(get_raw)
        pursuit_analysis.verbose(dict({"get_raw": get_raw}))
    return results


def pursuit_efficiency(input, **kwargs):
    get_raw = kwargs.get("get_raw", True)
    
    if isinstance(input, PursuitAnalysis):
        results = input.pursuit_efficiency(get_raw)
        input.verbose(dict({"get_raw": get_raw}))
    else:
        pursuit_analysis = PursuitAnalysis.generate(input, **kwargs)
        results = pursuit_analysis.pursuit_efficiency(get_raw)
        pursuit_analysis.verbose(dict({"get_raw": get_raw}))
    return results

 

 