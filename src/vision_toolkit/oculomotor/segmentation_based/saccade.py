 
from vision_toolkit.segmentation.processing.binary_segmentation import BinarySegmentation
from vision_toolkit.visualization.oculomotor.main_sequence import plot_main_sequence


import numpy as np
from numpy.linalg import norm
from scipy.stats import gamma


class SaccadeAnalysis(BinarySegmentation):
    """

    For a saccade [start,end]:
        * positions:      start .. end        (n_samples = end-start+1)
        * speeds:       start .. end-1      (n_vel = n_samples-1)  => slice a_sp[start:end]
        * accélérations:  start .. end-2      (n_acc = n_samples-2)  => diff(vitesse)*sf sur a_sp[start:end]
    """

    def __init__(self, input, **kwargs):
        
        verbose = kwargs.get("verbose", True)

        if verbose:
            print("Processing Saccade Analysis...\n")

        if isinstance(input, BinarySegmentation):
            self.__dict__ = input.__dict__.copy()
            self.config.update({"verbose": verbose})
        else:
            sampling_frequency = kwargs.get("sampling_frequency", None)
            assert sampling_frequency is not None, "Sampling frequency must be specified"
            super().__init__(input, **kwargs)

        self.s_f = float(self.config["sampling_frequency"])
        assert len(self.segmentation_results["saccade_intervals"]) > 0, "No saccade identified"

        if verbose:
            print("...Saccade Analysis done\n")

 

    def _intervals(self):
        return self.segmentation_results["saccade_intervals"]

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

    def _acc_segment(self, start, end):
       
        v = self._speed_segment(start, end)
        if v.size < 2:
            return np.array([], dtype=np.float64)
        return np.abs(np.diff(v)) * self.s_f

    def _safe_sd(self, x):
        x = np.asarray(x, dtype=np.float64)
        return float(np.nanstd(x, ddof=1)) if np.sum(np.isfinite(x)) >= 2 else 0.0


    def saccade_count(self):
        return {"count": int(len(self._intervals()))}

    def saccade_frequency(self):
        
        ct = len(self._intervals())
        denom = (self.config["nb_samples"] / self.s_f)
        f = ct / denom if denom > 0 else np.nan
        
        return {"frequency": float(f)}


    def saccade_frequency_wrt_labels(self):
        
        ct = len(self._intervals())
        labeled = float(np.sum(self.segmentation_results["is_labeled"]))
        denom = labeled / self.s_f
        
        f = ct / denom if denom > 0 else np.nan
        return {"frequency": float(f)}


    def saccade_durations(self, get_raw=True):
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


    def saccade_amplitudes(self, get_raw=True):
        
        x_a = self.data_set["x_array"]
        y_a = self.data_set["y_array"]
        z_a = self.data_set["z_array"]
        dist_ = self.distances[self.config["distance_type"]]

        s_a = []
        for start, end in self._intervals():
            s_a.append(
                dist_(
                    np.array([x_a[start], y_a[start], z_a[start]]),
                    np.array([x_a[end], y_a[end], z_a[end]]),
                )
            )

        s_a = np.asarray(s_a, dtype=np.float64)
        results = {
            "amplitude_mean": float(np.nanmean(s_a)),
            "amplitude_sd": self._safe_sd(s_a),
            "raw": np.round(s_a, 3),
        }
        
        if not get_raw:
            del results["raw"]
            
        return results


    def saccade_travel_distances(self, get_raw=True):
        
        x_a = self.data_set["x_array"]
        y_a = self.data_set["y_array"]
        z_a = self.data_set["z_array"]
        dist_ = self.distances[self.config["distance_type"]]

        d_cum = []
        for start, end in self._intervals():
            if end <= start:
                d_cum.append(np.nan)
                continue
            d = np.sum(
                np.array(
                    [
                        dist_(
                            np.array([x_a[k], y_a[k], z_a[k]]),
                            np.array([x_a[k + 1], y_a[k + 1], z_a[k + 1]]),
                        )
                        for k in range(start, end)
                    ],
                    dtype=np.float64,
                )
            )
            d_cum.append(d)

        d_cum = np.asarray(d_cum, dtype=np.float64)
        results = {
            "distance_mean": float(np.nanmean(d_cum)),
            "distance_sd": self._safe_sd(d_cum),
            "raw": d_cum,
        }
        if not get_raw:
            del results["raw"]
            
        return results


    def saccade_efficiencies(self, get_raw=True):
        
        s_a = self.saccade_amplitudes(get_raw=True)["raw"]
        d_cum = self.saccade_travel_distances(get_raw=True)["raw"]

        eff = np.full_like(s_a, np.nan, dtype=np.float64)
        mask = np.isfinite(s_a) & np.isfinite(d_cum) & (d_cum > 0)
        eff[mask] = s_a[mask] / d_cum[mask]

        results = {
            "efficiency_mean": float(np.nanmean(eff)),
            "efficiency_sd": self._safe_sd(eff),
            "raw": eff,
        }
        if not get_raw:
            del results["raw"]
            
        return results

    
    def comp_dir(self, v_i):
        v_i = np.asarray(v_i, dtype=np.float64) + 1e-10

        neg = v_i[:, 1] < 0
        pos = ~neg
        dir_ = np.zeros(len(v_i), dtype=np.float64)

        n_pos = np.linalg.norm(v_i[pos], axis=1)
        cos_pos = np.divide(v_i[pos, 0], n_pos, out=np.zeros_like(n_pos), where=n_pos > 0)
        cos_pos = np.clip(cos_pos, -1.0, 1.0)
        dir_[pos] = (180.0 / np.pi) * np.arccos(cos_pos)

        n_neg = np.linalg.norm(v_i[neg], axis=1)
        cos_neg = np.divide(v_i[neg, 0], n_neg, out=np.zeros_like(n_neg), where=n_neg > 0)
        cos_neg = np.clip(cos_neg, -1.0, 1.0)
        dir_[neg] = (180.0 / np.pi) * (2.0 * np.pi - np.arccos(cos_neg))

        return dir_


    def saccade_directions(self, get_raw=True):
        
        x_a = np.asarray(self.data_set["x_array"], dtype=np.float64)
        y_a = np.asarray(self.data_set["y_array"], dtype=np.float64)

        v_i = []
        for start, end in self._intervals():
            v_i.append(np.array([x_a[end] - x_a[start], y_a[end] - y_a[start]], dtype=np.float64))

        dir_ = self.comp_dir(np.asarray(v_i, dtype=np.float64))
        results = {
            "direction_mean": float(np.nanmean(dir_)),
            "direction_sd": self._safe_sd(dir_),
            "raw": dir_,
        }
        
        if not get_raw:
            del results["raw"]
            
        return results


    def saccade_horizontal_deviations(self, absolute=True, get_raw=True):
        
        x_a = np.asarray(self.data_set["x_array"], dtype=np.float64)
        y_a = np.asarray(self.data_set["y_array"], dtype=np.float64)

        devs = []
        for start, end in self._intervals():
            v = np.array([x_a[end] - x_a[start], y_a[end] - y_a[start]], dtype=np.float64)
            nv = np.linalg.norm(v)
            if nv <= 0:
                devs.append(np.nan)
                continue
            cosang = np.dot(v / nv, np.array([1.0, 0.0]))
            cosang = np.clip(cosang, -1.0, 1.0)
            dev = (180.0 / np.pi) * np.arccos(cosang)
            if absolute and dev > 90:
                dev = 180 - dev
            devs.append(dev)

        devs = np.asarray(devs, dtype=np.float64)
        results = {
            "horizontal_deviation_mean": float(np.nanmean(devs)),
            "horizontal_deviation_sd": self._safe_sd(devs),
            "raw": devs,
        }
        
        if not get_raw:
            del results["raw"]
        return results
    

    def saccade_successive_deviations(self, get_raw=True):
        
        x_a = np.asarray(self.data_set["x_array"], dtype=np.float64)
        y_a = np.asarray(self.data_set["y_array"], dtype=np.float64)
        _ints = self._intervals()

        devs = []
        for i in range(1, len(_ints)):
            b0, b1 = _ints[i - 1]
            a0, a1 = _ints[i]

            v_b = np.array([x_a[b1] - x_a[b0], y_a[b1] - y_a[b0]], dtype=np.float64)
            v_a = np.array([x_a[a1] - x_a[a0], y_a[a1] - y_a[a0]], dtype=np.float64)

            nb = np.linalg.norm(v_b)
            na = np.linalg.norm(v_a)
            if nb <= 0 or na <= 0:
                devs.append(np.nan)
                continue

            cosang = np.dot(v_b / nb, v_a / na)
            cosang = np.clip(cosang, -1.0, 1.0)
            devs.append((180.0 / np.pi) * np.arccos(cosang))

        devs = np.asarray(devs, dtype=np.float64)
        results = {
            "successive_deviation_mean": float(np.nanmean(devs)),
            "successive_deviation_sd": self._safe_sd(devs),
            "raw": devs,
        }
        if not get_raw:
            del results["raw"]
            
        return results


    def saccade_initial_directions(self, duration_threshold=0.020, get_raw=True):
        
        t_du = int(duration_threshold * self.s_f) + 1
        x_a = np.asarray(self.data_set["x_array"], dtype=np.float64)
        y_a = np.asarray(self.data_set["y_array"], dtype=np.float64)

        v_i = []
        for start, end in self._intervals():
            # nombre de samples dans l'intervalle = end-start+1 => déplacement initial max = end-start
            t_s = min(t_du, max(end - start, 0))
            v_i.append(np.array([x_a[start + t_s] - x_a[start], y_a[start + t_s] - y_a[start]], dtype=np.float64))

        dir_ = self.comp_dir(np.asarray(v_i, dtype=np.float64))
        results = {
            "initial_direction_mean": float(np.nanmean(dir_)),
            "initial_direction_sd": self._safe_sd(dir_),
            "raw": dir_,
        }
        
        if not get_raw:
            del results["raw"]
            
        return results


    def saccade_initial_deviations(self, duration_threshold=0.020, get_raw=True):
        
        t_du = int(duration_threshold * self.s_f) + 1
        x_a = np.asarray(self.data_set["x_array"], dtype=np.float64)
        y_a = np.asarray(self.data_set["y_array"], dtype=np.float64)

        devs = []
        for start, end in self._intervals():
            t_s = min(t_du, max(end - start, 0))

            v_i = np.array([x_a[start + t_s] - x_a[start], y_a[start + t_s] - y_a[start]], dtype=np.float64)
            v_t = np.array([x_a[end] - x_a[start], y_a[end] - y_a[start]], dtype=np.float64)

            ni = np.linalg.norm(v_i)
            nt = np.linalg.norm(v_t)
            if ni <= 0 or nt <= 0:
                devs.append(np.nan)
                continue

            cosang = np.dot(v_i / ni, v_t / nt)
            cosang = np.clip(cosang, -1.0, 1.0)
            devs.append((180.0 / np.pi) * np.arccos(cosang))

        devs = np.asarray(devs, dtype=np.float64)
        results = {
            "initial_deviation_mean": float(np.nanmean(devs)),
            "initial_deviation_sd": self._safe_sd(devs),
            "raw": devs,
        }
        
        if not get_raw:
            del results["raw"]
            
        return results


    def shortest_distance(self, p_i, p_b=None, p_e=None):
        
        if p_b is None:
            p_b = p_i[0]
        if p_e is None:
            p_e = p_i[-1]

        denom = np.linalg.norm(p_e - p_b)
        if denom <= 0:
            return np.zeros(len(p_i), dtype=np.float64)

        h_e = (p_e - p_b) / denom
        d_ = p_i - p_b
        n_ = np.linalg.norm(d_, axis=1)
        h_v = np.divide(d_.T, n_, where=n_ > 0).T

        alpha_i = (h_v @ h_e.reshape(2, 1)).T[0]
        alpha_i = np.clip(alpha_i, -1.0, 1.0)

        alpha_i = np.arccos(alpha_i)
        
        return np.sin(alpha_i) * np.linalg.norm(d_, axis=1)


    def linear_distance(self, p_i, p_b=None, p_e=None):
        
        if p_b is None:
            p_b = p_i[0]
        if p_e is None:
            p_e = p_i[-1]

        denom = np.linalg.norm(p_e - p_b)
        if denom <= 0:
            return np.zeros(len(p_i), dtype=np.float64)

        h_e = (p_e - p_b) / denom
        d_ = p_i - p_b
        n_ = np.linalg.norm(d_, axis=1)
        h_v = np.divide(d_.T, n_, where=n_ > 0).T

        alpha_i = (h_v @ h_e.reshape(2, 1)).T[0]
        alpha_i = np.clip(alpha_i, -1.0, 1.0)

        alpha_i = np.arccos(alpha_i)
        h_d_ = np.cos(alpha_i) * np.linalg.norm(d_, axis=1)

        h_d_d = np.zeros(len(p_i), dtype=np.float64)
        h_d_d[1:] = h_d_[1:] - h_d_[:-1]
        
        return h_d_d


    def saccade_max_curvatures(self, get_raw=True):
        
        x_a = np.asarray(self.data_set["x_array"], dtype=np.float64)
        y_a = np.asarray(self.data_set["y_array"], dtype=np.float64)

        m_curv = []
        for start, end in self._intervals():
            p_i = np.stack([x_a[start:end + 1], y_a[start:end + 1]], axis=1)
            p_d = self.shortest_distance(p_i)
            m_curv.append(float(np.nanmax(p_d)) if p_d.size else np.nan)

        m_curv = np.asarray(m_curv, dtype=np.float64)
        results = {
            "max_curvature_mean": float(np.nanmean(m_curv)),
            "max_curvature_sd": self._safe_sd(m_curv),
            "raw": m_curv,
        }
        
        if not get_raw:
            del results["raw"]
            
        return results


    def saccade_area_curvatures(self, get_raw=True):
        
        x_a = np.asarray(self.data_set["x_array"], dtype=np.float64)
        y_a = np.asarray(self.data_set["y_array"], dtype=np.float64)

        area = []
        for start, end in self._intervals():
            p_i = np.stack([x_a[start:end + 1], y_a[start:end + 1]], axis=1)
            p_d = self.shortest_distance(p_i)
            l_d = self.linear_distance(p_i)
            area.append(float(np.nansum(p_d * l_d)) if p_d.size else np.nan)

        area = np.asarray(area, dtype=np.float64)
        results = {
            "curvature_area_mean": float(np.nanmean(area)),
            "curvature_area_sd": self._safe_sd(area),
            "raw": area,
        }
       
        if not get_raw:
            del results["raw"]
            
        return results

  
    def saccade_mean_velocities(self):
        
        m_sp, sd_sp = [], []
        for start, end in self._intervals():
            seg = self._speed_segment(start, end)  # start..end-1
            m_sp.append(float(np.nanmean(seg)) if seg.size else np.nan)
            sd_sp.append(self._safe_sd(seg) if seg.size else np.nan)

        return {"velocity_means": np.asarray(m_sp, dtype=np.float64),
                "velocity_sd": np.asarray(sd_sp, dtype=np.float64)}


    def saccade_average_velocity_means(self, weighted=False, get_raw=True, weight_mode="diffs"):
       
        m_sp = self.saccade_mean_velocities()["velocity_means"]

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


    def saccade_average_velocity_deviations(self, get_raw=True, weight_mode="diffs"):
        
        sd_sp = self.saccade_mean_velocities()["velocity_sd"]

        n_samples = self._n_samples_per_interval(self._intervals())
        if weight_mode == "samples":
            w = np.maximum(n_samples, 0.0)
        else:  # diffs
            w = np.maximum(n_samples - 1.0, 0.0)

        denom = np.nansum(w)
        a_sd = float(np.sqrt(np.nansum(w * (sd_sp ** 2)) / denom)) if denom > 0 else np.nan

        results = {"average_velocity_sd": a_sd, "raw": sd_sp}
        
        if not get_raw:
            del results["raw"]
            
        return results

    def saccade_peak_velocities(self, get_raw=True):
        
        p_sp = []
        for start, end in self._intervals():
            seg = self._speed_segment(start, end)  # start..end-1 (PAS end)
            p_sp.append(float(np.nanmax(seg)) if np.any(np.isfinite(seg)) else np.nan)

        p_sp = np.asarray(p_sp, dtype=np.float64)
        results = {
            "velocity_peak_mean": float(np.nanmean(p_sp)),
            "velocity_peak_sd": self._safe_sd(p_sp),
            "raw": p_sp,
        }
       
        if not get_raw:
            del results["raw"]
            
        return results

    def get_pk_vel_idx(self):
      
        idxs = []
        for start, end in self._intervals():
            seg = self._speed_segment(start, end)   
            if seg.size == 0 or not np.any(np.isfinite(seg)):
                idxs.append(int(start))
            else:
                idxs.append(int(start + np.nanargmax(seg)))  # <= end-1
        return np.asarray(idxs, dtype=np.int64)


    def saccade_mean_acceleration_profiles(self):
        
        m_ac, sd_ac = [], []
        for start, end in self._intervals():
            acc = self._acc_segment(start, end)  # indices start..end-2
            if acc.size == 0 or not np.any(np.isfinite(acc)):
                m_ac.append(np.nan)
                sd_ac.append(np.nan)
            else:
                m_ac.append(float(np.nanmean(acc)))
                sd_ac.append(self._safe_sd(acc))
        
        return {"acceleration_profile_means": np.asarray(m_ac, dtype=np.float64),
                "acceleration_profile_sd": np.asarray(sd_ac, dtype=np.float64)}


    def saccade_mean_accelerations(self):
        
        pk = self.get_pk_vel_idx()
        a_sp = self._speed_array()

        m_ac, sd_ac = [], []
        for i, (start, end) in enumerate(self._intervals()):
            pk_i = int(min(pk[i], end - 1))
            v_pre = a_sp[start:pk_i + 1]  # vitesses du début jusqu'au pic inclus
            if v_pre.size < 2:
                m_ac.append(np.nan)
                sd_ac.append(np.nan)
                continue
            
            acc_pre = np.abs(np.diff(v_pre)) * self.s_f
            m_ac.append(float(np.nanmean(acc_pre)))
            sd_ac.append(self._safe_sd(acc_pre))

        return {"acceleration_means": np.asarray(m_ac, dtype=np.float64),
                "acceleration_sd": np.asarray(sd_ac, dtype=np.float64)}


    def saccade_mean_decelerations(self):
        
        pk = self.get_pk_vel_idx()
        a_sp = self._speed_array()

        m_dc, sd_dc = [], []
        for i, (start, end) in enumerate(self._intervals()):
            pk_i = int(min(pk[i], end - 1))
            v_post = a_sp[pk_i:end]  # vitesses du pic jusqu'à end-1
            if v_post.size < 2:
                m_dc.append(np.nan)
                sd_dc.append(np.nan)
                continue
            
            acc_post = np.abs(np.diff(v_post)) * self.s_f
            m_dc.append(float(np.nanmean(acc_post)))
            sd_dc.append(self._safe_sd(acc_post))

        return {"deceleration_means": np.asarray(m_dc, dtype=np.float64),
                "deceleration_sd": np.asarray(sd_dc, dtype=np.float64)}


    def acc_average(self, data, weighted, get_raw):
        
        data = np.asarray(data, dtype=np.float64)

        if not weighted:
            data_s = data[~np.isnan(data)]
            results = {"average_means": float(np.nanmean(data_s)) if data_s.size else np.nan, "raw": data}
            if not get_raw:
                del results["raw"]
                
            return results

        # pondération par le nb d'accélérations valides = n_samples - 2
        n_samples = self._n_samples_per_interval(self._intervals())
        w = np.maximum(n_samples - 2.0, 0.0)

        tmp = w * data
        mask = np.isfinite(tmp) & np.isfinite(w)
        denom = np.nansum(w[mask])
        w_v = float(np.nansum(tmp[mask]) / denom) if denom > 0 else np.nan

        results = {"weighted_average_means": w_v, "raw": data}
        
        if not get_raw:
            del results["raw"]
            
        return results


    def saccade_average_acceleration_profiles(self, weighted=False, get_raw=True):
        
        m_ac = self.saccade_mean_acceleration_profiles()["acceleration_profile_means"]
        return self.acc_average(m_ac, weighted, get_raw)


    def saccade_average_acceleration_means(self, weighted=False, get_raw=True):
        
        m_ac = self.saccade_mean_accelerations()["acceleration_means"]
        return self.acc_average(m_ac, weighted, get_raw)


    def saccade_average_deceleration_means(self, weighted=False, get_raw=True):
        m_dc = self.saccade_mean_decelerations()["deceleration_means"]
        return self.acc_average(m_dc, weighted, get_raw)


    def saccade_peak_accelerations(self, get_raw=True):
        
        pk = self.get_pk_vel_idx()
        a_sp = self._speed_array()

        peaks = []
        for i, (start, end) in enumerate(self._intervals()):
            pk_i = int(min(pk[i], end - 1))
            v_pre = a_sp[start:pk_i + 1]
            if v_pre.size < 2:
                peaks.append(np.nan)
                continue
            
            acc_pre = np.abs(np.diff(v_pre)) * self.s_f
            peaks.append(float(np.nanmax(acc_pre)) if np.any(np.isfinite(acc_pre)) else np.nan)

        peaks = np.asarray(peaks, dtype=np.float64)
        results = {"peak_acceleration_mean": float(np.nanmean(peaks)),
                   "raw": peaks}
        
        if not get_raw:
            del results["raw"]
            
        return results

    def saccade_peak_decelerations(self, get_raw=True):
        
        pk = self.get_pk_vel_idx()
        a_sp = self._speed_array()

        peaks = []
        for i, (start, end) in enumerate(self._intervals()):
            pk_i = int(min(pk[i], end - 1))
            v_post = a_sp[pk_i:end]
            if v_post.size < 2:
                peaks.append(np.nan)
                continue
            
            acc_post = np.abs(np.diff(v_post)) * self.s_f
            peaks.append(float(np.nanmax(acc_post)) if np.any(np.isfinite(acc_post)) else np.nan)

        peaks = np.asarray(peaks, dtype=np.float64)
        results = {"peak_deceleration_mean": float(np.nanmean(peaks)),
                   "raw": peaks}
        
        if not get_raw:
            del results["raw"]
            
        return results


    def saccade_skewness_exponents(self, get_raw=True):
        
        pk = self.get_pk_vel_idx()
        _ints = self._intervals()

        b_i = np.array([it[0] for it in _ints], dtype=np.float64)
        s_l = np.array([it[1] - it[0] + 1 for it in _ints], dtype=np.float64)

        denom = s_l - 1.0  # nb de vitesses valides
        skw = np.full_like(denom, np.nan, dtype=np.float64)
        mask = denom > 0
        skw[mask] = (pk[mask].astype(np.float64) - b_i[mask]) / denom[mask]

        results = {
            "skewness_exponent_mean": float(np.nanmean(skw)),
            "skewness_exponent_sd": self._safe_sd(skw),
            "raw": skw,
        }
        
        if not get_raw:
            del results["raw"]
            
        return results


    def saccade_gamma_skewness_exponents(self, get_raw=True):
        
        a_sp = self._speed_array()

        skw = []
        for start, end in self._intervals():
            seg = a_sp[start:end]  # vitesses valides uniquement
            seg = seg[np.isfinite(seg)]
            seg = seg[seg > 0]

            if seg.size < 3:
                skw.append(np.nan)
                continue

            try:
                fit_shape, fit_loc, fit_scale = gamma.fit(seg)
                skw.append(float(fit_shape))
            except Exception:
                skw.append(np.nan)

        skw = np.asarray(skw, dtype=np.float64)
        results = {
            "skewness_exponent_mean": float(np.nanmean(skw)),
            "skewness_exponent_sd": self._safe_sd(skw),
            "raw": skw,
        }
        
        if not get_raw:
            del results["raw"]
            
        return results


    def saccade_amplitude_duration_ratios(self, get_raw=True):
        
        a_s = self.saccade_amplitudes(get_raw=True)["raw"]
        d_s = self.saccade_durations(get_raw=True)["raw"]

        r_ = np.full_like(a_s, np.nan, dtype=np.float64)
        mask = np.isfinite(a_s) & np.isfinite(d_s) & (d_s > 0)
        r_[mask] = a_s[mask] / d_s[mask]

        results = {"ratio_mean": float(np.nanmean(r_)),
                   "ratio_sd": self._safe_sd(r_),
                   "raw": r_}
        
        if not get_raw:
            del results["raw"]
            
        return results


    def saccade_peak_velocity_amplitude_ratios(self, get_raw=True):
        
        p_v = self.saccade_peak_velocities(get_raw=True)["raw"]
        a_s = self.saccade_amplitudes(get_raw=True)["raw"]

        r_ = np.full_like(p_v, np.nan, dtype=np.float64)
        mask = np.isfinite(p_v) & np.isfinite(a_s) & (a_s > 0)
        r_[mask] = p_v[mask] / a_s[mask]

        results = {"ratio_mean": float(np.nanmean(r_)),
                   "ratio_sd": self._safe_sd(r_),
                   "raw": r_}
        
        if not get_raw:
            del results["raw"]
            
        return results


    def saccade_peak_velocity_duration_ratios(self, get_raw=True):
        
        p_v = self.saccade_peak_velocities(get_raw=True)["raw"]
        d_s = self.saccade_durations(get_raw=True)["raw"]

        r_ = np.full_like(p_v, np.nan, dtype=np.float64)
        mask = np.isfinite(p_v) & np.isfinite(d_s) & (d_s > 0)
        r_[mask] = p_v[mask] / d_s[mask]

        results = {"ratio_mean": float(np.nanmean(r_)),
                   "ratio_sd": self._safe_sd(r_),
                   "raw": r_}
        
        if not get_raw:
            del results["raw"]
            
        return results


    def saccade_peak_velocity_velocity_ratios(self, get_raw=True):
        
        p_v = self.saccade_peak_velocities(get_raw=True)["raw"]
        a_d_r = self.saccade_amplitude_duration_ratios(get_raw=True)["raw"]

        r_ = np.full_like(p_v, np.nan, dtype=np.float64)
        mask = np.isfinite(p_v) & np.isfinite(a_d_r) & (a_d_r > 0)
        r_[mask] = p_v[mask] / a_d_r[mask]

        results = {"ratio_mean": float(np.nanmean(r_)),
                   "ratio_sd": self._safe_sd(r_),
                   "raw": r_}
        
        if not get_raw:
            del results["raw"]
            
        return results

    def saccade_acceleration_deceleration_ratios(self, get_raw=True):
        
        a_c = self.saccade_peak_accelerations(get_raw=True)["raw"]
        d_c = self.saccade_peak_decelerations(get_raw=True)["raw"]

        r_ = np.full_like(a_c, np.nan, dtype=np.float64)
        mask = np.isfinite(a_c) & np.isfinite(d_c) & (d_c > 0)
        r_[mask] = a_c[mask] / d_c[mask]

        results = {"ratio_mean": float(np.nanmean(r_)),
                   "ratio_sd": self._safe_sd(r_),
                   "raw": r_}
        
        if not get_raw:
            del results["raw"]
            
        return results


    def saccade_main_sequence(self, get_raw=True):
        
        a_s = self.saccade_amplitudes(get_raw=True)["raw"]
        d_s = self.saccade_durations(get_raw=True)["raw"]
        p_v = self.saccade_peak_velocities(get_raw=True)["raw"]

        mask = np.isfinite(a_s) & np.isfinite(d_s) & (d_s > 0) & np.isfinite(p_v) & (a_s > 0) & (p_v > 0)
        a_s_f = a_s[mask]
        d_s_f = d_s[mask]
        p_v_f = p_v[mask]

        # si pas assez de points, renvoyer NaN proprement
        if a_s_f.size < 2:
            results = {
                "slope_amplitude_duration": np.nan,
                "slope_log_peak_velocity_log_amplitude": np.nan,
                "raw_amplitude_duration": np.vstack((a_s, d_s)),
                "raw_log_peak_velocity_log_amplitude": np.vstack((np.log(p_v + 1e-12), np.log(a_s + 1e-12))),
            }
        else:
            l_p_v = np.log(p_v_f)
            l_a = np.log(a_s_f)

            coefs_ad = np.polyfit(d_s_f, a_s_f, 1)
            coefs_pa = np.polyfit(l_a, l_p_v, 1)

            if self.config.get("display_results", False):
                plot_main_sequence(d_s_f, a_s_f, coefs_ad, l_p_v, l_a, coefs_pa, self.config)

            results = {
                "slope_amplitude_duration": float(coefs_ad[0]),
                "slope_log_peak_velocity_log_amplitude": float(coefs_pa[0]),
                "raw_amplitude_duration": np.vstack((a_s, d_s)),
                "raw_log_peak_velocity_log_amplitude": np.vstack((np.log(p_v + 1e-12), np.log(a_s + 1e-12))),
            }

        if not get_raw:
            del results["raw_amplitude_duration"]
            del results["raw_log_peak_velocity_log_amplitude"] 

        return results


def saccade_count(input, **kwargs):
    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_count()
        input.verbose()

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_count()
        saccade_analysis.verbose()

    return results


def saccade_frequency(input, **kwargs):
    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_frequency()
        input.verbose()

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_frequency()
        saccade_analysis.verbose()

    return results


def saccade_frequency_wrt_labels(input, **kwargs):
    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_frequency_wrt_labels()
        input.verbose()

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_frequency_wrt_labels()
        saccade_analysis.verbose()

    return results


def saccade_durations(input, **kwargs):
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_durations(get_raw)
        input.verbose(dict({"get_raw": get_raw}))

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_durations(get_raw)
        saccade_analysis.verbose(dict({"get_raw": get_raw}))

    return results


def saccade_amplitudes(input, **kwargs):
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_amplitudes(get_raw)
        input.verbose(dict({"get_raw": get_raw}))

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_amplitudes(get_raw)
        saccade_analysis.verbose(dict({"get_raw": get_raw}))

    return results


def saccade_travel_distances(input, **kwargs):
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_travel_distances(get_raw)
        input.verbose(dict({"get_raw": get_raw}))

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_travel_distances(get_raw)
        saccade_analysis.verbose(dict({"get_raw": get_raw}))

    return results


def saccade_efficiencies(input, **kwargs):
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_efficiencies(get_raw)
        input.verbose(dict({"get_raw": get_raw}))

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_efficiencies(get_raw)
        saccade_analysis.verbose(dict({"get_raw": get_raw}))

    return results


def saccade_directions(input, **kwargs):
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_directions(get_raw)
        input.verbose(dict({"get_raw": get_raw}))

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_directions(get_raw)
        saccade_analysis.verbose(dict({"get_raw": get_raw}))

    return results


def saccade_horizontal_deviations(input, **kwargs):
    absolute = kwargs.get("saccade_absolute_horizontal_deviations", True)
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_horizontal_deviations(absolute, get_raw)
        input.verbose(
            dict(
                {"get_raw": get_raw, "saccade_absolute_horizontal_deviations": absolute}
            )
        )

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_horizontal_deviations(absolute, get_raw)
        saccade_analysis.verbose(
            dict(
                {"get_raw": get_raw, "saccade_absolute_horizontal_deviations": absolute}
            )
        )

    return results


def saccade_successive_deviations(input, **kwargs):
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_successive_deviations(get_raw)
        input.verbose(dict({"get_raw": get_raw}))

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_successive_deviations(get_raw)
        saccade_analysis.verbose(dict({"get_raw": get_raw}))

    return results


def saccade_initial_directions(input, **kwargs):
    duration_threshold = kwargs.get("saccade_init_direction_duration_threshold", 0.020)
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_initial_directions(duration_threshold, get_raw)
        input.verbose(
            dict(
                {
                    "get_raw": get_raw,
                    "saccade_init_direction_duration_threshold": duration_threshold,
                }
            )
        )

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_initial_directions(
            duration_threshold, get_raw
        )
        saccade_analysis.verbose(
            dict(
                {
                    "get_raw": get_raw,
                    "saccade_init_direction_duration_threshold": duration_threshold,
                }
            )
        )

    return results


def saccade_initial_deviations(input, **kwargs):
    duration_threshold = kwargs.get("saccade_init_deviation_duration_threshold", 0.020)
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_initial_deviations(duration_threshold, get_raw)
        input.verbose(
            dict(
                {
                    "get_raw": get_raw,
                    "saccade_init_deviation_duration_threshold": duration_threshold,
                }
            )
        )

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_initial_deviations(
            duration_threshold, get_raw
        )
        saccade_analysis.verbose(
            dict(
                {
                    "get_raw": get_raw,
                    "saccade_init_deviation_duration_threshold": duration_threshold,
                }
            )
        )

    return results


def saccade_max_curvatures(input, **kwargs):
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_max_curvatures(get_raw)
        input.verbose(dict({"get_raw": get_raw}))

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_max_curvatures(get_raw)
        saccade_analysis.verbose(dict({"get_raw": get_raw}))

    return results


def saccade_area_curvatures(input, **kwargs):
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_area_curvatures(get_raw)
        input.verbose(dict({"get_raw": get_raw}))

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_area_curvatures(get_raw)
        saccade_analysis.verbose(dict({"get_raw": get_raw}))

    return results


def saccade_mean_velocities(input, **kwargs):
    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_mean_velocities()
        input.verbose()

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_mean_velocities()
        saccade_analysis.verbose()

    return results


def saccade_average_velocity_means(input, **kwargs):
    weighted = kwargs.get("saccade_weighted_average_velocity_means", False)
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_average_velocity_means(weighted, get_raw)
        input.verbose(
            dict(
                {
                    "get_raw": get_raw,
                    "saccade_weighted_average_velocity_means": weighted,
                }
            )
        )

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_average_velocity_means(weighted, get_raw)
        saccade_analysis.verbose(
            dict(
                {
                    "get_raw": get_raw,
                    "saccade_weighted_average_velocity_means": weighted,
                }
            )
        )

    return results


def saccade_average_velocity_deviations(input, **kwargs):
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_average_velocity_deviations(get_raw)
        input.verbose(dict({"get_raw": get_raw}))

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_average_velocity_deviations(get_raw)
        saccade_analysis.verbose(dict({"get_raw": get_raw}))

    return results


def saccade_peak_velocities(input, **kwargs):
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_peak_velocities(get_raw)
        input.verbose(dict({"get_raw": get_raw}))

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_peak_velocities(get_raw)
        saccade_analysis.verbose(dict({"get_raw": get_raw}))

    return results


def saccade_mean_acceleration_profiles(input, **kwargs):
    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_mean_acceleration_profiles()
        input.verbose()

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_mean_acceleration_profiles()
        saccade_analysis.verbose()

    return results


def saccade_mean_accelerations(input, **kwargs):
    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_mean_accelerations()
        input.verbose()

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_mean_accelerations()
        saccade_analysis.verbose()

    return results


def saccade_mean_decelerations(input, **kwargs):
    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_mean_decelerations()
        input.verbose()

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_mean_decelerations()
        saccade_analysis.verbose()

    return results


def saccade_average_acceleration_profiles(input, **kwargs):
    weighted = kwargs.get("saccade_weighted_average_acceleration_profiles", False)
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_average_acceleration_profiles(weighted, get_raw)
        input.verbose(
            dict(
                {
                    "get_raw": get_raw,
                    "saccade_weighted_average_acceleration_profiles": weighted,
                }
            )
        )

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_average_acceleration_profiles(
            weighted, get_raw
        )
        saccade_analysis.verbose(
            dict(
                {
                    "get_raw": get_raw,
                    "saccade_weighted_average_acceleration_profiles": weighted,
                }
            )
        )

    return results


def saccade_average_acceleration_means(input, **kwargs):
    weighted = kwargs.get("saccade_weighted_average_acceleration_means", False)
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_average_acceleration_means(weighted, get_raw)
        input.verbose(
            dict(
                {
                    "get_raw": get_raw,
                    "saccade_weighted_average_acceleration_means": weighted,
                }
            )
        )

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_average_acceleration_means(weighted, get_raw)
        saccade_analysis.verbose(
            dict(
                {
                    "get_raw": get_raw,
                    "saccade_weighted_average_acceleration_means": weighted,
                }
            )
        )

    return results


def saccade_average_deceleration_means(input, **kwargs):
    weighted = kwargs.get("saccade_weighted_average_deceleration_means", False)
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_average_deceleration_means(weighted, get_raw)
        input.verbose(
            dict(
                {
                    "get_raw": get_raw,
                    "saccade_weighted_average_deceleration_means": weighted,
                }
            )
        )

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_average_deceleration_means(weighted, get_raw)
        saccade_analysis.verbose(
            dict(
                {
                    "get_raw": get_raw,
                    "saccade_weighted_average_deceleration_means": weighted,
                }
            )
        )

    return results


def saccade_peak_accelerations(input, **kwargs):
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_peak_accelerations(get_raw)
        input.verbose(dict({"get_raw": get_raw}))

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_peak_accelerations(get_raw)
        saccade_analysis.verbose(dict({"get_raw": get_raw}))

    return results


def saccade_peak_decelerations(input, **kwargs):
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_peak_decelerations(get_raw)
        input.verbose(dict({"get_raw": get_raw}))

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_peak_decelerations(get_raw)
        saccade_analysis.verbose(dict({"get_raw": get_raw}))

    return results


def saccade_skewness_exponents(input, **kwargs):
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_skewness_exponents(get_raw)
        input.verbose(dict({"get_raw": get_raw}))

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_skewness_exponents(get_raw)
        saccade_analysis.verbose(dict({"get_raw": get_raw}))

    return results


def saccade_gamma_skewness_exponents(input, **kwargs):
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_gamma_skewness_exponents(get_raw)
        input.verbose(dict({"get_raw": get_raw}))

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_gamma_skewness_exponents(get_raw)
        saccade_analysis.verbose(dict({"get_raw": get_raw}))

    return results


def saccade_amplitude_duration_ratios(input, **kwargs):
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_amplitude_duration_ratios(get_raw)
        input.verbose(dict({"get_raw": get_raw}))

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_amplitude_duration_ratios(get_raw)
        saccade_analysis.verbose(dict({"get_raw": get_raw}))

    return results


def saccade_peak_velocity_amplitude_ratios(input, **kwargs):
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_peak_velocity_amplitude_ratios(get_raw)
        input.verbose(dict({"get_raw": get_raw}))

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_peak_velocity_amplitude_ratios(get_raw)
        saccade_analysis.verbose(dict({"get_raw": get_raw}))

    return results


def saccade_peak_velocity_duration_ratios(input, **kwargs):
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_peak_velocity_duration_ratios(get_raw)
        input.verbose(dict({"get_raw": get_raw}))

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_peak_velocity_duration_ratios(get_raw)
        saccade_analysis.verbose(dict({"get_raw": get_raw}))

    return results


def saccade_peak_velocity_velocity_ratios(input, **kwargs):
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_peak_velocity_velocity_ratios(get_raw)
        input.verbose(dict({"get_raw": get_raw}))

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_peak_velocity_velocity_ratios(get_raw)
        saccade_analysis.verbose(dict({"get_raw": get_raw}))

    return results


def saccade_acceleration_deceleration_ratios(input, **kwargs):
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_acceleration_deceleration_ratios(get_raw)
        input.verbose(dict({"get_raw": get_raw}))

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_acceleration_deceleration_ratios(get_raw)
        saccade_analysis.verbose(dict({"get_raw": get_raw}))

    return results


def saccade_main_sequence(input, **kwargs):
    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, SaccadeAnalysis):
        results = input.saccade_main_sequence(get_raw)
        input.verbose(dict({"get_raw": get_raw}))

    else:
        saccade_analysis = SaccadeAnalysis.generate(input, **kwargs)
        results = saccade_analysis.saccade_main_sequence(get_raw)
        saccade_analysis.verbose(dict({"get_raw": get_raw}))

    return results
