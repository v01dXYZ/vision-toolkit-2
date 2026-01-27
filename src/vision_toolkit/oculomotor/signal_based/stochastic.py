# -*- coding: utf-8 -*-
import numpy as np

from vision_toolkit.oculomotor.signal_based.signal_based_base import SignalBased
from vision_toolkit.visualization.oculomotor.stochastic import (plot_dacf, plot_dfa,
                                                        plot_msd)


class StochasticAnalysis(SignalBased):
    def __init__(self, input, **kwargs):
        """


        Parameters
        ----------
        input : TYPE
            DESCRIPTION.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        verbose = kwargs.get("verbose", True)

        if verbose:
            print("Processing Stochastic Analysis...")

        super().__init__(input, **kwargs)

        if verbose:
            print("...Stochastic Analysis done")


 
    def _clean_signal(self, x, status=None, 
                      interpolate=True, max_gap_samples=None):
        
        x = np.asarray(x, dtype=float).copy()
    
        # Apply status mask -> NaN
        if status is not None:
            status = np.asarray(status)
            if status.shape[0] != x.shape[0]:
                raise ValueError("status must have the same length as x")
            valid = status.astype(bool)
            x[~valid] = np.nan
    
        # If no interpolation requested, just validate enough points exist
        if not interpolate:
            if np.sum(np.isfinite(x)) < 2:
                return None
            return x
    
        # Nothing to do
        if not np.isnan(x).any():
            return x
    
        idx = np.arange(x.shape[0])
        good = np.isfinite(x)
    
        # Need at least 2 points to interpolate
        if good.sum() < 2:
            return None
    
        if max_gap_samples is None:
            # Interpolate all NaNs
            x[~good] = np.interp(idx[~good], idx[good], x[good])
            return x
    
        # Interpolate only "short" NaN runs, keep long runs as NaN
        max_gap_samples = int(max_gap_samples)
        if max_gap_samples < 1:
            return x
    
        nan_mask = ~good
        # Find contiguous NaN runs [start, end)
        d = np.diff(nan_mask.astype(int))
        starts = np.where(d == 1)[0] + 1
        ends = np.where(d == -1)[0] + 1
    
        if nan_mask[0]:
            starts = np.r_[0, starts]
        if nan_mask[-1]:
            ends = np.r_[ends, x.shape[0]]
    
        # Interpolate each short run only if it is bracketed by valid samples
        for s, e in zip(starts, ends):
            run_len = e - s
            if run_len <= max_gap_samples:
                left = s - 1
                right = e
                # Need valid points on both sides for meaningful interpolation
                if left >= 0 and right < x.shape[0] and np.isfinite(x[left]) and np.isfinite(x[right]):
                    x[s:e] = np.interp(idx[s:e], idx[[left, right]], x[[left, right]])
    
        return x
    
        
    def msd_computation(self, x, lag):
        
        n_s = x.shape[1]

        msd = np.sum((x[:, lag:] - x[:, :-lag]) ** 2, axis=1)
        msd = msd / (n_s - lag)

        return msd


    def msd_generate(self, _inputs, MSD_min_lag, MSD_max_lag, MSD_nb_lags):
        log_min = np.round(np.log10(MSD_min_lag))
        log_max = np.round(np.log10(MSD_max_lag))

        log_lags = np.linspace(log_min, log_max, MSD_nb_lags)
        lags = np.round(10**log_lags)

        new_lags = list(set(lags))
        new_lags.sort()

        lags_msd = [int(lag) for lag in new_lags]
        msds = np.zeros((2, len(lags_msd)))

        for k, lag in enumerate(lags_msd):
            msds[:, k] = self.msd_computation(_inputs, lag)

        return lags_msd, msds


    def comp_scaling_exponents(self, vals, time_lags, short_i, long_i):
        lg_msds = np.log10(vals)
        lg_lag = np.log10(time_lags)

        coefs = dict()

        for k, _dir in enumerate(["x", "y"]):
            lg_msd = lg_msds[k, :]

            s_coefs = np.polyfit(lg_lag[short_i], lg_msd[short_i], 1)
            l_coefs = np.polyfit(lg_lag[long_i], lg_msd[long_i], 1)

            coefs.update({_dir: [s_coefs, l_coefs]})

        return coefs


    def MSD(
        self,
        short_time_scale,
        long_time_scale,
        MSD_min_lag,
        MSD_max_lag,
        MSD_nb_lags,
        get_raw,
    ):
        if MSD_max_lag is None:
            MSD_max_lag = int(self.config["nb_samples"] / 4)

        if MSD_max_lag > self.config["nb_samples"] / 4:
            MSD_max_lag = int(self.config["nb_samples"] / 4)
            print("'MSD_max_lag' updated to new value: {n_}".format(n_=MSD_max_lag))

        status = self.data_set.get("status", None)
        x = self._clean_signal(self.data_set["x_array"], status=status, interpolate=True)
        y = self._clean_signal(self.data_set["y_array"], status=status, interpolate=True)
        if x is None or y is None:
            raise ValueError("Not enough valid samples for stochastic analysis")
        
        pos = np.vstack([x, y])  # shape (2, n)
        
        n_s = pos.shape[1]
        if MSD_max_lag is None or MSD_max_lag > n_s // 4:
            MSD_max_lag = max(1, n_s // 4)
        
        if MSD_min_lag < 1:
            MSD_min_lag = 1
        
        if MSD_min_lag >= MSD_max_lag:
            raise ValueError(
                f"Invalid MSD lag range: min={MSD_min_lag}, max={MSD_max_lag}"
            )

        lags, msds = self.msd_generate(pos, MSD_min_lag, MSD_max_lag, MSD_nb_lags)
        time_lags = np.array(lags) / self.config["sampling_frequency"]

        short_i = np.where(
            np.logical_and(
                time_lags > short_time_scale[0], time_lags < short_time_scale[1]
            )
        )

        long_i = np.where(
            np.logical_and(
                time_lags > long_time_scale[0],
                time_lags < min(long_time_scale[1], max(time_lags)),
            )
        )

        coefs = self.comp_scaling_exponents(msds, time_lags, short_i, long_i)

        alphas = {
            "x_scaling_exponents": {
                "short_time_scale": coefs["x"][0][0],
                "long_time_scale": coefs["x"][1][0],
            },
            "y_scaling_exponents": {
                "short_time_scale": coefs["y"][0][0],
                "long_time_scale": coefs["y"][1][0],
            },
        }

        if self.config["display_results"]:
            plot_msd(time_lags, msds, coefs, short_time_scale, long_time_scale)

        self.verbose(
            dict(
                {
                    "MSD_min_lag": MSD_min_lag,
                    "MSD_max_lag": MSD_max_lag,
                    "MSD_nb_lags": MSD_nb_lags,
                    "stochastic_analysis_short_time_scale": short_time_scale,
                    "stochastic_analysis_long_time_scale": long_time_scale,
                    "get_raw": get_raw,
                }
            )
        )

        results = dict(
            {
                "scaling_exponents": alphas,
                "lags_MSD": np.array(lags),
                "time_lags_MSD": time_lags,
                "MSDs": msds,
            }
        )

        if not get_raw:
            return alphas

        else:
            return results


    def dacf_computation(self, x, lag, DACF_order):
        
        n_s = x.shape[1]

        dacf = np.sum(
            (x[:, lag + DACF_order :] - x[:, lag:-DACF_order])
            * (x[:, DACF_order:-lag] - x[:, : -lag - DACF_order]),
            axis=1,
        )

        dacf /= DACF_order**2 * (n_s - lag - DACF_order)

        return dacf


    def dacf_generate(
        self,
        _inputs,
        DACF_min_lag,
        DACF_max_lag,
        DACF_nb_lags,
        DACF_order,
        DACF_normalization,
    ):
        lags = np.linspace(DACF_min_lag, DACF_max_lag, DACF_nb_lags)

        lags_dacf = [int(lag) for lag in lags]

        dacfs = np.zeros((2, DACF_nb_lags))

        for k, lag in enumerate(lags_dacf):
            dacfs[:, k] = self.dacf_computation(_inputs, lag, DACF_order)

        if DACF_normalization:
            den = dacfs[:, 0].reshape(_inputs.shape[0], 1)
            dacfs = dacfs / (den + 1e-30)

        return lags_dacf, dacfs


    def DACF(
        self, DACF_min_lag, DACF_max_lag, DACF_nb_lags, DACF_order, DACF_normalization
    ):
        if DACF_max_lag is None:
            DACF_max_lag = int(self.config["nb_samples"] / 4)

        if DACF_max_lag > self.config["nb_samples"] / 4:
            DACF_max_lag = int(self.config["nb_samples"] / 4)
            print("'DACF_max_lag' updated to new value: {n_}".format(n_=DACF_max_lag))
 
        status = self.data_set.get("status", None)
        x = self._clean_signal(self.data_set["x_array"], status=status, interpolate=True)
        y = self._clean_signal(self.data_set["y_array"], status=status, interpolate=True)
        if x is None or y is None:
            raise ValueError("Not enough valid samples for stochastic analysis")
        
        pos = np.vstack([x, y])  # shape (2, n)
        
        n_s = pos.shape[1]
        DACF_max_lag = min(DACF_max_lag, n_s // 4)
        
        if DACF_max_lag <= DACF_min_lag:
            raise ValueError(f"Invalid DACF lag range: min={DACF_min_lag}, max={DACF_max_lag} (n_s={n_s})")
         
        lags, dacfs = self.dacf_generate(
            pos,
            DACF_min_lag,
            DACF_max_lag,
            DACF_nb_lags,
            DACF_order,
            DACF_normalization,
        )
        time_lags = np.array(lags) / self.config["sampling_frequency"]

        if self.config["display_results"]:
            plot_dacf(DACF_order, time_lags, dacfs)

        self.verbose(
            dict(
                {
                    "DACF_min_lag": DACF_min_lag,
                    "DACF_max_lag": DACF_max_lag,
                    "DACF_nb_lags": DACF_nb_lags,
                    "DACF_order": DACF_order,
                    "DACF_normalization": DACF_normalization,
                }
            )
        )

        results = dict(
            {"lags_DACF": np.array(lags), "time_lags_DACF": time_lags, "DACFs": dacfs}
        )

        return results

    def dfa_find_segments(self, _input, w_l, DFA_overlap):
        
        x = np.asarray(_input)
        n_s = x.shape[0]
        w_l = int(w_l)
    
        if w_l < 2 or w_l > n_s:
            return np.empty((0, w_l), dtype=float)
    
        if DFA_overlap:
            step = max(1, w_l // 2)
            starts = np.arange(0, n_s - w_l + 1, step)
            segs = np.stack([x[i:i + w_l] for i in starts], axis=0) if len(starts) else np.empty((0, w_l))
        else:
            n_trim = n_s - (n_s % w_l)
            if n_trim < w_l:
                return np.empty((0, w_l), dtype=float)
            segs = x[:n_trim].reshape((n_trim // w_l, w_l))
    
        return segs
    

    def dfa_fractal_trends(self, segs, w_l, DFA_order):
        
        if segs.shape[0] == 0:
            return np.empty_like(segs)
    
        x = np.arange(w_l)
        # polyfit vectorisé: segs.T shape (w_l, n_segs)
        coefs = np.polyfit(x, segs.T, DFA_order).T
        trnds = np.array([np.polyval(coefs[j], x) for j in range(len(segs))])
        
        return trnds


    def dfa_fractal_fluctuation(self, segs_f, segs_r, trnds_f, trnds_r):
        
        if segs_f.shape[0] == 0 or segs_r.shape[0] == 0:
            return np.nan
    
        detrnds_f = segs_f - trnds_f
        detrnds_r = segs_r - trnds_r
    
        f = np.sum(detrnds_f**2, axis=1) / detrnds_f.shape[1]
        r = np.sum(detrnds_r**2, axis=1) / detrnds_r.shape[1]
        fluct = np.sqrt(np.mean(np.concatenate([f, r])))
        
        return fluct

    
    def dfa_fractal(self,
                    short_time_scale,
                    long_time_scale,
                    DFA_min_lag,
                    DFA_max_lag,
                    DFA_nb_lags,
                    DFA_order,
                    DFA_overlap,):
        
        fluct_s = np.zeros((2, DFA_nb_lags))
        lags = np.zeros(DFA_nb_lags)
        h_exps = {}
    
        for k, _dir in enumerate(["x_array", "y_array"]):
            
            fs = self.config["sampling_frequency"]
            max_gap = int(0.1 * fs)  # 100 ms
            
            status = self.data_set.get("status", None)
            _input = self._clean_signal(
                self.data_set[_dir],
                status=status,
                interpolate=True,
                max_gap_samples=max_gap
            )

            if _input is None:
                raise ValueError("Not enough valid samples for DFA")
    
            n_s = _input.shape[0]
    
            # clamp local: évite fenêtres impossibles
            max_lag = min(int(DFA_max_lag), max(2, n_s // 4))
            min_lag = max(int(DFA_min_lag), DFA_order + 2)
    
            # génère des fenêtres log-spaced
            raw_ws = (10 ** np.linspace(np.log10(min_lag), np.log10(max_lag), DFA_nb_lags)).astype(int)
    
            # on calcule fluct sur chaque w, même si doublons, mais on stabilise le fit ensuite
            fluct = np.zeros(DFA_nb_lags, dtype=float)
    
            for i, w in enumerate(raw_ws):
                lags[i] = w
    
                seg_f = self.dfa_find_segments(_input, w, DFA_overlap)
                seg_r = self.dfa_find_segments(np.flip(_input), w, DFA_overlap)
    
                tr_f = self.dfa_fractal_trends(seg_f, w, DFA_order)
                tr_r = self.dfa_fractal_trends(seg_r, w, DFA_order)
    
                fluct[i] = self.dfa_fractal_fluctuation(seg_f, seg_r, tr_f, tr_r)
    
            fluct_s[k, :] = fluct
    
            # Fit Hurst: déduplique lags + enlève NaN/0
            l = lags.copy()
            f = fluct.copy()
            ok = np.isfinite(f) & (f > 0) & (l > 0)
            l = l[ok]
            f = f[ok]
    
            # déduplication sur lags (garde moyenne des fluctuations pour même lag)
            if len(l) >= 2:
                uniq = {}
                for lag_i, f_i in zip(l, f):
                    uniq.setdefault(int(lag_i), []).append(float(f_i))
                l_u = np.array(sorted(uniq.keys()), dtype=float)
                f_u = np.array([np.mean(uniq[int(li)]) for li in l_u], dtype=float)
    
                if len(l_u) >= 2:
                    h_exps[_dir[0]] = np.polyfit(np.log10(l_u), np.log10(f_u), 1)[0]
                else:
                    h_exps[_dir[0]] = np.nan
            else:
                h_exps[_dir[0]] = np.nan
    
        time_lags = lags / self.config["sampling_frequency"]
    
        short_i = np.where((time_lags > short_time_scale[0]) & (time_lags < short_time_scale[1]))
        long_i = np.where((time_lags > long_time_scale[0]) & (time_lags < min(long_time_scale[1], np.nanmax(time_lags))))
    
        coefs = self.comp_scaling_exponents(fluct_s, time_lags, short_i, long_i)
    
        results = {
            "lags_DFA": lags,
            "time_lags_DFA": time_lags,
            "fluctuations_DFA": fluct_s,
        }
        
        return results, h_exps, coefs


    def DFA(
        self,
        short_time_scale,
        long_time_scale,
        DFA_min_lag,
        DFA_max_lag,
        DFA_nb_lags,
        DFA_order,
        DFA_overlap,
        get_raw,
    ):
        if DFA_max_lag is None:
            DFA_max_lag = int(self.config["nb_samples"] / 4)

        if DFA_max_lag > self.config["nb_samples"] / 4:
            DFA_max_lag = int(self.config["nb_samples"] / 4)
            print("'DFA_max_lag' updated to new value: {n_}".format(n_=DFA_max_lag))

        results, h_exps, coefs = self.dfa_fractal(
            short_time_scale,
            long_time_scale,
            DFA_min_lag,
            DFA_max_lag,
            DFA_nb_lags,
            DFA_order,
            DFA_overlap,
        )

        alphas = {
            "x_scaling_exponents": {
                "short_time_scale": coefs["x"][0][0],
                "long_time_scale": coefs["x"][1][0],
            },
            "y_scaling_exponents": {
                "short_time_scale": coefs["y"][0][0],
                "long_time_scale": coefs["y"][1][0],
            },
        }

        if self.config["display_results"]:
            plot_dfa(
                results["time_lags_DFA"],
                results["fluctuations_DFA"],
                coefs,
                short_time_scale,
                long_time_scale,
            )

        self.verbose(
            dict(
                {
                    "DFA_min_lag": DFA_min_lag,
                    "DFA_max_lag": DFA_max_lag,
                    "DFA_nb_lags": DFA_nb_lags,
                    "DFA_order": DFA_order,
                    "DFA_overlap": DFA_overlap,
                    "stochastic_analysis_short_time_scale": short_time_scale,
                    "stochastic_analysis_long_time_scale": long_time_scale,
                    "get_raw": get_raw,
                }
            )
        )

        if not get_raw:
            return {"hurst_exponents": h_exps, "local_scaling_exponents": alphas}
        else:
            results.update(
                {"hurst_exponents": h_exps, "local_scaling_exponents": alphas}
            )
            return results


def MSD(input, **kwargs):
    short_time_scale = kwargs.get(
        "stochastic_analysis_short_time_scale", [0.012, 0.120]
    )
    long_time_scale = kwargs.get("stochastic_analysis_long_time_scale", [0.120, 1.200])

    MSD_min_lag = kwargs.get("MSD_min_lag", 1)
    MSD_max_lag = kwargs.get("MSD_max_lag", None)
    MSD_nb_lags = kwargs.get("MSD_nb_lags", 50)

    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, StochasticAnalysis):
        results = input.MSD(
            short_time_scale,
            long_time_scale,
            MSD_min_lag,
            MSD_max_lag,
            MSD_nb_lags,
            get_raw,
        )

    else:
        stochastic_analysis = StochasticAnalysis.generate(input, **kwargs)

        results = stochastic_analysis.MSD(
            short_time_scale,
            long_time_scale,
            MSD_min_lag,
            MSD_max_lag,
            MSD_nb_lags,
            get_raw,
        )

    return results


def DACF(input, **kwargs):
    DACF_min_lag = kwargs.get("DACF_min_lag", 1)
    DACF_max_lag = kwargs.get("DACF_max_lag", None)
    DACF_nb_lags = kwargs.get("DACF_nb_lags", 50)

    DACF_order = kwargs.get("DACF_order", 20)
    DACF_normalization = kwargs.get("DACF_normalization", True)

    if isinstance(input, StochasticAnalysis):
        results = input.DACF(
            DACF_min_lag, DACF_max_lag, DACF_nb_lags, DACF_order, DACF_normalization
        )

    else:
        stochastic_analysis = StochasticAnalysis.generate(input, **kwargs)
        results = stochastic_analysis.DACF(
            DACF_min_lag, DACF_max_lag, DACF_nb_lags, DACF_order, DACF_normalization
        )

    return results


def DFA(input, **kwargs):
    short_time_scale = kwargs.get(
        "stochastic_analysis_short_time_scale", [0.012, 0.120]
    )
    long_time_scale = kwargs.get("stochastic_analysis_long_time_scale", [0.120, 1.200])

    DFA_order = kwargs.get("DFA_order", 2)
    DFA_overlap = kwargs.get("DFA_overlap", False)

    DFA_min_lag = kwargs.get("DFA_min_lag", DFA_order + 2)
    DFA_max_lag = kwargs.get("DFA_max_lag", None)
    DFA_nb_lags = kwargs.get("DFA_nb_lags", 50)

    get_raw = kwargs.get("get_raw", True)

    if isinstance(input, StochasticAnalysis):
        results = input.DFA(
            short_time_scale,
            long_time_scale,
            DFA_min_lag,
            DFA_max_lag,
            DFA_nb_lags,
            DFA_order,
            DFA_overlap,
            get_raw,
        )

    else:
        stochastic_analysis = StochasticAnalysis.generate(input, **kwargs)
        results = stochastic_analysis.DFA(
            short_time_scale,
            long_time_scale,
            DFA_min_lag,
            DFA_max_lag,
            DFA_nb_lags,
            DFA_order,
            DFA_overlap,
            get_raw,
        )

    return results
