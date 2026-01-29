# -*- coding: utf-8 -*-

import numpy as np  
from scipy.ndimage import gaussian_filter

from vision_toolkit.scanpath.scanpath_base import Scanpath 
from vision_toolkit.segmentation.processing.binary_segmentation import BinarySegmentation
from vision_toolkit.scanpath.single.saliency.saliency_map_base import SaliencyMap
from vision_toolkit.utils.binning import spatial_bin

class PairSaliencyMap:
    
    def __init__(self, 
                 scanpath_1, scanpath_2,  
                 x_size, y_size,
                 **kwargs):
       
        kwargs.update(size_plan_x=x_size,
                      size_plan_y=y_size)
        
        self.sm_1 = SaliencyMap(scanpath_1, comp_saliency_map = True,
                                **kwargs)
        
        self.sm_2 = SaliencyMap(scanpath_2, comp_saliency_map = True,
                                **kwargs)
        
      
    def comp_pearson_corr(self):
        
        s_m_1 = self.sm_1.s_m.flatten()
        s_m_2 = self.sm_2.s_m.flatten()
        
        S = np.stack((s_m_1, s_m_2), axis=0)
        cov_ = np.cov(S)[0,1]
     
        p_c = cov_/(np.std(s_m_1) * np.std(s_m_2))
        
        return p_c
    
       
    def comp_kl_divergence(self):
        
        s_m_1 = self.sm_1.s_m.flatten()
        s_m_2 = self.sm_2.s_m.flatten()
        
        kl_d = np.sum(s_m_1 * np.log(s_m_1/s_m_2)) 
        
        return kl_d
    
    
    

class SaliencyReference:
    def __init__(self, input, ref_saliency_map, **kwargs):
        
        if isinstance(input, list):
            if isinstance(input[0], Scanpath):
                self.scanpaths = input
            elif isinstance(input[0], (str, BinarySegmentation)):
                self.scanpaths = [Scanpath.generate(inp, **kwargs) for inp in input]
            else:
                raise ValueError("input list must contain Scanpath, csv path, or BinarySegmentation")
        else:
            if isinstance(input, Scanpath):
                self.scanpaths = [input]
            elif isinstance(input, (str, BinarySegmentation)):
                self.scanpaths = [Scanpath.generate(input, **kwargs)]
            else:
                raise ValueError("input must be Scanpath, csv path, or BinarySegmentation")

        sp0 = self.scanpaths[0]
        self.size_plan_x = sp0.config["size_plan_x"]
        self.size_plan_y = sp0.config["size_plan_y"]

        # --- grille (doit matcher la ref) ---
        self.p_n_x = kwargs.get("scanpath_saliency_pixel_number_x", 100)
        self.p_n_y = kwargs.get("scanpath_saliency_pixel_number_y", None)
        if self.p_n_y is None:
            ratio = self.size_plan_x / self.size_plan_y
            self.p_n_y = int(round(self.p_n_x / ratio))

        # forcer odd si tu veux garder ta convention
        self.p_n_x = self.p_n_x + 1 if (self.p_n_x % 2) == 0 else self.p_n_x
        self.p_n_y = self.p_n_y + 1 if (self.p_n_y % 2) == 0 else self.p_n_y
        
        self.eps = 1e-12

        # --- charger ref map ---
        if isinstance(ref_saliency_map, dict):
            # accepte plusieurs clés
            for k in ("saliency_map", "absolute_duration_saliency_map", "relative_duration_saliency_map"):
                if k in ref_saliency_map:
                    self.ref_sm = np.asarray(ref_saliency_map[k], dtype=float)
                    break
            else:
                raise KeyError("ref_saliency_map dict must contain a saliency map key.")
        else:
            self.ref_sm = np.asarray(ref_saliency_map, dtype=float)

        if self.ref_sm.shape != (self.p_n_y, self.p_n_x):
            raise ValueError(
                f"ref map shape {self.ref_sm.shape} != ({self.p_n_y}, {self.p_n_x}). "
                "Use same binning params as the reference."
            )

        # cache pour percentile
        self._ref_sorted = np.sort(self.ref_sm.ravel())

        self.verbose = kwargs.get("verbose", True)


    def _fix_bins(self, scanpath_index=0):
        
        sp = self.scanpaths[scanpath_index]
        seq = sp.values  # (3, n_fix)
        b = spatial_bin(seq[0:2], self.p_n_x, self.p_n_y, self.size_plan_x, self.size_plan_y)
        xs = b[0].astype(int)
        ys = b[1].astype(int)

        finite_xy = np.isfinite(seq[0]) & np.isfinite(seq[1])
        mask = finite_xy & (0 <= xs) & (xs < self.p_n_x) & (0 <= ys) & (ys < self.p_n_y)
        
        return xs[mask], ys[mask]


    def _fix_bins_all(self):
        """Concatène les bins (xs, ys) de tous les scanpaths."""
        xs_all, ys_all = [], []
    
        for i in range(len(self.scanpaths)):
            xs, ys = self._fix_bins(i)
            if xs.size > 0:
                xs_all.append(xs)
                ys_all.append(ys)
    
        if not xs_all:
            return np.array([], dtype=int), np.array([], dtype=int)
    
        return np.concatenate(xs_all), np.concatenate(ys_all)


    def scanpath_saliency_percentile(self, scanpath_index='all'):
        """
        

        Parameters
        ----------
        scanpath_index : TYPE, optional
            DESCRIPTION. The default is 'all'.

        Returns
        -------
        dict
            DESCRIPTION.

        """
        if scanpath_index == "all":
            xs, ys = self._fix_bins_all()
        else:
            xs, ys = self._fix_bins(scanpath_index)
    
        if xs.size == 0:
            return {"percentile": np.nan}
    
        vals = self.ref_sm[ys, xs]
        ranks = np.searchsorted(self._ref_sorted, vals, side="left")
        perc = 100.0 * float(np.mean(ranks / self._ref_sorted.size))
    
        return {"percentile": perc}
    

    def scanpath_saliency_nss(self, scanpath_index="all", sigma_kernel=0.0, delta_neighborhood=None):
        """
        

        Parameters
        ----------
        scanpath_index : TYPE, optional
            DESCRIPTION. The default is "all".
        sigma_kernel : TYPE, optional
            DESCRIPTION. The default is 0.0.
        delta_neighborhood : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        dict
            DESCRIPTION.

        """
        flat = self.ref_sm.ravel()
        mu = float(np.mean(flat))
        sigma = float(np.std(flat, ddof=1))
        if sigma == 0:
            return {"nss": np.nan}
    
        z = (self.ref_sm - mu) / sigma
    
        # --- appliquer tolérance spatiale ---
        # truncate = nb d'écarts-types jusqu'auquel on garde la gaussienne
        # rayon approx = truncate * sigma_kernel  => on veut ~ delta_neighborhood
        z_tol = z
        if sigma_kernel and sigma_kernel > 0:
            if delta_neighborhood is None:
                # gaussienne non tronquée (truncate par défaut 4.0)
                z_tol = gaussian_filter(z, sigma=sigma_kernel, mode="constant", cval=0.0)
            else:
                # tronquer pour que la gaussienne soit ~nulle hors du voisinage voulu
                truncate = max(float(delta_neighborhood) / float(sigma_kernel), 0.0)
                z_tol = gaussian_filter(
                    z, sigma=sigma_kernel, truncate=truncate,
                    mode="constant", cval=0.0
                )
        else:
            # sigma_kernel == 0 : z_tol = z (NSS standard)
            z_tol = z
    
        # --- récupérer les bins de fixation ---
        if scanpath_index == "all":
            xs, ys = self._fix_bins_all()
        else:
            xs, ys = self._fix_bins(scanpath_index)
    
        if xs.size == 0:
            return {"nss": np.nan}
    
        # --- NSS = moyenne sur les fixations ---
        return {"nss": float(np.mean(z_tol[ys, xs]))}
     
        

    def _as_prob_map(self, m):
        """
        Convert any non-negative map into a valid probability map:
        - clip negatives to 0
        - normalize to sum=1 (fallback to uniform if sum==0)
        - clip to [eps, 1] to avoid log(0) in IG
        """
        m = np.asarray(m, dtype=float)
        m = np.maximum(m, 0.0)
    
        s = float(m.sum())
        if s <= 0:
            m = np.full_like(m, 1.0 / m.size, dtype=float)
        else:
            m = m / s
    
        return np.clip(m, self.eps, 1.0)
    
    
    def baseline_uniform(self):
        """Uniform baseline as a probability map."""
        m = np.full((self.p_n_y, self.p_n_x), 1.0 / (self.p_n_x * self.p_n_y), dtype=float)
        return np.clip(m, self.eps, 1.0)
    
    
    def baseline_center_gaussian(self, sigma_x=None, sigma_y=None):
        """
        Gaussian center-bias baseline as a probability map.
        sigma_x / sigma_y are expressed in bins.
    
        Defaults:
          sigma_x = 0.25 * p_n_x
          sigma_y = 0.25 * p_n_y
        """
        yy, xx = np.mgrid[0:self.p_n_y, 0:self.p_n_x]
        cx = (self.p_n_x - 1) / 2.0
        cy = (self.p_n_y - 1) / 2.0
    
        if sigma_x is None:
            sigma_x = 0.25 * self.p_n_x
        if sigma_y is None:
            sigma_y = 0.25 * self.p_n_y
    
        g = np.exp(-0.5 * (((xx - cx) / float(sigma_x)) ** 2 + ((yy - cy) / float(sigma_y)) ** 2))
        return self._as_prob_map(g)
    
    
    def _get_baseline(self, baseline, baseline_sigma_x=None, baseline_sigma_y=None):
        """
        baseline:
          - "uniform"
          - "center_gaussian"
        """
        if not isinstance(baseline, str):
            raise TypeError('baseline must be a string: "uniform" or "center_gaussian"')
    
        if baseline == "uniform":
            return self.baseline_uniform()
    
        if baseline == "center_gaussian":
            return self.baseline_center_gaussian(
                sigma_x=baseline_sigma_x,
                sigma_y=baseline_sigma_y,
            )
    
        raise ValueError('baseline must be one of: "uniform", "center_gaussian"')
    
    
    def scanpath_saliency_information_gain(self, scanpath_index="all",
        baseline="center_gaussian", baseline_sigma_x=None, baseline_sigma_y=None):
        """
        

        Parameters
        ----------
        scanpath_index : TYPE, optional
            DESCRIPTION. The default is "all".
        baseline : TYPE, optional
            DESCRIPTION. The default is "center_gaussian".
        baseline_sigma_x : TYPE, optional
            DESCRIPTION. The default is None.
        baseline_sigma_y : TYPE, optional
            DESCRIPTION. The default is None.
         : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # model probability map
        q_model = self._as_prob_map(self.ref_sm)
    
        # baseline probability map
        q_base = self._get_baseline(
            baseline,
            baseline_sigma_x=baseline_sigma_x,
            baseline_sigma_y=baseline_sigma_y,
        )
    
        # fixation bins
        if scanpath_index == "all":
            xs, ys = self._fix_bins_all()
        else:
            xs, ys = self._fix_bins(scanpath_index)
    
        if xs.size == 0:
            return {"information_gain": np.nan}
    
        ig = float(np.mean(np.log2(q_model[ys, xs]) - np.log2(q_base[ys, xs])))
    
        out = {"information_gain": ig}
        return out
    

    def _as_range_map(self, m, jitter=False, rng=None):
        """
        Normalize a saliency map to [0, 1] by range (min-max).
        Optionally add tiny jitter to break ties (recommended for AUC-Judd).
        """
        s = np.asarray(m, dtype=float)

        if jitter:
            if rng is None:
                rng = np.random.default_rng()
            # "tiny random value to each pixel" to avoid large uniform regions
            # (as described in the AUC appendix)
            s = s + rng.uniform(0.0, 1e-7, size=s.shape)

        mn = float(np.nanmin(s))
        mx = float(np.nanmax(s))
        if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
            # degenerate map -> constant; return zeros
            return np.zeros_like(s, dtype=float)

        s = (s - mn) / (mx - mn)
       
        return np.clip(s, 0.0, 1.0)


    def _gt_binary_map(self, scanpath_index="all"):
        """
        Build a binary fixation map QB (shape = (p_n_y, p_n_x)),
        from fixation bins. Duplicates collapse to 1 (binary).
        """
        if scanpath_index == "all":
            xs, ys = self._fix_bins_all()
        else:
            xs, ys = self._fix_bins(scanpath_index)

        gt = np.zeros((self.p_n_y, self.p_n_x), dtype=np.uint8)
        if xs.size == 0:
            return gt

        gt[ys, xs] = 1
        return gt

   
    def scanpath_saliency_auc_judd(self, scanpath_index="all", jitter=True, seed=None):
        """
        

        Parameters
        ----------
        scanpath_index : TYPE, optional
            DESCRIPTION. The default is "all".
        jitter : TYPE, optional
            DESCRIPTION. The default is True.
        seed : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        dict
            DESCRIPTION.

        """
        gt = self._gt_binary_map(scanpath_index)
        n_fix = int(gt.sum())
        if n_fix == 0:
            return {"auc_judd": np.nan}

        rng = np.random.default_rng(seed)
        s_map = self._as_range_map(self.ref_sm, jitter=jitter, rng=rng)

        # thresholds from saliency at fixated pixels
        fix_vals = s_map[gt == 1]
        thresholds = np.unique(fix_vals)
        if thresholds.size == 0:
            return {"auc_judd": np.nan}

        n_pix = gt.size
        n_nonfix = n_pix - n_fix
        if n_nonfix <= 0:
            return {"auc_judd": np.nan}

        # ROC points (FP on x-axis, TP on y-axis)
        fp = [0.0]
        tp = [0.0]

        # loop over thresholds (typically <= #fixations)
        for thr in thresholds:
            above = (s_map >= thr)
            overlap = int(np.sum(above & (gt == 1)))

            tp_rate = overlap / float(n_fix)
            fp_rate = (int(np.sum(above)) - overlap) / float(n_nonfix)

            fp.append(fp_rate)
            tp.append(tp_rate)

        fp.append(1.0)
        tp.append(1.0)
 
        fp = np.asarray(fp, dtype=float)
        tp = np.asarray(tp, dtype=float)
        order = np.argsort(fp)
        auc = float(np.trapz(tp[order], fp[order]))

        return {"auc_judd": auc}

 
    def scanpath_saliency_auc_borji(self, scanpath_index="all", splits=100, stepsize=0.1, seed=None):
        """
        

        Parameters
        ----------
        scanpath_index : TYPE, optional
            DESCRIPTION. The default is "all".
        splits : TYPE, optional
            DESCRIPTION. The default is 100.
        stepsize : TYPE, optional
            DESCRIPTION. The default is 0.1.
        seed : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        dict
            DESCRIPTION.

        """
        gt = self._gt_binary_map(scanpath_index)
        n_fix = int(gt.sum())
        if n_fix == 0:
            return {"auc_borji": np.nan}

        s_map = self._as_range_map(self.ref_sm, jitter=False, rng=None)

        # thresholds: fixed step size from 0..1
        # include 0 and 1 endpoints in ROC, but thresholds themselves can be interior
        thresholds = np.arange(0.0, 1.0 + 1e-12, float(stepsize))
        thresholds = np.unique(np.clip(thresholds, 0.0, 1.0))

        rng = np.random.default_rng(seed)
        H, W = s_map.shape
        n_pix = H * W

        # precompute fixated saliency values (for TP)
        fix_vals = s_map[gt == 1]

        aucs = []
        for _ in range(int(splits)):
            # sample negatives uniformly at random (as many as fixations)
            neg_idx = rng.integers(0, n_pix, size=n_fix, endpoint=False)
            neg_y = neg_idx // W
            neg_x = neg_idx % W
            neg_vals = s_map[neg_y, neg_x]

            fp = [0.0]
            tp = [0.0]

            for thr in thresholds:
                tp_rate = float(np.mean(fix_vals >= thr))
                fp_rate = float(np.mean(neg_vals >= thr))
                fp.append(fp_rate)
                tp.append(tp_rate)

            fp.append(1.0)
            tp.append(1.0)

            fp = np.asarray(fp, dtype=float)
            tp = np.asarray(tp, dtype=float)
            order = np.argsort(fp)
            aucs.append(float(np.trapz(tp[order], fp[order])))

        return {"auc_borji": float(np.mean(aucs))}


def scanpath_saliency_percentile(input, reference_map,
                                 **kwargs):
      
    scanpath_index = kwargs.get('scanpath_saliency_percentile_scanpath_index', 'all')
    if isinstance(input, SaliencyReference):
        results = input.scanpath_saliency_percentile(scanpath_index)
    else:
        perc_i = SaliencyReference(input,reference_map,
                                  **kwargs)
        results = perc_i.scanpath_saliency_percentile(scanpath_index)
        
    return results


def scanpath_saliency_nss(input, reference_map,
                          **kwargs):
    
    scanpath_index = kwargs.get('scanpath_saliency_nss_scanpath_index', 'all')
    sigma_kernel = kwargs.get(
        "scanpath_saliency_nss_sigma_kernel", .5
    )
    delta_neighborhood = kwargs.get(
        "scanpath_saliency_nss_delta", 1
    )
  
    if isinstance(input, SaliencyReference):
        results = input.scanpath_saliency_nss(
            scanpath_index=scanpath_index,
            sigma_kernel=sigma_kernel,
            delta_neighborhood=delta_neighborhood
        )
    else:
        nss_i = SaliencyReference(input,reference_map,
                                  **kwargs)
        results = nss_i.scanpath_saliency_nss(
            scanpath_index=scanpath_index,
            sigma_kernel=sigma_kernel,
            delta_neighborhood=delta_neighborhood
        )
    return results


def scanpath_saliency_information_gain(input, reference_map,
                          **kwargs):
    
    scanpath_index = kwargs.get('scanpath_saliency_ig_scanpath_index', 'all')
    baseline = kwargs.get('scanpath_saliency_ig_baseline', 'center_gaussian')
    baseline_sigma_x = kwargs.get(
        "scanpath_saliency_ig_sigma_x", None
    )
    baseline_sigma_y = kwargs.get(
        "scanpath_saliency_ig_sigma_y", None
    )
  
    if isinstance(input, SaliencyReference):
        results = input.scanpath_saliency_information_gain(
            scanpath_index=scanpath_index,
            baseline=baseline,
            baseline_sigma_x=baseline_sigma_x,
            baseline_sigma_y=baseline_sigma_y
        )
    else:
        ig_i = SaliencyReference(input,reference_map,
                                  **kwargs)
        results = ig_i.scanpath_saliency_information_gain(
            scanpath_index=scanpath_index,
            baseline=baseline,
            baseline_sigma_x=baseline_sigma_x,
            baseline_sigma_y=baseline_sigma_y
        )
    return results


def scanpath_saliency_auc_judd(input, reference_map,
                          **kwargs):
    
    scanpath_index = kwargs.get('scanpath_saliency_auc_judd_scanpath_index', 'all')
    jitter = kwargs.get(
        "scanpath_saliency_auc_judd_jitter", True
    )
    seed = kwargs.get(
        "scanpath_saliency_auc_judd_seed", None
    )
  
    if isinstance(input, SaliencyReference):
        results = input.scanpath_saliency_auc_judd(
            scanpath_index=scanpath_index,
            jitter=jitter,
            seed=seed
        )
    else:
        aucj_i = SaliencyReference(input,reference_map,
                                  **kwargs)
        results = aucj_i.scanpath_saliency_auc_judd(
            scanpath_index=scanpath_index,
            jitter=jitter,
            seed=seed
        )
    return results



def scanpath_saliency_auc_borji(input, reference_map,
                          **kwargs):
    
    scanpath_index = kwargs.get('scanpath_saliency_auc_borji_scanpath_index', 'all')
    splits = kwargs.get('scanpath_saliency_auc_borji_splits', 100)
    stepsize = kwargs.get(
        "scanpath_saliency_auc_borji_step_size", 0.1
    )
    seed = kwargs.get(
        "scanpath_saliency_auc_borji_seed", None
    )
  
    if isinstance(input, SaliencyReference):
        results = input.scanpath_saliency_auc_borji(
            scanpath_index=scanpath_index,
            splits=splits,
            stepsize=stepsize,
            seed=seed
        )
    else:
        aucb_i = SaliencyReference(input,reference_map,
                                  **kwargs)
        results = aucb_i.scanpath_saliency_auc_borji(
            scanpath_index=scanpath_index,
            splits=splits,
            stepsize=stepsize,
            seed=seed
        )
    return results






  