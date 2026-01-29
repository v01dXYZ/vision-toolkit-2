# -*- coding: utf-8 -*-

import numpy as np
from scipy import signal

from vision_toolkit.utils.binning import spatial_bin
from vision_toolkit.visualization.scanpath.single.saliency import plot_saliency_map
from vision_toolkit.scanpath.scanpath_base import Scanpath
from vision_toolkit.segmentation.processing.binary_segmentation import BinarySegmentation


class SaliencyMap:
    
    def __init__(self, input, comp_saliency_map=True, **kwargs):
        
        verbose = kwargs.get("verbose", True)
        display_results = kwargs.get("display_results", True)
        display_path = kwargs.get("display_path", None)
        
        map_type = kwargs.get("saliency_map_type", 'saliency_map')

        std = kwargs.get("scanpath_saliency_gaussian_std", 3)
        k_l = kwargs.get("scanpath_saliency_kernel_length", 50)
        pixel_number_x = kwargs.get("scanpath_saliency_pixel_number_x", 100)
        pixel_number_y = kwargs.get("scanpath_saliency_pixel_number_y", None)

        # Build scanpaths list
        if isinstance(input, list):
            if isinstance(input[0], str):
                scanpaths = [Scanpath.generate(inp, **kwargs) for inp in input]
            elif isinstance(input[0], BinarySegmentation):
                scanpaths = [Scanpath.generate(inp, **kwargs) for inp in input]
            elif isinstance(input[0], Scanpath):
                scanpaths = input
            else:
                raise ValueError("Input must be a list of Scanpath, BinarySegmentation or csv")
        else:
            if isinstance(input, str):
                scanpaths = [Scanpath.generate(input, **kwargs)]
            elif isinstance(input, BinarySegmentation):
                scanpaths = [Scanpath.generate(input, **kwargs)]
            elif isinstance(input, Scanpath):
                scanpaths = [input]
            else:
                raise ValueError("Input must be a Scanpath, a BinarySegmentation, or a csv")

        self.scanpaths = scanpaths
        self.size_plan_x = self.scanpaths[0].config["size_plan_x"]
        self.size_plan_y = self.scanpaths[0].config["size_plan_y"]

        # Keep aspect ratio if y not provided
        ratio = self.size_plan_x / self.size_plan_y
        if pixel_number_y is None:
            pixel_number_y = int(round(pixel_number_x / ratio))

        # Force odd sizes (optional, but fine)
        self.p_n_x = pixel_number_x + 1 if (pixel_number_x % 2) == 0 else pixel_number_x
        self.p_n_y = pixel_number_y + 1 if (pixel_number_y % 2) == 0 else pixel_number_y

        self.std = std
        self.k_l = k_l + 1 if (k_l % 2) == 0 else k_l  # force odd kernel

        self.scanpaths[0].config.update(
            dict(
                scanpath_saliency_gaussian_std=std,
                scanpath_saliency_kernel_length=self.k_l,
                scanpath_saliency_pixel_number_x=self.p_n_x,
                scanpath_saliency_pixel_number_y=self.p_n_y,
                scanpath_saliency_map_type=map_type,
                verbose=verbose,
                display_results=display_results,
                display_path=display_path,
            )
        )

        # Sanity checks
        for sp in self.scanpaths:
            assert sp.config["size_plan_x"] == self.size_plan_x, 'All recordings must have the same "size_plan_x"'
            assert sp.config["size_plan_y"] == self.size_plan_y, 'All recordings must have the same "size_plan_y"'

        self.map_type = map_type
        self.saliency_map = None

        if comp_saliency_map:
            self.saliency_map = self.comp_map(self.scanpaths, map_type)

            if display_results:
                plot_saliency_map(self.saliency_map, display_path)

        self.scanpaths[0].verbose()


    def _gkern(self):
    
        try:
            from scipy.signal.windows import gaussian as _gauss
        except ImportError:
            from scipy.signal import gaussian as _gauss

        g1d = _gauss(self.k_l, self.std).reshape(self.k_l, 1)
        g2d = np.outer(g1d, g1d)
        s = g2d.sum()
        
        return g2d / s if s > 0 else g2d

    def _bin_xy(self, sp):
        
        seq = sp.values
        # spatial_bin returns integer bins already; shape (2, n_fix)
        return spatial_bin(seq[0:2], self.p_n_x, self.p_n_y, self.size_plan_x, self.size_plan_y)


    def comp_map(self, scanpaths, map_type):
        """
        Returns a (p_n_y, p_n_x) normalized map (sum = 1 when possible).
        """
        kern = self._gkern()

        if map_type == "relative_duration_saliency_map":
            # Per scanpath normalize, then mean
            maps = []
            for sp in scanpaths:
                xy = self._bin_xy(sp)
                dur = np.asarray(sp.values[2], dtype=float)

                m = np.zeros((self.p_n_y, self.p_n_x), dtype=float)
                xs = xy[0].astype(int)
                ys = xy[1].astype(int)

                mask = (
                    (0 <= xs) & (xs < self.p_n_x) &
                    (0 <= ys) & (ys < self.p_n_y) &
                    np.isfinite(dur) & (dur > 0)
                )
                np.add.at(m, (ys[mask], xs[mask]), dur[mask])

                tot = m.sum()
                m = (m / tot) if tot > 0 else m
                maps.append(m)

            f_m = np.mean(maps, axis=0) if maps else np.zeros((self.p_n_y, self.p_n_x), dtype=float)

        else:
            # Aggregate (sum) then normalize at the end
            f_m = np.zeros((self.p_n_y, self.p_n_x), dtype=float)

            for sp in scanpaths:
                xy = self._bin_xy(sp)
                xs = xy[0].astype(int)
                ys = xy[1].astype(int)

                if map_type == "saliency_map":
                    w = np.ones(xs.shape[0], dtype=float)
                    w_mask = np.isfinite(w)
                elif map_type == "absolute_duration_saliency_map":
                    w = np.asarray(sp.values[2], dtype=float)
                    w_mask = np.isfinite(w) & (w > 0)
                else:
                    raise ValueError(
                        'map_type must be one of: "saliency_map", "absolute_duration_saliency_map", "relative_duration_saliency_map"'
                    )

                mask = (
                    (0 <= xs) & (xs < self.p_n_x) &
                    (0 <= ys) & (ys < self.p_n_y) &
                    w_mask
                )
                np.add.at(f_m, (ys[mask], xs[mask]), w[mask])
        
        tot = f_m.sum()
        if tot > 0:
            f_m /= tot
            
        s_m = signal.convolve2d(f_m, kern, mode="same", boundary="fill", fillvalue=0)
 
        total = s_m.sum()
        if total > 0:
            s_m = s_m / total

        return s_m


def scanpath_saliency_map(input, **kwargs):
    
    kwargs.update({'saliency_map_type': 'saliency_map'})
    sm = SaliencyMap(input, **kwargs)
    
    return {"saliency_map": sm.saliency_map}


def scanpath_absolute_duration_saliency_map(input, **kwargs):
    
    kwargs.update({'saliency_map_type': 'absolute_duration_saliency_map'})
    sm = SaliencyMap(input, **kwargs)
    
    return {"absolute_duration_saliency_map": sm.saliency_map}


def scanpath_relative_duration_saliency_map(input, **kwargs):
    
    kwargs.update({'saliency_map_type': 'relative_duration_saliency_map'})
    sm = SaliencyMap(input, **kwargs)
    
    return {"relative_duration_saliency_map": sm.saliency_map}






















