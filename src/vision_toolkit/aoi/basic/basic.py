# -*- coding: utf-8 -*-

import copy

import numpy as np

from vision_toolkit.aoi.aoi_base import AoISequence
from vision_toolkit.scanpath.scanpath_base import Scanpath
from vision_toolkit.segmentation.processing.binary_segmentation import BinarySegmentation


class AoIBasicAnalysis:
    def __init__(self, input, **kwargs):
        """


        Parameters
        ----------
        input : str | BinarySegmentation | Scanpath | AoISequence
            DESCRIPTION.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        verbose = kwargs.get("verbose", True)

        if verbose:
            print("Processing Markov Based Analysis...\n")

        if isinstance(input, str):
            self.aoi_sequence = AoISequence.generate(input, **kwargs)

        elif isinstance(input, BinarySegmentation):
            self.aoi_sequence = AoISequence.generate(input, **kwargs)

        elif isinstance(input, AoISequence):
            self.aoi_sequence = input

        elif isinstance(input, Scanpath):
            self.aoi_sequence = AoISequence.generate(input, **kwargs)
        else:
            raise ValueError(
                "Input must be a csv, or a BinarySegmentation, or a Scanpath, or an AoISequence object"
            )

    def AoI_count(self):
        """


        Returns
        -------
        None.

        """

        ct = len(list(self.aoi_sequence.centers.keys()))
        result = dict({"count": ct})

        return result

    def AoI_duration(self, get_raw=True):
        
        seq = np.asarray(self.aoi_sequence.sequence, dtype=object)          
        dur = np.asarray(self.aoi_sequence.durations, dtype=np.float64)    
    
        if seq.shape[0] != dur.shape[0]:
            raise ValueError(
                f"sequence and durations must have the same length "
                f"(got {seq.shape[0]} vs {dur.shape[0]})"
            )
 
        centers = self.aoi_sequence.centers
        aoi_labels = list(centers.keys()) if isinstance(centers, dict) else list(np.unique(seq))
 
        t_dur = np.array(
            [np.nansum(dur[seq == lab]) if np.any(seq == lab) else 0.0 for lab in aoi_labels],
            dtype=np.float64
        )
    
        total = np.nansum(t_dur)
        prop_ = (t_dur / total) if total > 0 else np.zeros_like(t_dur)
    
        results = {
            "average_duration": float(np.nanmean(t_dur)) if t_dur.size else np.nan,
            "variance_duration": float(np.nanstd(t_dur)) if t_dur.size else np.nan,
            "raw": t_dur,
            "proportion": prop_,
        }
    
        if not get_raw:
            del results["raw"]
    
        return results


    def AoI_BCEA(self, BCEA_probability=0.68, get_raw=True):
        '''
        

        Parameters
        ----------
        BCEA_probability : TYPE
            DESCRIPTION.
        get_raw : TYPE, optional
            DESCRIPTION. The default is True.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        results : TYPE
            DESCRIPTION.

        '''
        seq = np.asarray(self.aoi_sequence.sequence, dtype=object)   # ex: "A","B",...
        positions = np.asarray(self.aoi_sequence.values[:2], dtype=np.float64)  # shape (2, N)
    
        if positions.shape[1] != seq.shape[0]:
            raise ValueError(
                f"positions and sequence must have the same length "
                f"(got {positions.shape[1]} vs {seq.shape[0]})"
            )
    
        centers = self.aoi_sequence.centers
        aoi_labels = list(centers.keys()) if isinstance(centers, dict) else list(np.unique(seq))
    
        bcea_s = []
        for lab in aoi_labels:
            idx = np.where(seq == lab)[0]
            if idx.size == 0:
                continue  
            l_pos = positions[:, idx]
            l_bcea = self.BCEA(l_pos[0], l_pos[1], BCEA_probability)
            bcea_s.append(l_bcea)
    
        bcea_s = np.asarray(bcea_s, dtype=np.float64)
     
        if bcea_s.size == 0:
            med_ = np.nan
            e_med = np.nan
        else:
            med_ = np.nanmedian(bcea_s)
            e_med = float(np.nansum(np.abs(bcea_s - med_)) / bcea_s.size)
    
        results = {
            "average_BCEA": float(np.nanmean(bcea_s)) if bcea_s.size else np.nan,
            "disp_BCEA": e_med,
            "raw": bcea_s,
        }
    
        if not get_raw:
            del results["raw"]
      
        return results



    def AoI_weighted_BCEA(self, BCEA_probability=0.68):
        '''
        

        Parameters
        ----------
        BCEA_probability : TYPE, optional
            DESCRIPTION. The default is 0.68.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        dict
            DESCRIPTION.

        '''
        seq = np.asarray(self.aoi_sequence.sequence, dtype=object)                
        pos = np.asarray(self.aoi_sequence.values[:2], dtype=np.float64)          
        dur = np.asarray(self.aoi_sequence.durations, dtype=np.float64)           
    
        if pos.shape[1] != seq.shape[0] or dur.shape[0] != seq.shape[0]:
            raise ValueError(
                f"Inconsistent lengths: positions={pos.shape[1]}, "
                f"sequence={seq.shape[0]}, durations={dur.shape[0]}"
            )
    
        centers = self.aoi_sequence.centers
        aoi_labels = list(centers.keys()) if isinstance(centers, dict) else list(np.unique(seq))
    
        weighted_sum = 0.0
        total_dur = 0.0
    
        for lab in aoi_labels:
            idx = np.where(seq == lab)[0]
            if idx.size == 0:
                continue
    
            l_pos = pos[:, idx]
            l_dur = float(np.nansum(dur[idx]))
            l_bcea = float(self.BCEA(l_pos[0], l_pos[1], BCEA_probability))
    
            weighted_sum += l_bcea * l_dur
            total_dur += l_dur
    
        avg_weighted = (weighted_sum / total_dur) if total_dur > 0 else np.nan
    
        return {"average_weighted_BCEA": float(avg_weighted)}


    def BCEA(self, x_a, y_a, probability):
        """


        Parameters
        ----------
        scanpath : TYPE
            DESCRIPTION.
        p : TYPE
            DESCRIPTION.

        Returns
        -------
        p_c : TYPE
            DESCRIPTION.

        """

        def pearson_corr(x, y):
            x = np.asarray(x)
            y = np.asarray(y)

            mx = x.mean()
            my = y.mean()

            xm, ym = x - mx, y - my

            _num = np.sum(xm * ym)
            _den = np.sqrt(np.sum(xm**2) * np.sum(ym**2))
            if _den < 1e-20:
                p_c = 0
            else:
                p_c = _num / _den

            ## For some small artifact of floating point arithmetic.
            p_c = max(min(p_c, 1.0), -1.0)

            return p_c

        k = -np.log(1 - probability)
        p_c = pearson_corr(x_a, y_a)
        sd_x = np.std(x_a, ddof=1)
        sd_y = np.std(y_a, ddof=1)

        bcea = 2 * np.pi * k * sd_x * sd_y * np.sqrt(1 - p_c**2)

        return bcea
