# -*- coding: utf-8 -*-

import numpy as np

from vision_toolkit.scanpath.similarity.distance_based.distance_based_base import DistanceBased
 

class TDEDistance(DistanceBased):
    def __init__(self, input, config, id_1, id_2):
        super().__init__(input)

        method = config.get("TDE_distance_method", "mean_minimal")  # "mean_minimal" or "hausdorff"
        k = int(config.get("TDE_distance_subsequence_length", 5))
        scaling = bool(config.get("TDE_distance_scaling", True))

        if k < 1 or k > min(self.n_1, self.n_2):
            self.dist_ = np.nan
            return

        s1 = self.s_1[0:2].astype(float)
        s2 = self.s_2[0:2].astype(float)

        if scaling:
            # normalisation simple (optionnelle) : mettre dans [0,1] approx via max coord
            L = max(np.max(s1), np.max(s2))
            if L > 0:
                s1 = s1 / L
                s2 = s2 / L

        # construit les sous-séquences de longueur k
        u1 = np.array([s1[:, i:i+k] for i in range(self.n_1 - k + 1)])  # shape: (N1-k+1, 2, k)
        u2 = np.array([s2[:, j:j+k] for j in range(self.n_2 - k + 1)])  # shape: (N2-k+1, 2, k)

        self.dist_ = float(self._tde_k(u1, u2, k, method))

    @staticmethod
    def _tde_k(u1, u2, k, method):
    
        d = np.empty(u1.shape[0], dtype=float)

        for i in range(u1.shape[0]):
            # distances à toutes les fenêtres de u2
            diffs = u2 - u1[i]                     # (N2-k+1, 2, k)
            norms = np.linalg.norm(diffs, axis=(1, 2))  # (N2-k+1,)
            d[i] = np.min(norms) / float(k)        # normalisation /k  

        if method == "hausdorff":
            return np.max(d)
        
        else:   
            return np.mean(d)
