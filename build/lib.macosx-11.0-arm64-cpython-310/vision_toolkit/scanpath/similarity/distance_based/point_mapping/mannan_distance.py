# -*- coding: utf-8 -*-

import numpy as np

from vision_toolkit.scanpath.similarity.distance_based.distance_based_base import DistanceBased
from vision_toolkit.visualization.scanpath.similarity.distance_based.point_mapping import plot_mannan_eye_analysis


class MannanSimilarity(DistanceBased):
    def __init__(self, input, config, id_1, id_2):
        super().__init__(input)

        nb_r = config["mannan_similarity_nb_random_scanpaths"]
 
        self.p_x = config["size_plan_x"]
        self.p_y = config["size_plan_y"]

        self.ol_1 = None
        self.ol_2 = None

        ## Compute Mannan distance for scanpaths seq_1 and seq_2
        d_u = self.mannan(self.s_1[0:2], self.s_2[0:2], keep=True)

        ## Compute Mannan distance for random scanpath vector
        ## of respective sizes n_1 and n_2
        d_r = np.zeros(nb_r)

        for i in range(nb_r):
            r1 = np.random.uniform(
                low=[0, 0], high=[self.p_x, self.p_y], size=(self.n_1, 2)
            )
            r2 = np.random.uniform(
                low=[0, 0], high=[self.p_x, self.p_y], size=(self.n_2, 2)
            )
        
            # Point 1 of the paper : random scanpaths aléatoires start at the center
            r1[0] = [self.p_x / 2.0, self.p_y / 2.0]
            r2[0] = [self.p_x / 2.0, self.p_y / 2.0]
        
            d_r[i] = self.mannan(r1.T, r2.T)

        ## Compute the Mannan Index of Similarity
        Dr = float(np.mean(d_r))
        self.sim_ = float("nan") if Dr <= 0 else 100.0 * (1.0 - d_u / Dr)

        if config["display_results"]:
            plot_mannan_eye_analysis(
                self.s_1, self.s_2, self.ol_1, self.ol_2, id_1, id_2
            )

    def mannan(self, s_1, s_2, keep=False):
        
        n_1 = s_1.shape[1]
        n_2 = s_2.shape[1]

        d_1, ol_1 = self.compute_mapping(s_1, s_2, n_1, n_2)
        d_2, ol_2 = self.compute_mapping(s_2, s_1, n_2, n_1)

        if keep:
            self.ol_1 = ol_1
            self.ol_2 = ol_2

        denom = np.sqrt(2 * n_1 * n_2 * (self.p_x**2 + self.p_y**2))
        num = np.sqrt(n_1 * np.sum(d_1**2) + n_2 * np.sum(d_2**2))

        return num / denom
