# -*- coding: utf-8 -*-

 
from scipy.spatial.distance import cdist

from vision_toolkit.scanpath.similarity.c_comparison_algorithms import c_comparison_algorithms as c_comparison
from vision_toolkit.scanpath.similarity.distance_based.distance_based_base import DistanceBased
from vision_toolkit.visualization.scanpath.similarity.distance_based.elastic import plot_DTW_frechet


class FrechetDistance(DistanceBased):
    def __init__(self, input, config, id_1, id_2):
        super().__init__(input, dist_to=False)

        c_m = cdist(self.s_1[0:2].T, self.s_2[0:2].T, metric="euclidean")

        self.opt_links, self.dist_ = c_comparison.discrete_frechet(
            self.s_1[0:2], self.s_2[0:2], c_m
        )

        if config["display_results"]: 
            plot_DTW_frechet(self.s_1[0:2], self.s_2[0:2], self.opt_links, id_1, id_2)

 