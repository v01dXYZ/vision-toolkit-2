# -*- coding: utf-8 -*-
# Stores the result (points of convex hull)


import numpy as np

from vision_toolkit.visualization.scanpath.single.geometrical import plot_convex_hull

 

class ConvexHull:

    def __init__(self, scanpath):
        pts = np.asarray(scanpath.values[0:2]).T
        pts = np.unique(pts, axis=0)

        self.s_ = pts.T
        self.n_ = pts.shape[0]

        self.h_ = set()
        self.h_a = None
        self.area = None

        if self.n_ < 3:
            raise ValueError("Convex hull not possible: need at least 3 unique points.")

        self.comp_hull()
        self.results = {"hull_area": self.area, "hull_apex": self.h_a}

        if scanpath.config.get("display_results", False):
            plot_convex_hull(scanpath.values, self.h_a,
                             scanpath.config.get("display_path"),
                             scanpath.config)
            

    def comp_hull(self):
        
        s_ = self.s_.T
        s_x = np.argmax(s_[:, 0])
        i_x = np.argmin(s_[:, 0])

        self.quick_hull(s_[i_x], s_[s_x], 1)
        self.quick_hull(s_[i_x], s_[s_x], -1)

        h_a = np.array(list(self.h_), dtype=float)

        if h_a.shape[0] < 3:
            self.h_a = h_a
            self.area = 0.0
            return

        self.h_a = self.sort_coordinates(h_a)
        self.area = self.poly_area()
        

    def quick_hull(self, p1, p2, side):
        s_ = self.s_.T
        n_ = self.n_
        ind = -1
        m_d = 0

        for i in range(n_):
            l_d = self.line_dist(p1, p2, s_[i])
            if (self.find_side(p1, p2, s_[i]) == side) and (l_d > m_d):
                ind = i
                m_d = l_d

        if ind == -1:
            self.h_.add((round(p1[0], 6), round(p1[1], 6)))
            self.h_.add((round(p2[0], 6), round(p2[1], 6)))
            return

        self.quick_hull(s_[ind], p1, -self.find_side(s_[ind], p1, p2))
        self.quick_hull(s_[ind], p2, -self.find_side(s_[ind], p2, p1))


    def find_side(self, p1, p2, p):
        val = (p[1] - p1[1]) * (p2[0] - p1[0]) - (p2[1] - p1[1]) * (p[0] - p1[0])

        if val > 0:
            return 1
        if val < 0:
            return -1
        return 0


    def line_dist(self, p1, p2, p):
        return abs((p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0]))


    def poly_area(self):
        
        x_ = self.h_a[:, 0]
        
        y_ = self.h_a[:, 1]
        return 0.5 * np.abs(np.dot(x_, np.roll(y_, 1)) - np.dot(y_, np.roll(x_, 1)))


    def sort_coordinates(self, s_):
        
        x_m, y_m = s_.mean(0)
        x, y = s_.T 
        a_s = np.arctan2(y - y_m, x - x_m) 
        i_ = np.argsort(-a_s)

        return s_[i_]