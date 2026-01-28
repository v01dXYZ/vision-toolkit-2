#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 14:34:35 2023

@author: quentinlaborde
"""

import numpy as np
from scipy.spatial import Voronoi
from scipy.stats import gamma as gamma_dist
from scipy.optimize import minimize_scalar
from scipy.special import gammaln
from shapely.geometry import Polygon

from vision_toolkit.visualization.scanpath.single.geometrical import plot_voronoi_cells



class VoronoiCells:
    def __init__(self, scanpath):
       
        self.x_size = scanpath.config["size_plan_x"]
        self.y_size = scanpath.config["size_plan_y"]
 
        self.fixations = np.asarray(scanpath.values[0:2])
 
        self.areas, self.new_vertices = self.comp_voronoi_areas()
 
        skewness = self.comp_skewness()
        gamma_param = self.comp_gamma()

        self.results = dict(
            {
                "skewness": skewness,
                "gamma_parameter": gamma_param,
                "voronoi_areas": self.areas,
            }
        )

        if scanpath.config.get("display_results", False):
            plot_voronoi_cells(
                scanpath.values,
                self.new_vertices,
                scanpath.config.get("display_path"),
                scanpath.config,
            )


    def comp_skewness(self):
        
        areas = np.asarray(self.areas, dtype=float)
        areas = areas[areas > 0]
        if areas.size < 2:
            return np.nan
    
        mu = np.mean(areas)
        sigma = np.std(areas, ddof=0)  # ddof=0 pour coller à la formule "1/N"
        if sigma == 0 or not np.isfinite(sigma):
            return np.nan
    
        z3 = ((areas - mu) / sigma) ** 3
        
        return float(np.mean(z3)) 


    def comp_gamma(self):
        
        x = np.asarray(self.areas, dtype=float)
        x = x[x > 0]
        if x.size == 0:
            return np.nan
    
        # Normalisation (papier): mean = 1
        m = np.mean(x)
        if m <= 0 or not np.isfinite(m):
            return np.nan
        x = x / m
    
        # Log-vraisemblance pour Gamma(shape=1/b, scale=b), loc=0
        def neg_ll(b):
            if b <= 0 or not np.isfinite(b):
                return np.inf
            k = 1.0 / b          # shape
            theta = b            # scale
            # logpdf sum: (k-1)log x - x/theta - k log theta - log Gamma(k)
            return -np.sum((k - 1) * np.log(x) - (x / theta) - k * np.log(theta) - gammaln(k))
    
        try:
            res = minimize_scalar(neg_ll, bounds=(1e-6, 1e3), method="bounded")
            return float(res.x) if res.success else np.nan
        
        except Exception:
            return np.nan
 
    
    def comp_voronoi_areas(self):
        
        pts = self.fixations.T  # (N, 2)
        x_size, y_size = self.x_size, self.y_size
 
        if pts.shape[0] == 0:
            return [], []

        pts_unique = np.unique(pts, axis=0)
 
        if pts_unique.shape[0] < 3:
            return [], []

        try:
            vor = Voronoi(pts_unique)
        except Exception:
          
            return [], []

        regions, vertices = self.voronoi_finite_polygons_2d(
            vor, radius=max(x_size, y_size) ** 2
        )
 
        coords = [[0.0, 0.0], [x_size, 0.0], [x_size, y_size], [0.0, y_size]]
        mask = Polygon(coords)

        new_vertices = []
        cell_areas = []

        for region in regions:
            polygon = vertices[region]
            poly_voronoi = Polygon(polygon)

            if poly_voronoi.is_empty:
                # Cellule dégénérée
                cell_areas.append(0.0)
                new_vertices.append(np.empty((0, 2)))
                continue

            c_polygon = poly_voronoi.intersection(mask)

            if c_polygon.is_empty:
                cell_areas.append(0.0)
                new_vertices.append(np.empty((0, 2)))
                continue
 
            cell_areas.append(c_polygon.area)
 
            boundary = c_polygon.exterior
            xs = boundary.coords.xy[0][:-1]
            ys = boundary.coords.xy[1][:-1]
            poly = np.array(list(zip(xs, ys)))

            new_vertices.append(poly)

        return cell_areas, new_vertices


    def voronoi_finite_polygons_2d(self, vor, radius):
       
        if vor.points.shape[1] != 2:
            raise ValueError("Requires 2D input")

        new_regions = []
        new_vertices = vor.vertices.tolist()
        center = vor.points.mean(axis=0)
 
        all_ridges = {}

        for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
            all_ridges.setdefault(p1, []).append((p2, v1, v2))
            all_ridges.setdefault(p2, []).append((p1, v1, v2))

        # Reconstruction des régions infinies
        for p1, region in enumerate(vor.point_region):
            vertices = vor.regions[region]
            if all(v >= 0 for v in vertices):
              
                new_regions.append(vertices)
                continue
 
            ridges = all_ridges[p1]
            new_region = [v for v in vertices if v >= 0]

            for p2, v1, v2 in ridges:
                if v2 < 0:
                    v1, v2 = v2, v1

                if v1 >= 0: 
                    continue
 
                t = vor.points[p2] - vor.points[p1]  # tangent
                norm = np.linalg.norm(t)
                if norm == 0 or not np.isfinite(norm):
                    continue
                t /= norm
                n = np.array([-t[1], t[0]])  # normale

                midpoint = vor.points[[p1, p2]].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                far_point = vor.vertices[v2] + direction * radius

                new_region.append(len(new_vertices))
                new_vertices.append(far_point.tolist())
 
            vs = np.asarray([new_vertices[v] for v in new_region])
            c = vs.mean(axis=0)

            angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
            new_region = np.array(new_region)[np.argsort(angles)]

            new_regions.append(new_region.tolist())

        return new_regions, np.asarray(new_vertices)