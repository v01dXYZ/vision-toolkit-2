# -*- coding: utf-8 -*-

import vision_toolkit as v



root = 'dataset/'

 
sp = v.Scanpath(root + 'data_1.csv', 
                sampling_frequency = 256,  
                segmentation_method = 'I_HMM',
                distance_type = 'angular',                        
                display_segmentation = True,
                size_plan_x = 1200,
                size_plan_y = 800,
                display_scanpath=True)

 
 
 
ga = v.GeometricalAnalysis(root + 'data_1.csv', 
                          sampling_frequency = 256, 
                          segmentation_method = 'I_HMM',
                          distance_type = 'angular',                        
                          display_segmentation = True,
                          display_scanpath=True,
                          size_plan_x = 1200,
                          size_plan_y = 800)
 
 
ga = v.GeometricalAnalysis(sp)

 
print(v.scanpath_BCEA(sp,
                    display_results=True))
print(v.scanpath_convex_hull(sp,
                    display_results=True))
print(v.scanpath_HFD(sp,
                    display_results=True))
print(v.scanpath_k_coefficient(sp,
                    display_results=True))
print(v.scanpath_length(sp,
                    display_results=True))
print(v.scanpath_voronoi_cells(sp,
                    display_results=True))