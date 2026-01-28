# -*- coding: utf-8 -*-

import vision_toolkit as v



root = 'dataset/'

 
sp1 = v.Scanpath(root + 'data_1.csv', 
                sampling_frequency = 256,                  
                segmentation_method = 'I_HMM',
                distance_type = 'angular',                        
                display_segmentation = True, 
                size_plan_x = 1200,
                size_plan_y = 800, 
                display_scanpath=True, 
                verbose=False)

sp2 = v.Scanpath(root + 'data_2.csv', 
                sampling_frequency = 256,                  
                segmentation_method = 'I_HMM',
                distance_type = 'angular',                        
                display_segmentation = False, 
                size_plan_x = 1200,
                size_plan_y = 800, 
                display_scanpath=True, 
                verbose=False)

print(v.scanpath_saliency_map([sp1,sp2]))
print(v.scanpath_absolute_duration_saliency_map([sp1,sp2]))
print(v.scanpath_relative_duration_saliency_map([sp1,sp2]))

  