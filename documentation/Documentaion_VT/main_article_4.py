# -*- coding: utf-8 -*-

import vision_toolkit as v
import os
import numpy as np

data = 'dataset/nat006.csv' 
image_ref = 'dataset/nat006.bmp'

bs = v.BinarySegmentation(data, 
                          sampling_frequency = 500,  
                          segmentation_method = 'I_HMM',
                          distance_type = 'euclidean',                        
                          display_segmentation = False,
                          verbose=False,
                          size_plan_x = 921,
                          size_plan_y = 630,  
                          )

sc = v.Scanpath(bs, 
                ref_image=image_ref,
                display_scanpath=True,
                verbose=False)




aoi_s = v.AoISequence(bs, 
                      ref_image=image_ref,
                      AoI_identification_method='I_AP',  
                      verbose=False)