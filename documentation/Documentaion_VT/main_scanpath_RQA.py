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
                display_scanpath=True)


rqa_rr = v.scanpath_RQA_recurrence_rate(sp1,
                                        scanpath_RQA_distance_threshold = 280)
rqa_det = v.scanpath_RQA_determinism(sp1,
                                        scanpath_RQA_distance_threshold = 280)
rqa_lam = v.scanpath_RQA_laminarity(sp1,
                                        scanpath_RQA_distance_threshold = 280)
rqa_ent = v.scanpath_RQA_entropy(sp1,
                                        scanpath_RQA_distance_threshold = 280)
rqa_CORM = v.scanpath_RQA_CORM(sp1,
                                        scanpath_RQA_distance_threshold = 280)