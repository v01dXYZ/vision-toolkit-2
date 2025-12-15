# -*- coding: utf-8 -*-

 
import reference_image_mapper as rim
 
gaze_data = 'input/_gaze.csv'
world_timestamps = 'input/_world_timestamps.csv'
reference_image = 'input/_reference_image.jpg'
world_camera = 'input/_worldCamera.mp4'

rim.process_rim(gaze_data, world_timestamps, 
                world_camera, reference_image)
 
    
    
    
    
    
    
    
    
    
    