# -*- coding: utf-8 -*-

import vision_toolkit as v
import numpy as np
 

v.processing_rim(gaze_data = 'src/input_rim/_gaze.csv',
                 time_stamps = 'src/input_rim/_world_timestamps.csv',
                 reference_image = 'src/input_rim/_reference_image.jpg',
                 world_camera = 'src/input_rim/_worldCamera.mp4',
                 output_name = 'src/_gaze',
                 output_dir =   'src/mappedGazeOutput')