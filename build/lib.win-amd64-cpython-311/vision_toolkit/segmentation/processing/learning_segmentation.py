# -*- coding: utf-8 -*-
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import numpy as np
 


from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from vision_toolkit.segmentation.segmentation_algorithms.I_HOV import pre_process_IHOV, process_IHOV 
from vision_toolkit.segmentation.segmentation_algorithms.I_CNN import pre_process_ICNN, process_ICNN, CNN1D, train_cnn1d




class LearningSegmentation():
        
    def __init__(self, input_dfs, event_dfs, 
                 sampling_frequency, segmentation_method, task, 
                 **kwargs):
        
        
        
        
        if segmentation_method == 'I_HOV':
            kwargs.update({'distance_type': 'euclidean'})
            
        else:
            kwargs.update({'distance_type': 'angular'})
        
        
        config = dict({
            'sampling_frequency': sampling_frequency,
            'segmentation_method': segmentation_method,
            'task': task,
            'distance_projection': kwargs.get('distance_projection'),  
            'distance_type': kwargs.get('distance_type', 'euclidean'), 
            'IHOV_classifier': kwargs.get('IHOV_classifier', 'rf'),
            'size_plan_x': kwargs.get('size_plan_x'),
            'size_plan_y': kwargs.get('size_plan_y'),
            'smoothing': kwargs.get('smoothing', 'savgol'),
            'min_int_size': kwargs.get('min_int_size', 2),
            'display_results': kwargs.get('display_results', True),
            'display_segmentation': kwargs.get('display_segmentation', False),
            'verbose': kwargs.get('verbose', True)
                       })
        
        
        if segmentation_method == 'I_HOV':
            
            assert config['distance_type']=='euclidean', "Distance type must be 'euclidean'"
            config.update({
                'IHOV_duration_threshold': kwargs.get('IHOV_duration_threshold', 
                                                      0.2),
                'IHOV_averaging_threshold': kwargs.get('IHOV_averaging_threshold', 
                                                      0.2),
                'IHOV_angular_bin_nbr': kwargs.get('IHOV_angular_bin_nbr', 
                                                      36)})
        elif segmentation_method == 'I_CNN':
            
            config.update({
                'ICNN_temporal_window_size': kwargs.get('ICNN_temporal_window_size', 250),
                'ICNN_num_epochs': kwargs.get('ICNN_num_epochs', 25),
                'ICNN_batch_size': kwargs.get('ICNN_batch_size', 1024),
                'ICNN_learning_rate': kwargs.get('ICNN_learning_rate', 0.001), 
                })
                
        if event_dfs is not None:       
            labels = []
            for label in event_dfs:
                local = pd.read_csv(label)  
                labels += list(local['event_label'])
            labels = np.array(labels)  
            t_k = labels>0
            self.labels = labels[t_k]
            
            gaze_x, gaze_y = [], []
            for input_ in input_dfs:
                local = pd.read_csv(input_) 
                gaze_x += list(local['gazeX'])
                gaze_y += list(local['gazeY'])
                
            self.dataset = dict({'x_array': np.array(gaze_x)[t_k],
                                 'y_array': np.array(gaze_y)[t_k]})
       
            config.update({'nb_samples': len(self.dataset['x_array'])})
            self.config = config
          
            
        else:
            gaze_x, gaze_y = [], []
            for input_ in input_dfs:
                local = pd.read_csv(input_) 
                gaze_x += list(local['gazeX'])
                gaze_y += list(local['gazeY'])
                
            self.dataset = dict({'x_array': np.array(gaze_x),
                                 'y_array': np.array(gaze_y)})
       
            config.update({'nb_samples': len(self.dataset['x_array'])})
            self.config = config
            
            
        self.IHOV_classifier = dict({
            'rf': RandomForestClassifier(max_depth=10, max_features='sqrt'),
            'svm': SVC(), 
            'knn': KNeighborsClassifier(n_neighbors=3)                 
                })
            
        
        
    @classmethod
    def fit(cls, input_dfs, event_dfs, 
                 sampling_frequency, segmentation_method, task,
                 **kwargs):
        
        mt = cls(input_dfs, event_dfs, 
                 sampling_frequency, segmentation_method, task,
                 **kwargs)
 
        if segmentation_method == 'I_HOV': 
            train_features = pre_process_IHOV(mt.dataset, 
                                              mt.config)
            train_labels = mt.labels
            
            clf = mt.IHOV_classifier[mt.config['IHOV_classifier']]
            clf.fit(train_features, train_labels)
            
            path = 'vision_toolkit/segmentation/segmentation_algorithms/trained_models/{sm}/i_{cl}_{task}.joblib'.format(
                                                sm = segmentation_method,
                                                cl = mt.config['IHOV_classifier'],
                                                task = mt.config['task']
                                                )
            joblib.dump(clf, path)
      
        elif segmentation_method == 'I_CNN':  
            train_cnn1d(mt.dataset, 
                        mt.labels,
                        mt.config)
               
      
        
      
            
    @classmethod
    def predict(cls, input_dfs,  
                 sampling_frequency, segmentation_method, task,
                 **kwargs):
        
        mt = cls(input_dfs, None, 
                 sampling_frequency, segmentation_method, task,
                 **kwargs) 
        
        
        if segmentation_method == 'I_HOV':  
            segmentation_results = process_IHOV(mt.dataset, 
                                                mt.config)
            
        if segmentation_method == 'I_CNN':
            segmentation_results = process_ICNN(mt.dataset,
                                                mt.config)
            
            
          
            
            
            
            
            
            
            
            
            
            
            