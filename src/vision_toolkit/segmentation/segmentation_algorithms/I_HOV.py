# -*- coding: utf-8 -*-

import numpy as np  
import matplotlib.pyplot as plt 

from vision_toolkit.utils.segmentation_utils import interval_merging, centroids_from_ints 
from vision_toolkit.utils.velocity_distance_factory import absolute_angular_distance

import time 
import joblib


def pre_process_IHOV(data_set, config):
    """
     
    Parameters
    ----------
    data_set : TYPE
        DESCRIPTION.
    config : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    assert config['distance_type'] == 'euclidean', "'Distance type' must be set to 'euclidean"
 
    start_time = time.time()
     # Extract generic data from the configuration dictionary
    # s_f: Sampling frequency (e.g., number of samples per second)
    # n_s: Number of samples in the dataset
    s_f = config['sampling_frequency']
    n_s = config['nb_samples']
    
    # Extract x and y coordinate arrays from the dataset
    # x_array and y_array are assumed to be 1D arrays of length n_s
    x_array = data_set['x_array']
    y_array = data_set['y_array']
    
    # Combine x and y coordinates into a 2D array of shape (2, n_s)
    # x_array and y_array are reshaped to (1, n_s) and concatenated along axis 0
    # Resulting coord has shape (2, n_s), where each column is [x, y] for a sample
    coord = np.concatenate((x_array.reshape(1, n_s),
                            y_array.reshape(1, n_s)), axis=0)
    
    # Compute the index interval corresponding to the duration threshold
    # IHOV_duration_threshold (in seconds) is multiplied by sampling frequency (s_f)
    # np.ceil ensures the result is rounded up to the nearest integer, giving the number of samples
    t_du = np.ceil(config['IHOV_duration_threshold'] * s_f)
    
    # Compute the number of samples to use around index i for averaging
    # IHOV_averaging_threshold (in seconds) is multiplied by sampling frequency
    # np.ceil rounds up to ensure at least the required number of samples
    t_av = np.ceil(config['IHOV_averaging_threshold'] * s_f)
    
    # Ensure t_av is odd for symmetric averaging (e.g., to have equal samples on both sides)
    # If t_av is even, increment it by 1
    if (t_av % 2) == 0:
        t_av += 1
    
    # Convert t_av to an integer for use in indexing
    t_av = int(t_av)
    
    # Extract the number of angular bins from the configuration
    # Used to divide 360 degrees into equal angular segments
    nb_ang_bin = config['IHOV_angular_bin_nbr']
    
    # Compute the size of each angular bin in degrees
    # 360 degrees divided by the number of bins gives the angular resolution
    ang_bin = 360 / nb_ang_bin
    
    # Compute vector matrix for window 1 (past window)
    # For each index i, compute vectors from the current coordinate (coord[:, i]) to
    # coordinates in a window of size t_du before i (from i-t_du to i-1)
    # np.maximum(0, ...) ensures indices don't go below 0 (start of array)
    # Result: w_1_m[i] is an array of vectors from coord[:, i] to previous coordinates
    w_1_m = np.array([coord[:, i].reshape(2, 1)
                      - coord[:, np.maximum(0, np.arange(i-t_du, i, dtype=int))]
                      for i in range(n_s)])
    
    # Compute vector matrix for window 2 (future window)
    # For each index i, compute vectors from the current coordinate (coord[:, i]) to
    # coordinates in a window of size t_du after i (from i+1 to i+t_du)
    # np.minimum(n_s-1, ...) ensures indices don't exceed the array length
    # Result: w_2_m[i] is an array of vectors from coord[:, i] to future coordinates
    w_2_m = np.array([coord[:, np.minimum(n_s-1, np.arange(i+1, i+t_du+1, dtype=int))]
                      - coord[:, i].reshape(2, 1) for i in range(n_s)])
    
    # Compute vector matrix for window 3 (past-to-future window)
    # For each index i, compute vectors from coordinates in a window of size t_du before i
    # (i-t_du to i-1) to coordinates in a window of size t_du after i (i+1 to i+t_du)
    # np.flip reverses the future indices to align them with past indices
    # Result: w_3_m[i] is an array of vectors from past coordinates to future coordinates
    w_3_m = np.array([coord[:, np.minimum(n_s-1, np.flip(np.arange(i+1, i+t_du+1, dtype=int)))]
                      - coord[:, np.maximum(0, np.arange(i-t_du, i, dtype=int))]
                      for i in range(n_s)])
    
    hists_w_1, hists_w_2, hists_w_3 = [], [], []
    
    
    def comp_hist(bin_, vect_):
        """
         
        Parameters
        ----------
        bin_ : TYPE
            DESCRIPTION.
        vect_ : TYPE
            DESCRIPTION.

        Returns
        -------
        n_hist : TYPE
            DESCRIPTION.

        """
        # Compute distances: the original implementation used velocity, but since
        # velocity histograms are re-normalized, distances yield the same result
        # np.linalg.norm computes the Euclidean norm (magnitude) of vectors along axis 0
        # vect_ is assumed to be a 2D array of shape (2, N), where each column is a vector [x, y]
        # dist_ is a 1D array of length N containing the magnitudes of the vectors
        dist_ = np.linalg.norm(vect_, axis=0)
        
        # Initialize two histograms with zeros, each of length nb_ang_bin
        # hist: Accumulates the sum of distances for each angular bin
        # n_hist: Stores the rotated and normalized histogram
        hist, n_hist = np.zeros(nb_ang_bin), np.zeros(nb_ang_bin)
        
        # Iterate over each angular bin (k ranges from 0 to nb_ang_bin-1)
        for k in range(nb_ang_bin):
            # Get indices where the angular bin array (bin_) equals the current bin index k
            # bin_ is assumed to be a 1D array of length N, where each element is an integer
            # from 0 to nb_ang_bin-1, representing the angular bin for each vector
            idx_ = np.where(bin_ == k)[0]
        
            # Sum the distances (from dist_) corresponding to the indices in idx_
            # hist[k] stores the total distance for vectors in the k-th angular bin
            hist[k] = np.sum(dist_[idx_])
        
        # Rotate the histogram to place the bin with the maximum distance at index 0
        # np.argmax(hist) finds the index of the bin with the largest total distance
        k = np.argmax(hist)
        
        # Calculate the shift (t_) needed to move the max bin to index 0
        # t_ is the number of positions to shift the histogram to the right
        t_ = nb_ang_bin - k
        
        # Perform the rotation:
        # Copy the histogram from index k to the end into the start of n_hist (0 to t_-1)
        n_hist[:t_] = hist[k:]
        # Copy the histogram from the start to index k-1 into the end of n_hist (t_ to end)
        n_hist[t_:] = hist[:k]
        
        # Normalize the rotated histogram so that its values sum to 1
        # This ensures n_hist represents a probability distribution
        n_hist /= np.sum(n_hist)
        
        # Return the normalized and rotated histogram
        return n_hist
    
    
    for i in range(n_s):
        
        # Compute histogram for window 1 
        l_w1 = w_1_m[i]
        l_bin_1 = np.floor(comp_angles(l_w1) / ang_bin)
        
        hists_w_1.append(comp_hist(l_bin_1,
                                   l_w1)) 
    
        # Compute histogram for window 2 
        l_w2 = w_2_m[i]
        l_bin_2 = np.floor(comp_angles(l_w2) / ang_bin)
        
        hists_w_2.append(comp_hist(l_bin_2,
                                   l_w2))
        
        # Compute histogram for window 3
        l_w3 = w_3_m[i]
        l_bin_3 = np.floor(comp_angles(l_w3) / ang_bin)
         
        hists_w_3.append(comp_hist(l_bin_3,
                                   l_w3))
        
    # For each angle interval, a moving average is performed along indexes
    # Then, averaged histograms for window 1 are re-normalized
    hists_w_1 = np.array(hists_w_1) 
    a_hists_w_1 = np.zeros_like(hists_w_1)
    
    for k in range(nb_ang_bin):
        a_hists_w_1[:,k] = averaging(hists_w_1[:,k], t_av)
  
    a_hists_w_1 /= np.sum(a_hists_w_1, axis = 1).reshape(n_s, 1) 
  
    # For each angle interval, a moving average is performed along indexes
    # Then, averaged histograms for window 2 are re-normalized
    hists_w_2 = np.array(hists_w_2) 
    a_hists_w_2 = np.zeros_like(hists_w_2)
    
    for k in range(nb_ang_bin):
        a_hists_w_2[:,k] = averaging(hists_w_2[:,k], t_av)
  
    a_hists_w_2 /= np.sum(a_hists_w_2, axis = 1).reshape(n_s, 1) 
    
    # For each angle interval, a moving average is performed along indexes
    # Then, averaged histograms for window 3 are re-normalized
    hists_w_3 = np.array(hists_w_3)
    a_hists_w_3 = np.zeros_like(hists_w_3)
    
    for k in range(nb_ang_bin):
        a_hists_w_3[:,k] = averaging(hists_w_3[:,k], t_av)
  
    a_hists_w_3 /= np.sum(a_hists_w_3, axis = 1).reshape(n_s, 1) 
    
    print("--- 1: %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    
    # Averaged and normalized histograms are concatened for each time index
    feat_ = np.concatenate((a_hists_w_1, a_hists_w_2, a_hists_w_3), axis = 1)
 
    return feat_
    
 
def comp_angles(v_mat):
    """
     
    Parameters
    ----------
    v_mat : TYPE
        DESCRIPTION.

    Returns
    -------
    dir_ : TYPE
        DESCRIPTION.

    """
    # Add a small constant (1e-10) to all elements of v_mat to avoid numerical instability
    # This prevents issues like division by zero when computing norms or angles
    v_mat += 1e-10
    
    # Initialize an array to store the direction (angle in degrees) for each vector
    # v_mat.shape[1] is the number of vectors (columns) in v_mat
    # dir_ is a 1D array of length N, initialized to zeros
    dir_ = np.zeros(v_mat.shape[1])
    
    # Identify vectors based on their y-component (v_mat[1,:])
    # _p: Boolean mask for vectors where y >= 0 (angles likely in [0, π] or [0°, 180°])
    # _m: Boolean mask for vectors where y < 0 (angles likely in (π, 2π] or (180°, 360°])
    _p = v_mat[1, :] >= 0
    _m = v_mat[1, :] < 0
    
    # Compute the Euclidean norm (magnitude) of vectors where y >= 0
    # v_mat[:, _p] selects columns of v_mat where _p is True
    # np.linalg.norm computes the norm along axis 0, resulting in a 1D array of norms
    n_p = np.linalg.norm(v_mat[:, _p], axis=0)
    
    # Compute the direction (angle in degrees) for vectors where y >= 0
    # np.arccos computes the angle θ = arccos(x / norm), where x is v_mat[0, :] (x-component)
    # np.divide(x, n_p, where=n_p > 0) avoids division by zero by only computing where norm > 0
    # (180/np.pi) converts radians to degrees
    # dir_[_p] stores angles in [0°, 180°] for vectors with y >= 0
    dir_[_p] = (180 / np.pi) * np.arccos(np.divide(v_mat[0, :][_p],
                                                   n_p,
                                                   where=n_p > 0))
    
    # Compute the Euclidean norm (magnitude) of vectors where y < 0
    # v_mat[:, _m] selects columns of v_mat where _m is True
    # np.linalg.norm computes the norm along axis 0, resulting in a 1D array of norms
    n_m = np.linalg.norm(v_mat[:, _m], axis=0)
    
    # Compute the direction (angle in degrees) for vectors where y < 0
    # For vectors with y < 0, angles are in (π, 2π] radians or (180°, 360°]
    # np.arccos(x / norm) gives an angle in [0, π]; we compute 2π - θ to get the full angle
    # np.divide(x, n_m, where=n_m > 0) avoids division by zero
    # (180/np.pi) converts radians to degrees
    # dir_[_m] stores angles in (180°, 360°] for vectors with y < 0
    dir_[_m] = (180 / np.pi) * (2 * np.pi - np.arccos(np.divide(v_mat[0, :][_m],
                                                                n_m,
                                                                where=n_m > 0)))
    
    # Return the array of angles (in degrees) for all vectors
    return dir_


def averaging(v_, t_av):
    """
     
    Parameters
    ----------
    v_ : TYPE
        DESCRIPTION.
    t_av : TYPE
        DESCRIPTION.

    Returns
    -------
    averaged : TYPE
        DESCRIPTION.

    """
    # Get the length of the input array v_
    # v_ is assumed to be a 1D NumPy array of length n_
    n_ = len(v_)
    
    # Compute half the averaging window size, rounded down
    # t_av is the total window size for averaging (assumed to be an odd integer from previous code)
    # h_t_av is used to adjust the convolution output to align with the original array
    h_t_av = int(t_av / 2)
    
    # Perform a moving average using convolution
    # np.ones(t_av) creates a 1D array of ones with length t_av (the averaging kernel)
    # np.convolve computes the convolution of v_ with the kernel, summing values in the window
    # Dividing by t_av normalizes the sum to compute the average
    # conv is a 1D array of length n_ + t_av - 1, containing the convolved (smoothed) values
    conv = np.convolve(v_, np.ones(t_av)) / t_av
    
    # Extract the central portion of the convolution result
    # The convolution output is longer than the input due to edge effects
    # Slicing from h_t_av to n_ + h_t_av selects the n_ elements aligned with the original v_
    # This removes the padded edges and returns an array of the same length as v_
    averaged = conv[h_t_av: n_ + h_t_av]
    
    # Return the averaged (smoothed) array
    return averaged
    

def process_IHOV(data_set, config):
    
    if config['verbose']:
        
        print("Processing HOV Identification...")  
        start_time = time.time()
    
    task = config['task']
 
    
    clf = joblib.load('vision_toolkit/segmentation/segmentation_algorithms/trained_models/{sm}/i_{cl}_{task}.joblib'.format(
                                        sm = config['segmentation_method'],
                                        cl = config['IHOV_classifier'],
                                        task = config['task']
                                        ))
    feature_mat = pre_process_IHOV(data_set, config) 
 
    preds = clf.predict(feature_mat)
    plt.plot(preds)
    plt.show()
    plt.clf() 
    
    if config['verbose']:
    
        print("Done")  
        print("--- Execution time: %s seconds ---" % (time.time() - start_time))   
  
    
    if task == 'binary':
        
        wi_fix = np.where(preds == 1)[0]
        i_fix = np.array([False]*config['nb_samples'])  
        i_fix[wi_fix] = True
	
        f_ints = interval_merging(wi_fix, 
                                  min_int_size = config['min_int_size'])
    
        x_a = data_set['x_array']
        y_a = data_set['y_array']
    
        ctrds = centroids_from_ints(f_ints,
                                    x_a, y_a)
 
        i_sac = i_fix == False
        wi_sac = np.where(i_sac == True)[0]

        s_ints = interval_merging(wi_sac,
                                  min_int_size = config['min_int_size'])
    

        return dict({
            'is_fixation': i_fix,
            'fixation_intervals': f_ints,
            'centroids': ctrds,
            'is_saccade': i_sac,
            'saccade_intervals': s_ints
                })
    
    elif task == "ternary":
        
        return dict({
            'is_fixation': preds == 1,
            'fixation_intervals': interval_merging(np.where((preds == 1) == True)[0],
                                                   min_int_size = config['min_int_size']),
            'is_saccade': preds == 2,
            'saccade_intervals': interval_merging(np.where((preds == 2) == True)[0],
                                                  min_int_size = config['min_int_size']),
            'is_pursuit': preds == 3,
            'pursuit_intervals': interval_merging(np.where((preds == 3) == True)[0],
                                                  min_int_size = config['min_int_size']),
            
            })  
    
    
   