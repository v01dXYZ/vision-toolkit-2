# -*- coding: utf-8 -*-
 
import numpy as np 
import torch
from torch import nn
import torch.optim as optim

from torch.nn import Module
from torch.nn import Conv2d, Conv1d
from torch.nn import Linear
from torch.nn import MaxPool2d, MaxPool1d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten
from torch.optim import Adam
 
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader 

from keras.datasets import mnist 
from sklearn.metrics import classification_report
 
import matplotlib.pyplot as plt 
import time

from vision_toolkit.utils.segmentation_utils import interval_merging
from vision_toolkit.utils.segmentation_utils import centroids_from_ints




class CNN1D(nn.Module):
    def __init__(self, numChannels, classes, input_length):
        super().__init__()

        self.conv1 = Conv1d(in_channels=numChannels, out_channels=16, kernel_size=10)
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool1d(kernel_size=2)

        self.conv2 = Conv1d(in_channels=16, out_channels=32, kernel_size=5)
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool1d(kernel_size=2)

        self.conv3 = Conv1d(in_channels=32, out_channels=64, kernel_size=2)
        self.relu3 = ReLU()
        self.maxpool3 = MaxPool1d(kernel_size=2)

        self.conv4 = Conv1d(in_channels=64, out_channels=128, kernel_size=2)
        self.relu4 = ReLU()
        self.maxpool4 = MaxPool1d(kernel_size=2)

        self.conv5 = Conv1d(in_channels=128, out_channels=256, kernel_size=2)
        self.relu5 = ReLU()
        self.maxpool5 = MaxPool1d(kernel_size=2)

        #  Dummy input to compute flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, numChannels, input_length)
            x = self.forward_conv(dummy_input)
            flatten_size = x.view(1, -1).shape[1]

        self.fc1 = Linear(flatten_size, 3000)
        self.relu6 = ReLU()
        self.fc2 = Linear(3000, classes)
        self.logSoftmax = LogSoftmax(dim=1)

    def forward_conv(self, x):
        x = self.maxpool1(self.relu1(self.conv1(x)))
        x = self.maxpool2(self.relu2(self.conv2(x)))
        x = self.maxpool3(self.relu3(self.conv3(x)))
        x = self.maxpool4(self.relu4(self.conv4(x)))
        x = self.maxpool5(self.relu5(self.conv5(x)))
        return x

    def forward(self, x):
        x = self.forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.relu6(self.fc1(x))
        x = self.fc2(x)
        output = self.logSoftmax(x)
        return output
    
class CNN2D(Module):
    
	def __init__(self, numChannels, classes):
	 
		super().__init__()
        
		# initialize sets of CONV => RELU => POOL layers
		self.conv1 = Conv2d(in_channels=numChannels, out_channels=64,
			kernel_size=(1, 5), padding=0, dilation=1)
		self.relu1 = ReLU()
		self.maxpool1 = MaxPool2d(kernel_size=(1, 2),) #stride=(2, 2))
         
		self.conv2 = Conv2d(in_channels=64, out_channels=128,
			kernel_size=(1, 5))
		self.relu2 = ReLU()
		self.maxpool2 = MaxPool2d(kernel_size=(1, 2),)# stride=(2, 2))
		 
		self.conv3 = Conv2d(in_channels=128, out_channels=256,
			kernel_size=(1, 5))
		self.relu3 = ReLU()
		self.maxpool3 = MaxPool2d(kernel_size=(1, 2),)# stride=(2, 2))
		
        
        # initialize first (and only) set of FC => RELU layers
		self.fc1 = Linear(in_features=5376, out_features=1000)
		self.relu4 = ReLU()
		
        # initialize softmax classifier
		self.fc2 = Linear(in_features=1000, out_features=classes)
		self.logSoftmax = LogSoftmax(dim=1)


	def forward(self, x):
      
		# pass the input through the first set of CONV => RELU =>
		# POOL layers
		#print(x.shape)
		x = self.conv1(x)
		#print(x.shape)   
		x = self.relu1(x)
		x = self.maxpool1(x)
		#print(x.shape)
		# pass the output from the previous layer through the second
		# set of CONV => RELU => POOL layers
		x = self.conv2(x)
		x = self.relu2(x)
		x = self.maxpool2(x)
        
		x = self.conv3(x)
		x = self.relu3(x)
		x = self.maxpool3(x)
        
		# flatten the output from the previous layer and pass it
		# through the only set of FC => RELU layers
		x = flatten(x, 1)
		#print(x.shape)
		x = self.fc1(x)
		x = self.relu4(x)
        
		# pass the output to the softmax classifier to get the output
		# predictions
		x = self.fc2(x)
		output = self.logSoftmax(x)
		
		return output
   
    
def pre_process_ICNN_spdir (data_set,
                            config): 
    
    n_s = config['nb_samples'] 
    t_du = np.floor(config['ICNN_temporal_window_size'] / 2) 
    
    x_array = data_set['x_array']
    y_array = data_set['y_array']  
    
    absolute_speeds = np.zeros(n_s)
    gaze_points = np.concatenate(
        (
            x_array.reshape(1, n_s),
            y_array.reshape(1, n_s), 
        ),
        axis=0,
    )
    a_sp = np.zeros(n_s)
    a_sp[:-1] = (
        np.linalg.norm(gaze_points[:, 1:] - gaze_points[:, :-1], axis=0)
        * config["sampling_frequency"]
    )
    
     
    # Compute position difference vectors
    diff_ = np.zeros((2, n_s))
    
    diff_[0,:-1] = x_array[1:] - x_array[:-1]
    diff_[1,:-1] = y_array[1:] - y_array[:-1]
    
    diff_[:,-1] = diff_[:,-2]
    suc_dir = np.zeros_like(a_sp)
    
    # to avoid numerical instability
    diff_ += 1e-10
    
    _m = diff_[1,:]<0
    _p = diff_[1,:]>=0
    
    n_p = np.linalg.norm(diff_[:,_p], axis = 0)
    suc_dir[_p] = np.arccos(np.divide(diff_[0,:][_p], 
                                      n_p,
                                      where = n_p > 0))
    
    n_m = np.linalg.norm(diff_[:,_m], axis = 0)
    suc_dir[_m] = (2 * np.pi - np.arccos(np.divide(diff_[0,:][_m], 
                                                   n_m,
                                                   where = n_m > 0)))    
    
    time = np.arange(0, n_s) * (1/config['sampling_frequency'])
    
    data = np.concatenate((suc_dir.reshape(1, n_s),
                           a_sp.reshape(1, n_s),
                           time.reshape(1, n_s)), axis = 0)
    
    # Compute windowed data 
    
    w_data = np.array([data[:, np.minimum(np.maximum(0, 
                                                     np.arange(i-t_du, 
                                                               i+t_du+1, 
                                                               dtype=int)),
                                          n_s-1)] for i in range(n_s)])
    
    return w_data


def pre_process_ICNN (data_set,
                      config):
    """
     
    Parameters
    ----------
    data_set : TYPE
        DESCRIPTION.
    config : TYPE
        DESCRIPTION.

    Returns
    -------
    dict
        DESCRIPTION.

    """ 
    t_du = np.floor(config['ICNN_temporal_window_size'] / 2) 
    n_s = config['nb_samples']
    
    x_coord = data_set['x_array']
    y_coord = data_set['y_array']
    time = np.arange(0, n_s) * (1/config['sampling_frequency'])
    
    data = np.concatenate((x_coord.reshape(1, n_s),
                            y_coord.reshape(1, n_s),
                            #time.reshape(1, n_s))
                          ), axis = 0)
    # data = np.concatenate((x_coord.reshape(1, n_s),
    #                        y_coord.reshape(1, n_s)), axis = 0)
    # Compute windowed data 
    w_data = np.array([data[:, np.minimum(np.maximum(0, 
                                                     np.arange(i-t_du, 
                                                               i+t_du+1, 
                                                               dtype=int)),
                                          n_s-1)] for i in range(n_s)])
    
    return w_data


def train_cnn1d(dataset, labels, config):
    """
    Train and save a CNN1D model using a single dataset.

    Parameters
    ----------
    data_set : dict
        Contains 'x_array', 'y_array', and 'labels' (1D array of class indices).
    config : dict
        Configuration with 'nb_samples', 'ICNN_temporal_window_size', 'sampling_frequency',
        'task' ('binary' or 'ternary'), 'verbose', and 'model_path'.
    num_epochs : int
        Number of training epochs.
    batch_size : int
        Batch size for training and validation.
    learning_rate : float
        Learning rate for the optimizer.
    val_split : float
        Fraction of data to use for validation (0 to 1).

    Returns
    -------
    None
    """
    
    
    training_data = pre_process_ICNN(dataset, 
                                     config)
    print(training_data.shape)
    X_tensor = torch.tensor(training_data, dtype=torch.float32) 
    y_tensor = torch.tensor(labels, dtype=torch.long)
    
    # plt.plot(labels)
    # plt.show()
    # plt.clf()
    # Validate input
    if config['task'] == 'binary':
        num_classes = 3
    
    elif config['task'] == 'ternary':
        num_classes = 4
    
    input_channels = 2 
    input_length = config['ICNN_temporal_window_size']
     
    # DataLoader
    dataset = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(dataset, 
                              batch_size=config['ICNN_batch_size'], 
                              shuffle=True)
        
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CNN1D(numChannels=input_channels, classes=num_classes, input_length = input_length).to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), 
                           lr=config['ICNN_learning_rate'])

 
    
    for epoch in range(config['ICNN_num_epochs']):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        preds = []
        targets = []
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
    
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
            preds += list(pred)
            targets += list (target)
            # plt.plot(pred)
            # plt.plot(target)
            # plt.show()
            # plt.clf()
        
        plt.plot(np.array(preds)[:500])
        plt.show()
        plt.clf()
        
        plt.plot(np.array(targets)[:500])
        plt.show()
        plt.clf()
    
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1} - Precision: {accuracy:.2f}%")
      
    path = 'vision_toolkit/segmentation/segmentation_algorithms/trained_models/I_CNN/i_cnn_{task}.pt'.format(task = config['task'])
    torch.save(model.state_dict(), path)  
    
    if config['verbose']:
        print("Training complete")

 
    
    
    
    
def process_ICNN (data_set,
                  config): 
     
    if config['verbose']:
        
        print("Processing CNN Identification...")  
        start_time = time.time()
        
    task = config['task']
    path = 'vision_toolkit/segmentation/segmentation_algorithms/trained_models/I_CNN/i_cnn_{task}.pt'.format(task = config['task'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    
    if task == 'binary':
        classes = 3
    
    elif task == 'ternary':
        classes = 4
        
    model = CNN1D(numChannels=2, classes=classes, input_length=config['ICNN_temporal_window_size'])
    model.load_state_dict(torch.load(path))
    model.eval()
        
    w_data = pre_process_ICNN(data_set, 
                              config)
        
    tensor_test_X = torch.tensor(w_data, dtype=torch.float32)
    testDataLoader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tensor_test_X),
                                                 batch_size=config['ICNN_batch_size']) 
        
    preds = []
 
    for x in testDataLoader: 
        x = x[0].to(device) 
        pred = model(x)
        preds.extend(pred.argmax(axis=1).cpu().numpy())
        
    preds = np.array(preds)  
    plt.plot(preds)
    plt.show()
    plt.clf()
    return 
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    model = CNN1D(numChannels = 3, classes = classes)
    model.load_state_dict(torch.load(path))
    model = model.to(device)
    
    model.eval()
    
    w_data = pre_process_ICNN(data_set, 
                              config)
    w_data = w_data.astype(np.float32)

    tensor_test_X = torch.tensor(w_data)
    testDataLoader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tensor_test_X),
                                                 batch_size=1024)
 
    preds = []
 
    for x in testDataLoader:
       
        x = x[0].to(device)
        
        pred = model(x)
        preds.extend(pred.argmax(axis=1).cpu().numpy())
            
    preds = np.array(preds)+1
    
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
    
    
    
    
    