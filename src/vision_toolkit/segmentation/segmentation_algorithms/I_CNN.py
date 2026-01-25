#
# -*- coding: utf-8 -*-
import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


 
class CNN1D(nn.Module):
    def __init__(self, numChannels, classes, input_length):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=numChannels, out_channels=16, kernel_size=10)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=2)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool1d(kernel_size=2)

        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=2)
        self.relu4 = nn.ReLU()
        self.maxpool4 = nn.MaxPool1d(kernel_size=2)

        self.conv5 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=2)
        self.relu5 = nn.ReLU()
        self.maxpool5 = nn.MaxPool1d(kernel_size=2)

        # infer flatten size
        with torch.no_grad():
            dummy = torch.zeros(1, numChannels, input_length)
            z = self.forward_conv(dummy)
            flatten_size = z.view(1, -1).shape[1]

        self.fc1 = nn.Linear(flatten_size, 3000)
        self.relu6 = nn.ReLU()
        self.fc2 = nn.Linear(3000, classes)

        self.logSoftmax = nn.LogSoftmax(dim=1)

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
        return self.logSoftmax(x)


 
def pre_process_ICNN(data_set, config, *, force_odd_window=True):
    """
    

    Parameters
    ----------
    data_set : TYPE
        DESCRIPTION.
    config : TYPE
        DESCRIPTION.
    * : TYPE
        DESCRIPTION.
    force_odd_window : TYPE, optional
        DESCRIPTION. The default is True.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    w_data : TYPE
        DESCRIPTION.

    """
    x = np.asarray(data_set["x_array"], dtype=np.float32)
    y = np.asarray(data_set["y_array"], dtype=np.float32)
    n_s = int(x.shape[0])

    if y.shape[0] != n_s:
        raise ValueError("pre_process_ICNN: x_array/y_array length mismatch")

    # keep config consistent
    config["nb_samples"] = n_s

    W = int(config.get("ICNN_temporal_window_size", 251))
    if W < 3:
        raise ValueError("ICNN_temporal_window_size must be >= 3.")

    if (W % 2) == 0:
        if force_odd_window:
            W += 1
            config["ICNN_temporal_window_size"] = W
        else:
            raise ValueError("ICNN_temporal_window_size must be odd for centered windows.")

    half = W // 2
    data = np.stack([x, y], axis=0)  # (2, n_s)

    w_data = np.empty((n_s, 2, W), dtype=np.float32)
    for i in range(n_s):
        idx = np.arange(i - half, i + half + 1, dtype=int)
        idx = np.clip(idx, 0, n_s - 1)
        w_data[i] = data[:, idx]

    return w_data


def _classes_from_task(task):
    if task == "binary":
        return 2
    elif task == "ternary":
        return 3
    else:
        raise ValueError("task must be 'binary' or 'ternary'.")
    

 
def train_cnn1d_windows(Xw, y0, config):
    """
    Train and save CNN1D from windowed data.

    """
    verbose = bool(config.get("verbose", True))
    if verbose:
        print("Training I_CNN (from windows)...")
        start = time.time()

    task = config["task"]
    classes = _classes_from_task(task)

    Xw = np.asarray(Xw, dtype=np.float32)
    y0 = np.asarray(y0, dtype=np.int64).reshape(-1)

    if Xw.ndim != 3 or Xw.shape[1] != 2:
        raise ValueError(f"Xw must be (N,2,W). Got {Xw.shape}")
    if Xw.shape[0] != y0.shape[0]:
        raise ValueError(f"Xw/y0 length mismatch: {Xw.shape[0]} vs {y0.shape[0]}")

    W = int(config.get("ICNN_temporal_window_size", Xw.shape[2]))
    if Xw.shape[2] != W:
        # keep consistent with config/model
        config["ICNN_temporal_window_size"] = int(Xw.shape[2])
        W = int(Xw.shape[2])

    # label range check
    if y0.size == 0:
        raise ValueError("Empty labels array.")
    if y0.min() < 0 or y0.max() >= classes:
        raise ValueError(f"Labels out of range for classes={classes}: min={y0.min()} max={y0.max()}")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = bool(config.get("ICNN_cudnn_benchmark", True))

    X_tensor = torch.tensor(Xw, dtype=torch.float32)
    y_tensor = torch.tensor(y0, dtype=torch.long)

    ds = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(ds, batch_size=int(config.get("ICNN_batch_size", 1024)), shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN1D(numChannels=2, classes=classes, input_length=W).to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(config.get("ICNN_learning_rate", 1e-3)))

    epochs = int(config.get("ICNN_num_epochs", 25))
    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0
        total_loss = 0.0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * yb.size(0)
            pred = out.argmax(dim=1)
            correct += int((pred == yb).sum().item())
            total += int(yb.size(0))

        if verbose:
            acc = 100.0 * correct / max(1, total)
            print(f"Epoch {epoch+1}/{epochs} | loss={total_loss/max(1,total):.4f} | acc={acc:.2f}%")

    path = f"vision_toolkit/segmentation/segmentation_algorithms/trained_models/I_CNN/i_cnn_{task}.pt"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

    if verbose:
        print(f"Training complete. Saved: {path}")
        print(f"--- {time.time() - start:.2f}s ---")

    return path


 
 