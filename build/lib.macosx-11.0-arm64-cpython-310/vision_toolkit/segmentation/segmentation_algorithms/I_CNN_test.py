# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Paramètres
num_samples = 1000
input_length = 50
num_features = 3  # multivarié
num_classes = 4

# Données simulées : bruitées + classes aléatoires
X = np.random.randn(num_samples, num_features, input_length)
y = np.random.randint(0, num_classes, size=(num_samples,))

# Conversion en tenseurs
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=False)





class CNN1DClassifier(nn.Module):
    def __init__(self, input_channels, input_length, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2)

        # Estimer la taille de sortie après les conv/pool
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_length)
            out = self.pool2(self.relu2(self.conv2(self.pool1(self.relu1(self.conv1(dummy))))))
            flatten_size = out.view(1, -1).shape[1]

        self.fc1 = nn.Linear(flatten_size, 100)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(100, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.log_softmax(self.fc2(x))
        return x



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN1DClassifier(input_channels=num_features, input_length=input_length, num_classes=num_classes).to(device)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Boucle d'entraînement
for epoch in range(10):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    preds = []
    targets = []
    
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        output = model(xb)
        loss = criterion(output, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
        
        preds += list(pred)
        targets += list (yb)
        
    plt.plot(np.array(preds)[:1000])
    plt.show()
    plt.clf()
    
    plt.plot(np.array(targets)[:1000])
    plt.show()
    plt.clf()

    acc = 100 * correct / total
    print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | Accuracy: {acc:.2f}%")











