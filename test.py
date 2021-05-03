# Imports
import os
from typing import Union

import torch.nn.functional as F  # All functions that don't have any parameters
import pandas as pd
import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torchvision
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
from pandas import io

# from skimage import io
from torch.utils.data import (
    Dataset,
    DataLoader,
)  # Gives easier dataset managment and creates mini batches
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions

from dataset import ArtChallengeDataset
from models import NN


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
image_shape = 3*256*64
learning_rate = 1e-3
batch_size = 5
num_epochs = 1

# Load Data
dataset = ArtChallengeDataset(
    csv_file="data/columns.csv", root_dir="data/columns", transform=transforms.ToTensor()
)

train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False) 



# Model
model = NN(image_shape=image_shape).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    losses = []

    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        flat_targets = torch.flatten(targets, start_dim=1, end_dim=-1)

        # forward
        scores = model(data)
        loss = criterion(scores, flat_targets)

        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

    print(f"Cost at epoch {epoch} is {sum(losses) / len(losses)}")


# Check accuracy on training to see how good our model is
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}"
        )

    model.train()


print("Checking accuracy on Training Set")
check_accuracy(train_loader, model)