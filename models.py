import torch
import torch.nn as nn
import torch.nn.functional as F  # All functions that don't have any parameters


class NN(nn.Module):
    def __init__(self, image_shape):
        super(NN, self).__init__()
        self.image_shape = image_shape
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.image_shape, 2048),
            nn.ReLU(),
            nn.Linear(2048, self.image_shape),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits