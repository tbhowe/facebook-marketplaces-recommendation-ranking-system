#%%

from Dataset import ImagesDataset
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
import torch

import pandas as pd
import numpy as np
import os

#%% initialise Dataset and data loader

FacebookImagesDataset=ImagesDataset()
batch_size=32
train_loader=DataLoader(FacebookImagesDataset, batch_size=batch_size, shuffle=True)

# example=next(iter(train_loader))

# features,labels = example

# print('oll yn kompoester yw. Splann!')


class CNN(torch.nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        # initialise weights and biases (parameters)
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, 3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, 3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, 3),
            # torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(230400, 2000),
            torch.nn.ReLU(),
            torch.nn.Linear(2000, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 13),
            # torch.nn.Softmax()
        )

    def forward(self, features):
        """Takes in features and makes a prediction"""
        return self.layers(features)

    
# %%
