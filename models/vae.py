# import necessary packages
import numpy as np
import torch
from torch import nn
from torch import optim

class VariationalAutoencoder(nn.Module):
    def __init__(self, X, y):
        super(VariationalAutoencoder, self).__init_()  # inherit from torch nn module
        self.X = X
        self.y = y
        numFeatures = X.shape[1]

        # define hyperparameters
        self.dense_layers

        self.encoder = nn.Sequential(
            # input layer
            nn.Linear(numFeatures, self.dense_layers),
            nn.LeakyReLU(),
            nn.Dropout(),

            # hidden layer
            nn.Linear(self.dense_layers, self.dense_Layers % 2),
            nn.LeakyReLU(),
            nn.Dropout()
            
        )

        self.decoder = nn.Sequential()
    
    # train model
    def _train():
        pass

    # test model
    def _test():
        pass

    # evaluate model performance
    def evaluate():
        pass