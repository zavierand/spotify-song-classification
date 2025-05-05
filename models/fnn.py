# we'll log the outputs
import logging

# frameworks
import numpy as np
import torch
from torch import nn 
from torch import optim

# log config for training
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# class module for forward feeding neural network
class FNN(nn.Module):
    def __init__(self, X, y, input, output):
        super(FNN, self).__init__()
        self.X = X
        self.y = y

        # parameters and hyperparameters
        self.numFeatures = input # param
        self.dense_nodes = 5  # hard code a value now, dynamically change later
        self.output_nodes = output 

        # loss function + optimzing function
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters())

        # define our model
        self.fnn = nn.Sequential(
            # input layer
            nn.Linear(self.numFeatures, self.dense_nodes),
            nn.ReLU(),
            nn.Dropout(),   # include dropout to randomly zero some features

            # first hidden layer
            nn.Linear(self.dense_nodes, self.dense_nodes),
            nn.ReLU(),
            nn.Dropout(),

            # second hidden layer
            nn.Linear(self.dense_nodes, self.dense_nodes),
            nn.ReLU(),
            nn.Dropout(),

            # third hidden layer
            nn.Linear(self.dense_nodes, self.dense_nodes),
            nn.ReLU(),
            nn.Dropout(),

            # output layer
            nn.Linear(self.dense_nodes, self.output_nodes),
            nn.LogSoftmax(dim = 1)
        )

    # forward
    def forward(self, x):
        return self.fnn(x)

    # train model
    def _train(self, X, y, num_epochs = 200):
        # forward pass
        self.forward(X)

        # we'll also calculate the probabilities for each class
        # also used for auc plots
        pred_probs = []
        # training loop
        for epoch in range(0, num_epochs):
            # forward prop
            y_pred = self.fnn(X)
            loss = self.criterion(y_pred, y)

            # now, we'll log each epoch information
            logging.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            print()

            # backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            probs = torch.exp(y_pred).detach().cpu().numpy()
            pred_probs.append(probs)

        # we'll return pred_probs to calc auc, y_pred vector
        return y_pred, pred_probs

    # evaluate model performance
    def _evaluate(self, X, y):
        self.fnn.eval() # set model to eval

        # disable optimization
        with torch.no_grad():
            y_pred = self.forward(X)
            y = y.view(-1)

            # compute loss
            loss = self.criterion(y_pred, y)

            # log loss
            logging.info(f'Loss: {loss}')

            # return loss for any reason we might need it
            return loss