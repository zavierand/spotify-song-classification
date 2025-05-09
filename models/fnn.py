# we'll log the outputs
import logging

# frameworks
import numpy as np
import torch
from torch import nn 
from torch import optim
from torch.utils.data import TensorDataset, DataLoader


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
        self.dense_nodes = 64  # hard code a value now, dynamically change later
        self.output_nodes = output 

        # define our model
        self.fnn = nn.Sequential(
            # input layer
            nn.Linear(self.numFeatures, self.dense_nodes),
            nn.ReLU(),
            nn.Dropout(p = 0.2),   # include dropout to randomly zero some features

            # first hidden layer
            nn.Linear(self.dense_nodes, self.dense_nodes * 2),
            nn.ReLU(),
            nn.Dropout(p = 0.2),

            # second hidden layer
            nn.Linear(self.dense_nodes * 2, self.dense_nodes * 2),
            nn.ReLU(),
            nn.Dropout(p = 0.2),

            # third hidden layer
            nn.Linear(self.dense_nodes * 2, self.dense_nodes),
            nn.ReLU(),
            nn.Dropout(p = 0.2),

            # output layer
            nn.Linear(self.dense_nodes, self.output_nodes),
        )

    # loss function + optimzing function
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    # forward
    def forward(self, x):
        return self.fnn(x)

    # train model
    def _train(self, X, y, num_epochs=200, batch_size=64):
        self.fnn.train()

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(num_epochs):
            total_loss = 0.0

            for batch_X, batch_y in loader:
                y_pred = self.forward(batch_X)
                loss = self.criterion(y_pred, batch_y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.4f}")

    # evaluate model performance
    def _evaluate(self, X, y):
        self.fnn.eval()

        with torch.no_grad():
            logits = self.forward(X)               # raw outputs
            loss = self.criterion(logits, y.view(-1))
            pred_probs = torch.softmax(logits, dim=1)  # get class probabilities
            y_pred = torch.argmax(pred_probs, dim=1)   # get predicted classes

            correct = (y_pred == y).sum().item()
            accuracy = correct / y.size(0)

            print(f'Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')

            return loss.item(), accuracy, y.cpu().numpy(), y_pred.cpu().numpy(), pred_probs.cpu().numpy()
