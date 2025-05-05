'''
//
//
// By Zavier

This file is just some functions that will be used for preprocessing
our data. From experience, this is to help clean up the notebook, where
we will try to focus more on results and computing, trying to leave definitions
to .py scripts to be imported into the notebook.

More can be seen in the models file detailing the different models used in for
this problem.
'''
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

class SongData:
    '''
    Class definition for SongData. 

    SongData class helps modularize our dataset to be used across different models while also 
    abstracting away and providing different methods of preprocessing the data. This is useful
    for when we evaluating EDA of different models.

    '''
    def __init__(self, X, y):
        self.labels, self.num_labels, self.label_count = self.labelExtraction(y)
        self.feature_info = self.featureExtraction(X)
        self.min_max_scaling(X)
        self.z_score(X)
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data(X, y)

    # label extraction
    def labelExtraction(self, y):
        '''
        return some info on the labels
        '''
        y_list = list(y)
        # get each unique label
        labels = list(set(y))

        # get number of labels
        num_labels = len(labels)

        # get count of each label
        ## we assume we're working with a dataframe
        label_count = {}
        for label in labels:
            label_count[label] = y_list.count(label)

        # return labels and the number of labels
        return labels, num_labels, label_count

    # feature extraction function
    def featureExtraction(self, X):
        '''
        return general information about the features:
        - Number of features
        - Feature names
        - Data types
        - Null value counts
        - Basic statistics
        '''
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(data = X)
        
        feature_info = {
            'num_features': X.shape[1],
            'feature_names': X.columns.tolist(),
            'data_types': X.dtypes.to_dict(),
            'null_counts': X.isnull().sum().to_dict(),
            'summary_stats': X.describe(include='all').to_dict()
        }
        return feature_info

    # scaling function
    def min_max_scaling(self, X):
        '''
        scale a feature between 0 and 1
        '''
        return (X - np.min(X)) / (np.max(X) - np.min(X))

    # normalization
    def z_score(self, X):
        '''
        normalize input feature
        '''
        return (X - np.mean(X)) / (np.std(X))

    # cosine similarity
    def cosine_similarity(self, u, v):
        '''
        compute the cosine similarity between our feature vectors.
        this could come in handy when dealing with possible collinearity.
        '''
        # check if vectors are np arrays before computing
        if not isinstance(u, np.ndarray):
            u = np.array(u)
        
        if not isinstance(v, np.ndarray):
            v = np.array(v)
        
        dot = np.dot(u, v)
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)

        epsilon = 1e-10 # think of this as laplacian correction to avoid 0 division
        return dot / (norm_u * norm_v + epsilon)

    # test data
    def split_data(self, X, y):
        '''
        As per the project sheet:
        '*Make sure to do the following train/test split: For *each* genre, use 500 randomly picked songs for the 
        test set and the other 4500 songs from that genre for the training set. So the complete test set will be 
        5000x1 randomly picked genres (one per song, 500 from each genre). Use all the other data in the 
        training set and make sure there is no leakage.'
        '''
        X_train = []
        X_test = []
        y_train = []
        y_test = []

        # convert to numpy arrays if not already
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        
        # get indices for each genre/label
        label_indices = {}
        for label in self.labels:
            label_indices[label] = np.where(y == label)[0]
        
        # for each genre, select 500 random songs for test set
        for label in self.labels:
            indices = label_indices[label]
            # shuffle indices to ensure randomness
            np.random.shuffle(indices)
            
            # select 500 indices for test set
            test_indices = indices[:500]
            # use remaining indices for training set
            train_indices = indices[500:]
            
            # add to our updated lists
            for idx in test_indices:
                X_test.append(X[idx])
                y_test.append(y[idx])
            
            for idx in train_indices:
                X_train.append(X[idx])
                y_train.append(y[idx])
        
        # convert to numpy arrays
        X_train_np = np.array(X_train)
        X_test_np = np.array(X_test)
        y_train_np = np.array(y_train)
        y_test_np = np.array(y_test)
        
        return X_train_np, X_test_np, y_train_np, y_test_np
    
    # convert data to tensors for deeper models
    @staticmethod
    def transformToTensor(X, y):
        '''
        Function returns the features and labels as tensors for the deep learning models
        '''
        X_tensor = torch.as_tensor(X, dtype = torch.float32)
        y_tensor = torch.as_tensor(y, dtype = torch.long)
        return X_tensor, y_tensor

    # convert to np arrays
    def transformToNP(a):
        '''
        Function returns the input matrix/vector as type np.ndarray
        '''
        return np.array(a)