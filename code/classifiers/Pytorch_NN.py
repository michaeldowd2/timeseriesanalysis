import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from code.classifiers.abstract.Classifier import Classifier

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

torch.manual_seed(42)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BinaryClassification(nn.Module):
    def __init__(self, dropout, features):
        super(BinaryClassification, self).__init__()
        self.layer_1 = nn.Linear(features, 128) 
        self.layer_2 = nn.Linear(128, 64)
        self.layer_3 = nn.Linear(64, 32)
        self.layer_out = nn.Linear(32, 1) 
        
        self.relu = nn.ReLU()
        self.dropout_1 = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)
        self.dropout_3 = nn.Dropout(p=dropout)
        self.batchnorm_1 = nn.BatchNorm1d(128)
        self.batchnorm_2 = nn.BatchNorm1d(64)
        self.batchnorm_3 = nn.BatchNorm1d(32)
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm_1(x)
        x = self.dropout_1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm_2(x)
        x = self.dropout_2(x)
        x = self.relu(self.layer_3(x))
        x = self.batchnorm_3(x)
        x = self.dropout_3(x)
        x = self.layer_out(x)
        return x

class Pytorch_NN(Classifier):
    def __init__(self, iterations, pca_transform, pca_comps, samples, test_ratio, dropout, included = {}, excluded = {}):
        self.iterations = iterations
        self.pca_transform = pca_transform
        self.pca_comps = pca_comps
        self.samples = samples
        self.test_ratio = test_ratio
        self.dropout = dropout
        self.included = included
        self.excluded = excluded
    
    def GenerateModelResults(self, dataset, run_range):
        include_current_for_test = False
        
        samples = self.samples
        pca_transform = self.pca_transform
        pca_comps = self.pca_comps
        test_ratio = self.test_ratio
        iterations = self.iterations
        dropout = self.dropout
        
        res_dict = {'date':[], 'params':[], 'test_F1': [], 'prediction':[]}
        for i in range(run_range[0], run_range[-1]+2):
            date = dataset.index[i]
            
            if include_current_for_test:
                prev_date = date
            else:
                prev_date = dataset.index[i-1]

            sampled_df = self.SampleDataframe(dataset, prev_date, samples)
            scaler, pca, model, test_F1 = self.TrainForDate(sampled_df, iterations, test_ratio, pca_comps, pca_transform, dropout)
            pred = self.PredictForDate(dataset, date, scaler, pca, pca_transform, model)

            res_dict['date'].append(date)
            prediction = -1
            if pred[0] == 1:
                prediction = 1
            res_dict['params'].append(model)
            res_dict['test_F1'].append(test_F1)
            res_dict['prediction'].append(prediction)

        res = pd.DataFrame(res_dict).set_index('date')
        return res
    
    def TrainForDate(self, sampled_df, iterations, test_ratio, pca_comps, pca_transform, dropout):
        data = sampled_df.to_numpy()
        features, label = data[:,:-1], data[:,-1]
        scaler = QuantileTransformer(n_quantiles=100, random_state=42)
        #scaler = StandardScaler()
        scl_data = scaler.fit_transform(features)
        x, y = scl_data, label
        
        
        #train on last data, test on first in range
        size = len(y)
        test_size = int(test_ratio*len(y))
        x_train, y_train = x[test_size:,:], y[test_size:]
        x_test, y_test = x[:test_size,:], y[:test_size]
        #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_ratio, random_state=42)

        #fit pca on train, transform test
        pca = PCA(n_components=pca_comps, random_state=42)
        if pca_transform:
            x_train = pca.fit_transform(x_train)
            x_test = pca.transform(x_test)
            print('pca variance: ' + str(sum(pca.explained_variance_ratio_)))

        print('x train: ' + str(x_train.shape) + ', y_train: ' + str(y_train.shape) + ', x_test: ' + str(x_test.shape) + ', y_test: ' + str(y_test.shape))
        model = BinaryClassification(dropout, x_train.shape[1])
        model.to(DEVICE)
        
        x_batch = torch.FloatTensor(x_train).to(DEVICE)
        y_batch = torch.FloatTensor(y_train).to(DEVICE)
        x_ts_ten = torch.FloatTensor(x_test).to(DEVICE)
        
        #trained_model = train_model(iterations, model, x_tr_ten, y_tr_ten)
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr= 1e-6)
        model.train()
        for i in range(0,iterations):
            optimizer.zero_grad()

            y_pred = model(x_batch)

            loss = criterion(y_pred, y_batch.unsqueeze(1))
            acc = self.binary_acc(y_pred, y_batch.unsqueeze(1))

            loss.backward()
            optimizer.step()

            if i % 1000 == 0:
                print(f'Iteration {i+0:03}: | Loss: {loss:.5f} | Acc: {acc:.3f}')
        
        model.eval()
        y_ts_pred_ten = torch.round(torch.sigmoid(model(x_ts_ten)))
        
        y_test_pred = y_ts_pred_ten.cpu().detach().numpy().flatten()
        
        accuracy_test = accuracy_score(y_test, y_test_pred)
        f1_test = f1_score(y_test, y_test_pred, average='weighted')

        #print('best params: ' + str(best_model.best_params_))
        self.PlotPredVsAct(y_test_pred, y_test, 'acc: ' + str(accuracy_test) + ', f1: ' + str(f1_test))
        
        return scaler, pca, model, f1_test
        
    def PredictForDate(self, df, date, scaler, pca, pca_transform, model):
        sampled_df = self.SampleDataframe(df, date, 1)
        data = sampled_df.to_numpy()
        x, y = data[:,:-1], data[:,-1]
        x = scaler.transform(x)
        if pca_transform:
            x = pca.transform(x)
        x_ten = torch.FloatTensor(x).to(DEVICE)
        y_pred_ten = torch.round(torch.sigmoid(model(x_ten)))
        y_pred = y_pred_ten.cpu().detach().numpy().flatten()
        return y_pred
       
    def binary_acc(self, y_pred, y_test):
        y_pred_tag = torch.round(torch.sigmoid(y_pred))
        correct_results_sum = (y_pred_tag == y_test).sum().float()
        acc = correct_results_sum/y_test.shape[0]
        acc = torch.round(acc * 100)
        return acc
    

