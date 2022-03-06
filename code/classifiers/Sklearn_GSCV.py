import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from code.classifiers.abstract.Classifier import Classifier

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import svm
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

class Sklearn_GSCV(Classifier):
    def __init__(self, pca_comps, samples, test_ratio, classifier, param_grid, included = {}, excluded = {}):
        self.pca_comps = pca_comps
        self.samples = samples
        self.test_ratio = test_ratio
        self.classifier = classifier
        self.param_grid = param_grid
        self.included = included
        self.excluded = excluded
    
    def GenerateModelResults(self, dataset, run_range):
        res_dict = {'date':[], 'params':[], 'test_F1': [], 'prediction':[]}
        for i in range(run_range[0], run_range[-1]+2):
            date = dataset.index[i]
            prev_date = dataset.index[i-1]

            sampled_df = self.SampleDataframe(dataset, prev_date, self.samples)
            scaler, pca, model, test_F1 = self.TrainForDate(sampled_df, self.test_ratio, self.classifier, self.param_grid, self.pca_comps)
            pred = self.PredictForDate(dataset, date, scaler, pca, model)

            res_dict['date'].append(date)
            prediction = -1
            if pred[0] == 1:
                prediction = 1
            res_dict['params'].append(model.best_params_)
            res_dict['test_F1'].append(test_F1)
            res_dict['prediction'].append(prediction)

        res = pd.DataFrame(res_dict).set_index('date')
        return res
    
    def TrainForDate(self, sampled_df, test_size, model, param_grid, pca_comps):
        data = sampled_df.to_numpy()
        features, label = data[:,:-1], data[:,-1]
        scaler = StandardScaler()
        scl_data = scaler.fit_transform(features)
        x, y = scl_data, label

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)

        #fit pca on train, transform test
        pca = PCA(n_components=pca_comps, random_state=42)
        x_train = pca.fit_transform(x_train)
        x_test = pca.transform(x_test)
        
        print('pca variance: ' + str(sum(pca.explained_variance_ratio_)))

        clf = GridSearchCV(model, param_grid, refit=True, scoring='neg_root_mean_squared_error')
        best_model = clf.fit(x_train, y_train) # model.fit(x_train,y_train)
        y_test_pred = best_model.predict(x_test)

        accuracy_test = accuracy_score(y_test, y_test_pred)
        f1_test = f1_score(y_test, y_test_pred, average='weighted')

        print('best params: ' + str(best_model.best_params_))
        self.PlotPredVsAct(y_test_pred, y_test, 'acc: ' + str(accuracy_test) + ', f1: ' + str(f1_test))
        
        return scaler, pca, best_model, f1_test
    
    def PredictForDate(self, df, date, scaler, pca, model):
        sampled_df = self.SampleDataframe(df, date, 1)
        data = sampled_df.to_numpy()
        x, y = data[:,:-1], data[:,-1]
        x = scaler.transform(x)
        x = pca.transform(x)
        y_pred = model.predict(x)
        return y_pred
    

