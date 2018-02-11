'''
Created on 5 Feb 2018

@author: thomasgumbricht
'''

import tempfile

import numpy as np
import pandas as pd
from scipy.stats import randint as sp_randint
from sklearn.cluster import FeatureAgglomeration
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import Pipeline
from sklearn.externals.joblib import Memory

from sklearn.model_selection import KFold
from sklearn import model_selection
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot

class RegressionModels:
    '''Machinelearning using regression models
    '''
    def __init__(self, columns,target):
        '''creates an instance of RegressionModels
        '''
        self.columns = columns
        self.target = target
        #create an empty dictionary for features to be discarded by each model
        self.modelDiscardD = {}
        self.modD = {}

    def ImportUrlDataset(self,url):    
        self.dataframe = pd.read_csv(url, delim_whitespace=True, names=self.columns)

    def ExtractDf(self,omitL):
        #extract the target column as y
        self.y = self.dataframe[self.target]
        #appeld the target to the list of features to be omitted
        omitL.append(self.target)
        #define the list of data to use
        self.columnsX = [item for item in self.dataframe.columns if item not in omitL]
        #extract the data columns as X
        self.X = self.dataframe[self.columnsX]
    
    def PlotRegr(self, obs, pred, title, color='black'):
        pyplot.xticks(())
        pyplot.yticks(())
        fig, ax = pyplot.subplots()
        ax.scatter(obs, pred, edgecolors=(0, 0, 0),  color=color)
        ax.plot([obs.min(), obs.max()], [obs.min(), obs.max()], 'k--', lw=3)
        ax.set_xlabel('Observations')
        ax.set_ylabel('Predictions')
        pyplot.title(title)
        pyplot.show()
        
    def ModelSelectSet(self):
        self.models = []
        if 'OLS' in self.modD:
            self.models.append(('OLS', linear_model.LinearRegression(**self.modD['OLS'])))
            self.modelDiscardD['OLS'] = []
        if 'TheilSen' in self.modD:
            self.models.append(('TheilSen', linear_model.TheilSenRegressor(**self.modD['TheilSen'])))
            self.modelDiscardD['TheilSen'] = []
        if 'Huber' in self.modD:
            self.models.append(('Huber', linear_model.HuberRegressor(**self.modD['Huber'])))
            self.modelDiscardD['Huber'] = []
        if 'KnnRegr' in self.modD:
            self.models.append(('KnnRegr', KNeighborsRegressor( **self.modD['KnnRegr'])))
            self.modelDiscardD['KnnRegr'] = []
        if 'DecTreeRegr' in self.modD:
            self.models.append(('DecTreeRegr', DecisionTreeRegressor(**self.modD['DecTreeRegr'])))
            self.modelDiscardD['DecTreeRegr'] = []
        if 'SVR' in self.modD:
            self.models.append(('SVR', SVR(**self.modD['SVR'])))
            self.modelDiscardD['SVR'] = []
        if 'RandForRegr' in self.modD:
            self.models.append(('RandForRegr', RandomForestRegressor( **self.modD['RandForRegr'])))
            self.modelDiscardD['RandForRegr'] = []
    
    def RegrModTrainTest(self, testsize=0.3, plot=True):
        #Split the data into training and test substes
        X_train, X_test, y_train, y_test = model_selection.train_test_split(self.X, self.y, test_size=testsize)
        #Loop over the defined models
        for m in self.models:
            #Retrieve the model name and the model itself
            name,mod = m
            #Remove the features listed in the modelDiscarD
            self.ExtractDf(self.modelDiscardD[name])
            #Fit the model
            mod.fit(X_train, y_train)
            #Predict the independent variable in the test subset
            predict = mod.predict(X_test)
            #Print out the model name
            print 'Model: %s' %(name)
            #Print out RMSE
            print("    Mean squared error: %.2f" \
                % mean_squared_error(y_test, predict))
            #Print explained variance score: 1 is perfect prediction
            print('    Variance score: %.2f' \
                % r2_score(y_test, predict))
            if plot:
                title = ('Model: %(mod)s; RMSE: %(rmse)2f; r2: %(r2)2f' \
                          % {'mod':name,'rmse':mean_squared_error(y_test, predict),'r2': r2_score(y_test, predict)} )
                self.PlotRegr(y_test, predict, title, color='green')

    def RegrModKFold(self,folds=10, plot=True):
        #set the kfold
        kfold = model_selection.KFold(n_splits=folds)
        for m in self.models:
            #Retrieve the model name and the model itself
            name,mod = m
            #Remove the features listed in the modelDiscarD
            self.ExtractDf(self.modelDiscardD[name])
            #cross_val_predict returns an array of the same size as `y` where each entry
            #is a prediction obtained by cross validation:
            predict = model_selection.cross_val_predict(mod, self.X, self.y, cv=kfold)
            #to retriece regressions scores, use cross_val_score
            scoring = 'r2'
            r2 = model_selection.cross_val_score(mod, self.X, self.y, cv=kfold, scoring=scoring)
            #The correlation coefficient
            #Print out the model name
            print 'Model: %s' %(name)
            print mod
            #Print out correlation coefficients
            print('    Regression coefficients: \n', r2)    
            #Print out RMSE
            print("Mean squared error: %.2f" \
                  % mean_squared_error(self.y, predict))
            #Explained variance score: 1 is perfect prediction
            print('Variance score: %.2f' \
                % r2_score(self.y, predict))
            if plot:
                title = ('Model: %(mod)s; RMSE: %(rmse)2f; r2: %(r2)2f' \
                          % {'mod':name,'rmse':mean_squared_error(self.y, predict),'r2': r2_score(self.y, predict)} )
                self.PlotRegr(self.y, predict, title, color='blue')
        
    def ReportSearch(self, results, n_top=3):
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results['rank_test_score'] == i)
            for candidate in candidates:
                print("Model with rank: {0}".format(i))
                print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                      results['mean_test_score'][candidate],
                      results['std_test_score'][candidate]))
                print("Parameters: {0}".format(results['params'][candidate]))
                print("")

    def WardClustering(self, nClusters):
        ward = FeatureAgglomeration(n_clusters=nClusters)
        #fit the clusters
        ward.fit(self.X, self.y) 
        #print out the clustering
        print 'labels_', ward.labels_
        #Reset self.X
        self.X = ward.transform(self.X)
        #print the shape of reduced X data
        print 'Agglomerated X data shape:',self.X.shape

    def TuneWardClustering(self, nClustersL, kfolds=2):
        cv = KFold(kfolds)  # cross-validation generator for model selection
        ridge = BayesianRidge()
        cachedir = tempfile.mkdtemp()
        mem = Memory(cachedir=cachedir)
        ward = FeatureAgglomeration(n_clusters=6, memory=mem)
        clf = Pipeline([('ward', ward), ('ridge', ridge)])
        # Select the optimal number of parcels with grid search
        clf = GridSearchCV(clf, {'ward__n_clusters': nClustersL}, n_jobs=1, cv=cv)
        clf.fit(self.X, self.y)  # set the best parameters
        #report the top three results
        self.ReportSearch(clf.cv_results_,3)
        #rerun with the best cluster agglomeration result
        tunedClusters = clf.best_params_['ward__n_clusters']
        return (tunedClusters)
   
    def PCAdecompose(self, minExplainRatio=0, nComps=3 ):
        if minExplainRatio > 0:
            pca = PCA()
            pca.fit(self.X)
            print 'PCA explained ratios', pca.explained_variance_ratio_
            nComps = len([item for item in pca.explained_variance_ratio_ if item >= minExplainRatio])
            print 'accepted components: %(n)d' %{'n':nComps}
        pca = PCA(n_components=nComps)
        pca.fit(self.X)
        self.X = pca.transform(self.X)
        print 'PCA explained ratios', pca.explained_variance_ratio_
        print 'PCA X data shape:',self.X.shape
        return (self.X.shape[1])
              
    def RandomTuningParams(self,nFeatures):
        self.paramDist = {}
        # specify parameters and distributions to sample from
        for m in self.models:
            name,mod = m
            print ('name'), (name), (mod.get_params())
            if name == 'KnnRegr':
                self.paramDist[name] = {"n_neighbors": sp_randint(4, 12),
                              'leaf_size': sp_randint(10, 50),
                              'weights': ('uniform','distance'),
                              'p': (1,2),
                              'algorithm': ('auto','ball_tree', 'kd_tree', 'brute')}
            elif name =='DecTreeRegr':
                self.paramDist[name] = {"max_depth": [3, None],
                              "min_samples_split": sp_randint(2, 6),
                              "min_samples_leaf": sp_randint(1, 4)}
            elif name =='SVR':
                self.paramDist[name] = {"kernel": ['linear', 'rbf'],
                              "epsilon": (0.05, 0.1, 0.2),
                              "C": (1, 2, 5, 10)}
            elif name =='RandForRegr':
                self.paramDist[name] = {"max_depth": [3, None],
                              "n_estimators": sp_randint(10, 50),
                              "max_features": sp_randint(1, nFeatures),
                              "min_samples_split": sp_randint(2, 6),
                              "min_samples_leaf": sp_randint(1, 5),
                              "bootstrap": [True,False]}
        
    def RandomTuning(self, fraction=0.5, nIterSearch=6, n_top=3):
        #Randomized search
        for m in self.models:
            #Retrieve the model name and the model itself
            name,mod = m
            print name, self.paramDist[name]
            search = RandomizedSearchCV(mod, param_distributions=self.paramDist[name],
                                               n_iter=nIterSearch)
            X_train, X_test, y_train, y_test = model_selection.train_test_split(self.X, self.y, test_size=(1-fraction))
            search.fit(X_train, y_train)
            self.ReportSearch(search.cv_results_,n_top)
            #Retrieve the top ranked tuning
            best = np.flatnonzero(search.cv_results_['rank_test_score'] == 1)
            tunedModD=search.cv_results_['params'][best[0]]
            #Append any initial modD hyper-parameter definition 
            for key in self.modD[name]:
                tunedModD[key] = self.modD[name][key] 
            regmods.modD[name] = tunedModD
    
    def ReportModParams(self):
        print 'Model hyper-parameters:'
        for m in self.models:
            #Retrieve the model name and the model itself
            name,mod = m
            print ('    name'), (name), (mod.get_params())
        
if __name__ == ('__main__'):
    columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    target = 'MEDV'
    regmods = RegressionModels(columns, target)
    regmods.ImportUrlDataset('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data')
    regmods.ExtractDf([])


    #Agglomerate the X data
    nClustersL = [4,5,6,7,8,9,10,11]
    nFeatures = regmods.TuneWardClustering(nClustersL)
    regmods.WardClustering(nFeatures)

    #Dimension reduction using PCA 
    nFeatures = regmods.PCAdecompose() #default: produces 3 eigen-vectors
    #nFeatures = regmods.PCAdecompose(0.1) #produces all eigen-vectors that explain at least 10% of the total variation
    #nFeatures = regmods.PCAdecompose(0, 4) #produces 4 eigen-vectors
    
    #Setup regressors to test
    regmods.modD['KnnRegr'] = {}
    regmods.modD['DecTreeRegr'] = {} 
    regmods.modD['SVR'] = {}
    regmods.modD['RandForRegr'] = {}
    
    #Invoke the models
    regmods.ModelSelectSet()
    
    #set the random tuning parameters
    regmods.RandomTuningParams(nFeatures)
    
    #Run the tuning
    regmods.RandomTuning()
    
    #Reset the models with the tuned hyper-parameters
    regmods.ModelSelectSet()
    
    #report model settings
    regmods.ReportModParams()
    
    #Run the models
    regmods.RegrModTrainTest()
    regmods.RegrModKFold() 
    