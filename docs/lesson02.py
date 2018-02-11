'''
Created on 31 Jan 2018

@author: thomasgumbricht
'''

import pandas as pd
from sklearn import model_selection
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot

class RegressionModels:
    '''Machinelearning using regression models
    '''
    def __init__(self, columns, target):
        '''creates an instance of RegressionModels
        '''
        self.columns = columns
        self.target = target
        
    def ImportUrlDataset(self,url):    
        self.dataframe = pd.read_csv(url, delim_whitespace=True, names=self.columns)
        
    def ExtractDf(self,omitL):
        #extract the target column as y
        self.y = self.dataframe[self.target]
        #appeld the target to the list of features to be omitted
        omitL.append(self.target)
        #define the list of data to use
        dataL = [item for item in self.dataframe.columns if item not in omitL]
        #extract the data columns as X
        self.X = self.dataframe[dataL]

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

    def ModelSelectSet(self,modD):
        self.models = []
        if 'OLS' in modD:
            self.models.append(('OLS', linear_model.LinearRegression(**modD['OLS'])))
        if 'TheilSen' in modD:
            self.models.append(('TheilSen', linear_model.TheilSenRegressor(**modD['TheilSen'])))
        if 'Huber' in modD:
            self.models.append(('Huber', linear_model.HuberRegressor(**modD['Huber'])))
        if 'KnnRegr' in modD:
            self.models.append(('KnnRegr', KNeighborsRegressor( **modD['KnnRegr'])))
        if 'DecTreeRegr' in modD:
            self.models.append(('DecTreeRegr', DecisionTreeRegressor(**modD['DecTreeRegr'])))
        if 'SVR' in modD:
            self.models.append(('SVR', SVR(**modD['SVR'])))
        if 'RandForRegr' in modD:
            self.models.append(('RandForRegr', RandomForestRegressor( **modD['RandForRegr'])))
    
    def RegrModTrainTest(self, testsize=0.3, plot=True):
        #Split the data into training and test substes
        X_train, X_test, y_train, y_test = model_selection.train_test_split(self.X, self.y, test_size=testsize)
        #Loop over the defined models
        for m in self.models:
            #Retrieve the model name and the model itself
            name,mod = m
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
            #cross_val_predict returns an array of the same size as `y` where each entry
            #is a prediction obtained by cross validation:
            predict = model_selection.cross_val_predict(mod, self.X, self.y, cv=kfold)
            #to retriece regressions scores, use cross_val_score
            scoring = 'r2'
            r2 = model_selection.cross_val_score(mod, self.X, self.y, cv=kfold, scoring=scoring)
            #The correlation coefficient
            #Print out the model name
            print 'Model: %s' %(name)
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
              
if __name__ == ('__main__'):
    columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    target = 'MEDV'
    regmods = RegressionModels(columns, target)
    regmods.ImportUrlDataset('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data')
    regmods.ExtractDf([])
    modD = {}
    modD['OLS'] = {}
    modD['TheilSen'] = {}
    modD['Huber'] = {}
    modD['KnnRegr'] = {'n_neighbors':8}
    modD['DecTreeRegr'] = {}
    #modD['RandForRegr'] = {}
    modD['RandForRegr'] = {'n_estimators':30}
    #modD['SVR'] = {}
    modD['SVR'] = {'kernel':'linear','C':1.5,'epsilon':0.05}
    #modD['SVR'] = {}
    modD['RandForRegr'] = {'n_estimators':30}
    #modD['KnnRegr'] = {}
    #modD['SVR'] = {'kernel':'poly'}
    regmods.ModelSelectSet(modD)
    regmods.RegrModTrainTest()
    regmods.RegrModKFold()