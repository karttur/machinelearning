'''
Created on 1 Feb 2018

@author: thomasgumbricht
'''

import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import RFE

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
        self.y = self.dataframe[target]
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
        #Loop over the defined models
        for m in self.models:
            #Retrieve the model name and the model itself
            name,mod = m
            #Remove the features listed in the modelDiscarD
            self.ExtractDf(self.modelDiscardD[name])
            #Split the data into training and test subsets
            X_train, X_test, y_train, y_test = model_selection.train_test_split(self.X, self.y, test_size=testsize)
            #Fit the model
            mod.fit(X_train, y_train)
            #Predict the independent variable in the test subset
            predict = mod.predict(X_test)
            #Print out the model name
            print '\nModel: %s (train/test)' %(name)
            print '    discarded features',self.modelDiscardD[name]
            print '    nr of independent features', X_train.shape[1]
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
            print '\nModel: %s (cross validation folds)' %(name)
            print '    discarded features',self.modelDiscardD[name]
            print '    nr of independent features', self.X.shape[1]
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
 
    def VarianceSelector(self,thresh=0):
        #Initiate the MinMaxScale
        scaler = MinMaxScaler()
        #Print the scaling function
        print ('Scaling function:'),(scaler.fit(self.X))
        #Scale the data as defined by the scaler
        Xscaled = scaler.transform(self.X)
        #Initiate  VarianceThreshold
        select = VarianceThreshold(threshold=thresh)
        #Fit the independent variables
        select.fit(Xscaled)  
        #Get the selected features from get_support as a boolean list with True or False  
        selectedFeatures = select.get_support()
        #Create a list to hold discarded columns
        discardL = []
        #print the selected features and their variance
        print ('\nSelected features:')
        for sf in range(len(selectedFeatures)):
            if selectedFeatures[sf]:
                print ('    '),(self.columns[sf]), (select.variances_[sf])
            else:
                discardL.append([self.columnsX[sf],select.variances_[sf]])
        
        print ('\nDiscarded features:')
        for sf in discardL:
            print ('    '),(sf[0]), (sf[1])
        #Redo the list of discarded features to only contain the column name
        discardL = [item[0] for item in discardL]
        #Set the list of discarded features to all defined models
        for key in self.modD:
            self.modelDiscardD[key] = discardL

    def UnivariateSelector(self,kselect):
        #Initiate  SelectKBest
        select = SelectKBest(score_func=f_regression, k=kselect)
        #Fit the independent variables
        select.fit(self.X, self.y)
        #Get the selected features from get_support as a boolean list with True or False
        selectedFeatures = select.get_support()
        #Create a list to hold discarded columns
        discardL = []
        #print the selected features and their variance
        print ('\nSelected features:')
        for sf in range(len(selectedFeatures)):
            if selectedFeatures[sf]:
                print ('    '),(self.columns[sf]), (select.pvalues_ [sf])
            else:
                discardL.append([self.columnsX[sf],select.pvalues_ [sf]])
        print ('\nDiscarded features:')
        for sf in discardL:
            print ('    '),(sf[0]), (sf[1])
        #Redo the list of discarded features to only contain the column name
        discardL = [item[0] for item in discardL]
        #Set the list of discarded features to all defined models
        for key in self.modD:
            self.modelDiscardD[key] = discardL
           
    def RFESelector(self,kselect):
        for mod in self.models:
            select = RFE(estimator=mod[1], n_features_to_select=kselect, step=1)
            select.fit(self.X, self.y)
            selectedFeatures = select.get_support()
            #Create a list to hold discarded columns
            discardL = []
            #print the selected features and their variance
            print ('\nRegressor: %(m)s' %{'m':mod[0]})
            print ('\nSelected features:')
            for sf in range(len(selectedFeatures)):
                if selectedFeatures[sf]:
                    print ('    '),(self.columns[sf])
                else:
                    discardL.append(self.columnsX[sf])
            print ('\nDiscarded features:')
            for sf in discardL:
                print ('    '),(sf)
            self.modelDiscardD[mod[0]] = discardL
            
if __name__ == ('__main__'):
    columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    target = 'MEDV'
    #Initiate the RegressionModels class
    regmods = RegressionModels(columns,target)
    #Import the dataset
    regmods.ImportUrlDataset('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data')
    #Rearrange the data in independent and dependent features
    regmods.ExtractDf([])
    #define the models to use
    regmods.modD = {}
    regmods.modD['OLS'] = {}
    regmods.modD['DecTreeRegr'] = {}
    regmods.modD['RandForRegr'] = {'n_estimators':30}
    regmods.modD['SVR'] = {'kernel':'linear','C':1.5,'epsilon':0.05}
    #Invoke the models
    regmods.ModelSelectSet()
    #Run the feature selection process  
    regmods.VarianceSelector(0.1)

    regmods.UnivariateSelector(2)
    kselect = 5
    regmods.RFESelector(kselect)

    print ('Summary discarded features')
    for key in regmods.modelDiscardD:
        print ( '    %s: %s' %(key, regmods.modelDiscardD[key]) )
    #Run the modelling
    regmods.RegrModTrainTest()
    regmods.RegrModKFold()

