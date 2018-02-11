'''
Created on 9 Feb 2018

@author: thomasgumbricht
'''

import pandas as pd
import numpy as np

from sklearn.linear_model import LarsCV, RidgeCV, LassoCV, LassoLarsCV, ElasticNetCV
from sklearn.linear_model import LinearRegression
from sklearn import model_selection
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


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
        self.modelSpearateX = False

    def _ImportUrlDataset(self,url):    
        self.dataframe = pd.read_csv(url, delim_whitespace=True, names=self.columns)

    def _SetDf(self,df):
        self.dataframe = df
        
    def _ExtractDf(self,omitL):
        #extract the target column as y
        self.y = self.dataframe[self.target]
        #appeld the target to the list of features to be omitted
        omitL.append(self.target)
        #define the list of data to use
        self.columnsX = [item for item in self.dataframe.columns if item not in omitL]
        #extract the data columns as X
        self.X = self.dataframe[self.columnsX]
        
    def _RemoveNan(self, omitL=[]):
        '''Remove Nan from dataframe (self.df)
        '''
        if self.dataframe.isnull().any().any():
            print 'Warning: There are null values in your dataframe\n    removing null records'
        #In some cases the pd.dropna is not sufficient for removing Nan
        #it is safer to use numpy
        columns = list(self.dataframe)
        arr = self.dataframe.values
        arr = arr[~np.isnan(arr).any(axis=1)]        
        self.dataframe = pd.DataFrame(data=arr, columns=columns)
        if self.dataframe.isnull().any().any():
            exit('There are null values in your dataframe that can not be removed')
        #reset X and y
        self.ExtractDf(omitL)
    
    def _SliceColumns(self,n):
        arr = self.dataframe.values
        columns = list(self.dataframe)
        #pop the target
        target = columns.pop(0)
        
        #Extract the target data
        y = arr[:,0]
        #Extract the data array
        X = arr[:,1:]
        Xd = X[:,::n] 
        columns = np.array(columns[1:arr.shape[1]])
        columns = columns[::n].tolist()
 
        columns.insert(0, target)
        self.columns = columns
        self.dataframe =  pd.DataFrame(data=np.column_stack((y,Xd)), columns=self.columns)
        #reset X and y
        self.ExtractDf([])

    def _SumColumns(self,n,targetCol=0,omitL=[]):
        '''Summarizes n columns to a new column and reduces the nr of columns,
        '''
        print 'Reducing X dimensionality by summing every %(n)d columns' %{'n':n}
        arr = self.dataframe.values
        columns = list(self.dataframe)
        #pop the target
        target = columns.pop(0)
        y = arr[:,0]
        #Extract the data array
        X = arr[:,1:]
        if X.shape[1] % n:
            for m in range(n):
                if not (X.shape[1]-m) % n:
                    skip = m
                    break
            X = X[:,skip:]    
            columns = columns[skip:]
        rows = X.shape[0]
        Xd = X.reshape(rows, -1, n).mean(axis=2)
        columns = columns[::n]
        columns.insert(0, target)
        self.columns = columns
        self.dataframe =  pd.DataFrame(data=np.column_stack((y,Xd)), columns=self.columns)
        #reset X and y
        self.ExtractDf(omitL)
        
    def _GetDerivateX(self):
        '''Get the derivate (difference) along the columns
        '''
        arr = self.dataframe.values
        columns = list(self.dataframe)
        #pop the target
        target = columns.pop(0)
        #pop the first remaining column
        columns.pop(0)
        #rename all columns
        columns = ['%(i)s-d' %{'i':item} for item in columns]
        #reinsert the target
        columns.insert(0, target)
        self.columns = columns
        #extract y from the array
        y = arr[:,0]
        #Extract the X data array
        X = arr[:,1:]
        #Calcculate the derivate
        Xd = np.diff(X)
        self.dataframe =  pd.DataFrame(data=np.column_stack((y,Xd)), columns=self.columns)
        print Xd.shape
        #Extract x and y, no omitting as this is derived data
        self.ExtractDf([])
            
    def _Explore(self, nrRows=12, shape=True, head=True, descript=True):
        if shape:
            print(self.dataframe.shape)
        if head:
            print(self.dataframe.head(nrRows))
        if descript:
            print (self.dataframe.describe())
    
    def _PlotRegr(self, obs, pred, title, color='black'):
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
        self.modelD = {}
        if 'OLS' in modD:
            self.modelD['OLS'] = {'mod':LinearRegression(**modD['OLS'])}
        if 'LarsCV' in modD:
            self.modelD['LarsCV'] = {'mod':LarsCV(**modD['LarsCV'])}
        if 'RidgeCV' in modD:
            self.modelD['RidgeCV'] = {'mod':RidgeCV(**modD['RidgeCV'])}
        if 'LassoCV' in modD:
            self.modelD['LassoCV'] = {'mod':LassoCV(**modD['LassoCV'])}
        if 'LassoLarsCV' in modD:
            self.modelD['LassoLarsCV'] = {'mod':LassoLarsCV(**modD['LassoLarsCV'])}
        if 'ElasticNetCV' in modD:
            self.modelD['ElasticNetCV'] = {'mod':ElasticNetCV(**modD['ElasticNetCV'])}
     
    def LarsCVFeatureSelect(self, plot=False):
        '''LarsCV feature selection and regression using the full dataset
        '''
        #Set the model to LarsCV
        mod = LarsCV()
        name = 'LarsCV'
        print 'Model', name
        #Define SelectFromModel using the model (LarsCV) and a threshold
        select = SelectFromModel(mod, threshold="mean")
        #Fit_transform the selection, it will return an X array only retaining features meeting the set threshold
        X_new = select.fit_transform(self.X,self.y)
        #Print the shape of the returned X_new array
        print 'X shape (new)', X_new.shape
        #Get the number of features that was selected
        nFeat = X_new.shape[1]
        #Print the support of the selection (a list of boolean results for selected and discarded features)
        print 'support',select.get_support()
        #Create a dictionary from the original X columns and the boolean support list identifying the selected features
        selectD = dict(zip(self.columnsX,select.get_support()))
        #Extract the names of the selected features to a list
        featureL = [item for item in selectD if selectD[item]]
        #Print the selected features
        print 'Selected features', featureL
        #Fit the regression
        fit = mod.fit(X_new, self.y)
        #Print the parameters of the fitted regression
        print 'fitted params', fit.get_params()
        #Print the coefficiens of the fitted regression
        print 'coefficients', fit.coef_
        #Predict the target
        predict = mod.predict(X_new)
        #Prepare a message for printing out the results
        title = ('Target: %(tar)s; Model: %(mod)s; n:%(n)s; RMSE: %(rmse)1.2f; r2: %(r2)1.2f' \
                      % {'tar':self.target,'mod':name,'n': nFeat, 'rmse':mean_squared_error(self.y, predict),'r2': r2_score(self.y, predict)} )
        msg = '    Regression results\n        %(t)s' %{'t':title}
        #Print the result message
        print (msg)
        if plot:
            self._PlotRegr(self.y, predict, title, color='maroon')

    def LarsCVFeatureSelectPipeline(self, plot=False):
        '''LarsCV feature selection and regression using the full dataset
        '''
        #Set the model to LarsCV
        mod = LarsCV()
        name = 'LarsCV'
        print 'Model', name
        #Define SelectFromModel using the model (LarsCV) and a threshold
        select = SelectFromModel(mod, threshold="mean")
        #Setup a pipeline linking the feature selection and the model
        #At this stage the names of each step can be set to anything ('select' and 'regr')
        pipeline = Pipeline([
            ('select',select),
            ('regr',mod)
        ])
        #Fit the pipeline, each step will do a fit_transform and send the result to the next step
        pipeline.fit(self.X, self.y)
        #All the parameters from each step in the pipeline can be retrieved
        #Print the support (list of selected and discarded) features from the 'select' step (step = 0)
        print 'step 0 support',pipeline.steps[0][1].get_support()
        #Each step can also be accessed using tis name
        print 'step select support', pipeline.named_steps['select'].get_support()
        #Create a dictionary from the original X columns and the boolean support list identifying the selected features
        selectD = dict(zip(self.columnsX,pipeline.named_steps['select'].get_support()))
        #Extract the names of the selected features to a list
        featureL = [item for item in selectD if selectD[item]]
        #Print the selected features
        print 'Selected features', featureL
        #Get the number of features that was selected
        nFeat = len(featureL)
        #Print the parameters of the fitted regression
        print 'fitted params', pipeline.steps[1][1].get_params()
        #Print the coefficiens of the fitted regression
        print 'coefficients', pipeline.named_steps['regr'].coef_
        #Predict the target
        predict = pipeline.predict(self.X)
        #Prepare a message for printing out the results
        title = ('Target: %(tar)s; Model: %(mod)s; n:%(n)s; RMSE: %(rmse)1.2f; r2: %(r2)1.2f' \
                      % {'tar':self.target,'mod':name,'n': nFeat, 'rmse':mean_squared_error(self.y, predict),'r2': r2_score(self.y, predict)} )
        msg = '    Regression results\n        %(t)s' %{'t':title}
        #Print the result message
        print (msg)
        if plot:
            self._PlotRegr(self.y, predict, title, color='maroon')

    def LarsCVFeatureSelectPipelineCV(self, plot=False):
        '''LarsCV feature selection and regression using the full dataset
        '''
        #Set the model to LarsCV
        mod = LarsCV()
        name = 'LarsCV'
        #Define SelectFromModel using the model (LarsCV) and a threshold
        select = SelectFromModel(mod)
        #Setup a pipeline linking the feature selection and the model
        #The names will be used for linking parameters to each step
        pipeline = Pipeline([
            ('select',select),
            ('regr',mod)
        ])
        #Define the parameters to test in a grid search for optimizing hyper-parameters
        #The first part of the key must correpspond to a step in the pipleine, 
        #and the last part must correspond to parameter accepted by the function defined in that step
        parameters = dict(select__threshold=["0.5*mean","0.75*mean", "mean","1.5*mean","1.25*mean"])
        #Define the grid search, set the pipeline as the estimator, 
        #and the parameters defined above as the param_grid
        cv = GridSearchCV(pipeline, param_grid=parameters, verbose=1)
        #Fit the cross validation search
        cv.fit(self.X, self.y)
        #All the parameters from each step in the bot the crossvalidation and the pipeline can be retrieved
        #Retrieve and print the best results from the grid search
        for i in range(1, 5):
            candidates = np.flatnonzero(cv.cv_results_['rank_test_score'] == i)
            for candidate in candidates:
                print("Model with rank: {0}".format(i))
                print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                      cv.cv_results_['mean_test_score'][candidate],
                      cv.cv_results_['std_test_score'][candidate]))
                print("Parameters: {0}".format(cv.cv_results_['params'][candidate]))
                print("")
        #print the best resutls from the cross validation search       
        print 'cv best estimator:', cv.best_estimator_
        print 'cv best score:', cv.best_score_
        print 'cv best params:', cv.best_params_
        print 'cv best index:', cv.best_index_
        #Use the best estimator (cv.best_estimator) for retrieving the corresponding parameters in the pipleine
        #Print the support (list of selected and discarded) features from the 'select' step (step = 0)
        print 'step 0 support', cv.best_estimator_.steps[-2][1].get_support()
        #Each step can also be accessed using tis name
        print 'step select support', cv.best_estimator_.named_steps['select'].get_support()
        #Create a dictionary from the original X columns and the boolean support list identifying the selected features
        selectD = dict(zip(self.columnsX, cv.best_estimator_.named_steps['select'].get_support() ))
        #Extract the names of the selected features to a list
        featureL = [item for item in selectD if selectD[item]]
        #Print the selected features
        print 'Selected features', featureL
        #Get the number of features that was selected
        nFeat = len(featureL)
        #Print the parameters of the fitted regression
        print 'fitted params', cv.best_estimator_.steps[-1][1].get_params()
        #Print the coefficiens of the fitted regression
        print 'coefficients', cv.best_estimator_.named_steps['regr'].coef_
        #Predict the target
        predict = cv.predict(self.X)
        #Prepare a message for printing out the results
        title = ('Target: %(tar)s; Model: %(mod)s; n:%(n)s; RMSE: %(rmse)1.2f; r2: %(r2)1.2f' \
                      % {'tar':self.target,'mod':name,'n': nFeat, 'rmse':mean_squared_error(self.y, predict),'r2': r2_score(self.y, predict)} )
        msg = '    Regression results\n        %(t)s' %{'t':title}
        #Print the result message
        print (msg)
        if plot:
            self._PlotRegr(self.y, predict, title, color='maroon')
     
    def ReportCVSearch(self, cv, n_top=3):
        results = cv.cv_results_
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results['rank_test_score'] == i)
            for candidate in candidates:
                print("    Model with rank: {0}".format(i))
                print("        Mean validation score: {0:.3f} (std: {1:.3f})".format(
                      results['mean_test_score'][candidate],
                      results['std_test_score'][candidate]))
                print("        Parameters: {0}".format(results['params'][candidate]))
        print '    cv best estimator:', cv.best_estimator_
        print '    cv best score:', cv.best_score_
        print '    cv best params:', cv.best_params_
        #Create a dictionary from the original X columns and the boolean support list identifying the selected features
        selectD = dict(zip(self.columnsX, cv.best_estimator_.named_steps['select'].get_support() ))
        #Extract the names of the selected features to a list
        featureL = [item for item in selectD if selectD[item]]
        #Print the selected features
        print '    Selected features', featureL
        #For models that expose "coef_", print "coef_"
        if hasattr(cv.best_estimator_.named_steps['regr'],'coef_'):
            #Print the coefficiens of the fitted regression
            print '    coefficients', cv.best_estimator_.named_steps['regr'].coef_    
        
    def PipelineLinearCVRegressors(self, testsize= 0.3, plot=False, verbose=1):
        '''Feature selection and regression with linear regressors with inbuilt CV
        '''   
        bestScore = -99    
        X_train, X_test, y_train, y_test = model_selection.train_test_split(self.X, self.y, test_size=testsize)
        for m in self.modelD:                
            #Define SelectFromModel using the model 
            select = SelectFromModel(self.modelD[m]['mod'])
            #Setup a pipeline linking the feature selection and the model
            pipeline = Pipeline([
                ('select',select),
                ('regr',self.modelD[m]['mod'])
            ])
            #Define the parameters to test in a grid search for optimizing hyper-parameters
            parameters = dict(select__threshold=["0.5*mean","0.75*mean", "mean","1.5*mean","2*mean"])     
            #Define the grid search, set the pipeline as the estimator
            cv = GridSearchCV(pipeline, param_grid=parameters, verbose=verbose)
            #Fit the cross validation search
            cv.fit(X_train, y_train)
            #Create a dictionary from the original X columns and the boolean support list identifying the selected features
            selectD = dict(zip(self.columnsX, cv.best_estimator_.named_steps['select'].get_support() ))
            #Extract the names of the selected features to a list
            featureL = [item for item in selectD if selectD[item]]
            if cv.best_score_ > bestScore:
                bestCV = cv
                bestModel = m

            if verbose:
                print 'Model',m
                #Retrieve and print the best results from the grid search
                if verbose > 1:
                    self.ReportCVSearch(cv,3)
                
                #Print the selected features
                print '    Selected features', featureL
                #Get the number of features that was selected
                nFeat = len(featureL)
                #Print the coefficiens of the fitted regression
                print '    coefficients', cv.best_estimator_.named_steps['regr'].coef_    
                #Predict the target
                predict = cv.predict(X_test)
                #Prepare a message for printing out the results
                title = ('Target: %(tar)s; Model: %(mod)s; n:%(n)s; RMSE: %(rmse)1.2f; r2: %(r2)1.2f' \
                              % {'tar':self.target,'mod':m,'n': nFeat, 'rmse':mean_squared_error(y_test, predict),'r2': r2_score(y_test, predict)} )
                msg = '    Regression results\n        %(t)s' %{'t':title}
                #Print the result message
                print (msg)
            if plot:
                self._PlotRegr(self.y, predict, title, color='maroon')
        return (bestModel,bestCV)
                       
    def ReportModParams(self):
        print 'Model hyper-parameters:'
        for m in self.models:
            #Retrieve the model name and the model itself
            name,mod = m
            print ('    name'), (name), (mod.get_params())
   
if __name__ == ('__main__'):
    columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    target = 'MEDV'
    regmods = RegressionModels(columns,target)
    regmods._ImportUrlDataset('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data')
    regmods._ExtractDf([])

    #LarxCV regression with manual steps
    regmods.LarsCVFeatureSelect()
    #LarxCV regression with pipeline
    regmods.LarsCVFeatureSelectPipeline()
    #LarxCV regression with pipeline embedded in cross validation search
    regmods.LarsCVFeatureSelectPipelineCV(True)

    #Set up models to test in general function
    modD = {}
    modD['OLS'] = {}
    modD['LarsCV'] = {}
    modD['RidgeCV'] = {'alphas':[0.1, 1.0, 10.0]}
    modD['LassoCV'] = {'alphas':[0.1, 1.0, 10.0]}
    modD['LassoLarsCV'] = {}
    modD['ElasticNetCV'] = {'alphas':[0.1, 1.0, 10.0], 'l1_ratio':[0.01, 0.25, 0.5, 0.75, 0.99]}
    regmods.ModelSelectSet(modD)
    #Test all setup models
    bestModel,bestCV = regmods.PipelineLinearCVRegressors(0.3, False, 0)
    print 'Best model:', bestModel
    regmods.ReportCVSearch(bestCV, 1)

    