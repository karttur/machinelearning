#numpy: for handling 2d (array) data
import numpy as np
#Pandas: for reading and organizing the data
import pandas as pd
#Sci-kit learn (sklearn) machinelarning package (selected sub-packges and modules)
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot
from sklearn.model_selection import train_test_split

class RegressionModel:
    '''Machinelearning using linear regression
    '''
    def __init__(self,columns,target):
        '''creates an empty instance of RegressionMode
        '''  

        self.columns = columns
        self.target = target

    def ImportSklearnDataset(self):
        #the sci-kit (sklearn) package already includes the data, just import it
        from sklearn import datasets
        #Load the Boston dataset from the datasets package library
        dataset = datasets.load_boston()        
        #The sklearn organised dataset divides the dataset between the independent variables ("data") and the dependent variable ("target").
        #To create a pandas dataframe you have to stack them
        self.dataframe =  pd.DataFrame(data=np.column_stack((dataset.data,dataset.target)), columns=self.columns)
    
    def ImportUrlDataset(self,url):    
        self.dataframe = pd.read_csv(url, delim_whitespace=True, names=self.columns)
    
    def Explore(self, shape=True, head=True, descript=True):
        if shape:
            print(self.dataframe.shape)
        if head:
            print(self.dataframe.head(12))
        if descript:
            print (self.dataframe.describe())
    
    def PlotExplore(self, histo = True, box = True):
        from math import sqrt,ceil
        if histo:
            self.dataframe.hist()
            pyplot.show()
        if box:
            nrows = int(round(sqrt(len(self.dataframe.columns))))
            ncols = int(ceil(len(self.dataframe.columns)/float(nrows)))
            self.dataframe.plot(kind='box', subplots=True, layout=(ncols,nrows))
        pyplot.show()
    
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
    
    def ExtractDf(self,omitL):
        #extract the target column as y
        self.y = self.dataframe[self.target]
        #append the target to the list of features to be omitted
        omitL.append(target)
        #define the list of data to use
        self.columnsX = [item for item in self.dataframe.columns if item not in omitL]
        #extract the data columns as X
        self.X = self.dataframe[self.columnsX]
    
    def MultiRegression(self, plot=True):
        regr = LinearRegression()
        #Fit the regression to all the data
        regr.fit(self.X, self.y)
        #Predict the independent variable
        predict = regr.predict(self.X)
        #The coefficients
        print('Coefficients: \\n', regr.coef_)
        #The mean squared error
        print("Mean squared error: %.2f" \
            % mean_squared_error(self.y, predict))
        #Explained variance score: 1 is perfect prediction
        print('Variance score: %.2f' \
            % r2_score(self.y, predict))
        if plot:
            title = ('Mean squared error: %(rmse)2f; Variance score: %(r2)2f' \
                      % {'rmse':mean_squared_error(self.y, predict),'r2': r2_score(self.y, predict)} )
            self.PlotRegr(self.y, predict, title, color='maroon')
    
    def MultiRegressionModel(self,testsize=0.3, plot=True):
        #Split the dataset into training and test
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=testsize)
        regr = LinearRegression()
        #Fit the regression to all the data
        regr.fit(X_train, y_train)
        #Predict the independent variable
        predict = regr.predict(X_test)
        #The coefficients
        print('Coefficients: \\n', regr.coef_)
        #The mean squared error
        print("Mean squared error: %.2f" \
            % mean_squared_error(y_test, predict))
        #Explained variance score: 1 is perfect prediction
        print('Variance score: %.2f' \
            % r2_score(y_test, predict))
        if plot:
            title = ('Mean squared error: %(rmse)2f; Variance score: %(r2)2f' \
                      % {'rmse':mean_squared_error(y_test, predict),'r2': r2_score(y_test, predict)} )
            self.PlotRegr(y_test, predict, title, color='green')
    
    def MultiRegressionKfoldModel(self,folds=10, plot=True):
        regr = LinearRegression()
        #Set the kfold
        kfold = model_selection.KFold(n_splits=folds)
        #cross_val_predict returns an array of the same size as `y` where each entry
        #is a prediction obtained by cross validation:
        predict = model_selection.cross_val_predict(regr, self.X, self.y, cv=kfold)
        #To retrieve regressions scores, use cross_val_score
        scoring = 'r2'
        r2 = model_selection.cross_val_score(regr, self.X, self.y, cv=kfold, scoring=scoring)
        #The correlation coefficient
        print('Regression coefficients: \n', r2)
        #The mean squared error
        print("Mean squared error: %.2f" \
            % mean_squared_error(self.y, predict))
        #Explained variance score: 1 is perfect prediction
        print('Variance score: %.2f' \
            % r2_score(self.y, predict))
        if plot:
            title = ('Mean squared error: %(rmse)2f; Variance score: %(r2)2f' \
                      % {'rmse':mean_squared_error(self.y, predict),'r2': r2_score(self.y, predict)} )
            self.PlotRegr(self.y, predict, title, color='blue')
    
    

if __name__ == ('__main__'):
    columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    target = 'MEDV'
    regmod = RegressionModel(columns,target)
    #either
    #regmod.ImportSklearnDataset()
    #or
    regmod.ImportUrlDataset('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data')
    regmod.Explore()
    regmod.PlotExplore()
    target ='MEDV'
    omitL =[]
    regmod.ExtractDf(omitL)
    print regmod.dataframe.shape
    print regmod.y.shape
    print regmod.X.shape
    regmod.MultiRegression()
    regmod.MultiRegressionModel(0.2)
    regmod.MultiRegressionKfoldModel()
