---
layout: post
title: Machine learning 4 - Parameter tuning
modified: '2018-02-03 18:16'
categories: blog
excerpt: Hyper-parameter tuning of machine learning models in Python
tags:
  - rainfall change
  - change
  - GIS installations
  - macOS
sensors:
  - TRMM
regions:
  - tropics
image: avg-trmm-3b43v7-precip_3B43_trmm_2001-2016_A
figure1A: machinelearning4_knnregr_random_tt
figure1B: machinelearning4_knnregr_random_kfold
figure2A: machinelearning4_dectreeregr_random_tt
figure2B: machinelearning4_dectreeregr_random_kfold
figure3A: machinelearning4_svr_random_tt
figure3B: machinelearning4_svr_random_kfold
figure4A: machinelearning4_randforregr_random_tt
figure4B: machinelearning4_randforregr_random_kfold
figure5A: machinelearning4_knnregr_exhaust_tt
figure5B: machinelearning4_knnregr_exhaust_kfold
figure6A: machinelearning4_dectreeregr_exhaust_tt
figure6B: machinelearning4_dectreeregr_exhaust_kfold
figure7A: machinelearning4_svr_exhaust_tt
figure7B: machinelearning4_svr_exhaust_kfold
figure8A: machinelearning4_randforregr_exhaust_tt
figure8B: machinelearning4_randforregr_exhaust_kfold
date: '2018-02-04 10:55'
comments: true
share: true
---

<script src="https://karttur.github.io/common/assets/js/karttur/togglediv.js"></script>
**Contents**
	\- [Introduction](#introduction)
	\- [Prerequisits](#prerequisits)
	\- [Python module skeleton](#python-module-skeleton)
	\- [Model hyper-parameters](#model-hyper-parameters)
		\- [KNeighborsRegressor](#kneighborsregressor)
		\- [DecisionTreeRegressor](#decisiontreeregressor)
		\- [Support Vector Machine Regressor (SVR)](#support-vector-machine-regressor-svr)
		\- [RandomForestRegressor](#randomforestregressor)
	\- [Tuning the parameters](#tuning-the-parameters)
		\- [Report function](#report-function)
	\- [Model parameterization](#model-parameterization)
		\- [KNeighborsRegressor](#kneighborsregressor)
		\- [DecisionTreeRegressor](#decisiontreeregressor)
		\- [Support Vector Machine Regressor (SVR)](#support-vector-machine-regressor-svr)
		\- [RandomForestRegressor](#randomforestregressor)
		\- [Model Predictions](#model-predictions)
	\- [Resources](#resources)


## Introduction

If you have followed Karttur's blog posts on machine learning, you have learnt to apply different regressors for predicting continuous variables, and how to discriminate among the independent variables. But so far you have either only used the default parameters (called hyper-parameters in Scikit learn) defining the regressors, or tested a few other settings. So while the regressors that you applied have used training data (or folded cross validation) for the internal parameters linking the independent variables to the target feature, the hyper-parameters defining how this is done have been fixed.

The Scikit learn package includes two modules for tuning (or optimizing) the hyper-parameter settings, [Exhaustive Grid Search (GridSearchCV)](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV) and [Randomized search (RandomizedSearchCV)](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV). You get an introduction to both at the Scikit learn page on [tuning the hyper-parameters](http://scikit-learn.org/stable/modules/grid_search.html#grid-search-tips).

As in the previous Karttur posts on machine learning, this post will evolve around creating a Python module (.py file) for applying the tuning of hyper-parameters for different regressors.

You can copy and paste the code in the post, but you might have to edit some parts: indentations does not always line up, quotations marks can be of different kinds and are not always correct when published/printed as html text, and the same also happens with underscores and slashes.

## Prerequisits

The prerequisites are the same as in the previous posts in this series: a Python environment with numpy, pandas, sklearn (Scikit learn) and matplotlib installed.

## Python module skeleton

As in the previous posts on machine learning, we start by a Python module (.py file) skeleton that will be used as a vehicle for developing a complete sklearn hyper-parameter selection module. The skeleton code is hidden under the button.

<button id= "toggle01btn" onclick="hiddencode('toggle01')">Hide/Show module skeleton</button>
<div id="toggle01" style="display:none">
{% capture text-capture %}
{% raw %}
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib
matplotlib.use(\'TkAgg\')
from matplotlib import pyplot

class RegressionModels:
    \'\'\'Machinelearning using regression models
    \'\'\'
    def \_\_init\_\_(self, columns,target):
        \'\'\'creates an instance of RegressionModels
        \'\'\'
        self.columns = columns
        self.target = target
        #create an empty dictionary for features to be discarded by each model
        self.modelDiscardD = {}

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

    def ModelSelectSet(self,modD):
        self.models = []
        if 'OLS' in modD:
            self.models.append(('OLS', linear_model.LinearRegression(\*\*modD['OLS'])))
            self.modelDiscardD['OLS'] = []
        if 'TheilSen' in modD:
            self.models.append(('TheilSen', linear_model.TheilSenRegressor(\*\*modD['TheilSen'])))
            self.modelDiscardD['TheilSen'] = []
        if 'Huber' in modD:
            self.models.append(('Huber', linear_model.HuberRegressor(\*\*modD['Huber'])))
            self.modelDiscardD['Huber'] = []
        if 'KnnRegr' in modD:
            self.models.append(('KnnRegr', KNeighborsRegressor( \*\*modD['KnnRegr'])))
            self.modelDiscardD['KnnRegr'] = []
        if 'DecTreeRegr' in modD:
            self.models.append(('DecTreeRegr', DecisionTreeRegressor(\*\*modD['DecTreeRegr'])))
            self.modelDiscardD['DecTreeRegr'] = []
        if 'SVR' in modD:
            self.models.append(('SVR', SVR(\*\*modD['SVR'])))
            self.modelDiscardD['SVR'] = []
        if 'RandForRegr' in modD:
            self.models.append(('RandForRegr', RandomForestRegressor( \*\*modD['RandForRegr'])))
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
            #cross_val_predict returns an array of the same size as \`y\` where each entry
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

if \_\_name\_\_ == ('\_\_main\_\_'):
    columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    target = 'MEDV'
    regmods = RegressionModels(columns, target)
    regmods.ImportUrlDataset('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data')
```
{% endraw %}
{% endcapture %}
{% include widgets/toggle-code.html  toggle-text=text-capture  %}
</div>

## Model hyper-parameters

All sklearn models have a different suite of hyper-parameters that can be set. These parameters can be of four types:

* Integers
* Real (or float)
* Lists of alternatives
* Boolean (True or False)

When tuning hyper-parameters the first thing to decide is which parameters to tune. You can find out which hyper-parameters that can be passed to all sklearn models in the Scikit pages for each regressor. But you can also get them as a dictionary in Python, and you will explore them further down. Create the function <span class='pydef'>RandomTuningParams</span>, under the class <span class='pydef'>RegressionModels</span>. At first you will only use the function for exploring the parameters to set, the actual parameter settings for tuning will be added later.

```
    def RandomTuningParams(self):
        # specify parameters and distributions to sample from
        for m in self.models:
            name,mod = m
            print ('name'), (name), (mod.get_params())
```

### KNeighborsRegressor

To explore the hyper-parameters of Scikit learn regressors, define the models you want to explore, invoke them, and call the <span class='pydef'>Tuningparameters</span> function to see the hyper-parameters. The first example is for exploring the parameters for <span class='pydef'>KNeighborsRegressor</span> that is abbreviated 'KnnRegr' when added to the _modD_ dictionary. Add the lines below to the \_\_main\_\_ section.

```
    regmods.modD = {}
    regmods.modD['KnnRegr'] = {}
    #Invoke the models
    regmods.ModelSelectSet()
    #Tuning parameters
    regmods.RandomTuningParams(11)
```

Run the module, and check the listed hyper-parameters and their default values.

```
name KnnRegr {'n_neighbors': 5, 'n_jobs': 1, 'algorithm': 'auto', 'metric': 'minkowski',
'metric_params': None, 'p': 2, 'weights': 'uniform', 'leaf_size': 30}

```

### DecisionTreeRegressor

Add the 'DecTreeRegr' (<span class='package'>DecisionTreeRegressor</span>) regressor to the _modD_ dictionary
```
    regmods.modD['DecTreeRegr'] = {}
```
The model hyper-parameters for the <span class='package'>DecisionTreeRegressor</span>:
```
name DecTreeRegr {'presort': False, 'splitter': 'best', 'min_impurity_decrease': 0.0, 'max_leaf_nodes': None,
'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'criterion': 'mse',
'random_state': None, 'min_impurity_split': None, 'max_features': None, 'max_depth': None}
```

### Support Vector Machine Regressor (SVR)

Add the 'SVR' (<span class='package'>SVR</span>) to the _modD_ dictionary
```
    regmods.modD['SVR'] = {}
```
The model hyper-parameters for the <span class='package'>SVR</span>:
```
name SVR {'kernel': 'rbf', 'C': 1.0, 'verbose': False, 'degree': 3, 'epsilon': 0.1, 'shrinking': True,
'max_iter': -1, 'tol': 0.001, 'cache_size': 200, 'coef0': 0.0, 'gamma': 'auto'}

```

### RandomForestRegressor

Add the 'RandForRegr' (<span class='package'>RandomForestRegressor</span>) regressor to the _modD_ dictionary
```
    regmods.modD['RandForRegr'] = {}
```
The model hyper-parameters for the <span class='package'>RandomForestRegressor</span>:
```
name RandForRegr {'warm_start': False, 'oob_score': False, 'n_jobs': 1, 'min_impurity_decrease': 0.0,
'verbose': 0, 'max_leaf_nodes': None, 'bootstrap': True, 'min_samples_leaf': 1, 'n_estimators': 10,
'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'criterion': 'mse', 'random_state': None,
'min_impurity_split': None, 'max_features': 'auto', 'max_depth': None}
```

## Tuning the parameters

Before you can use the module for tuning the hyper-parameters, you must create a reporting function, and the functions for setting the hyper-parameters to tune.

### Report function

Add the reporting function (<span class='pydef'>ReportSearch</span>).
```
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
```

### Randomized tuning

As I wanted to try both randomized and exhaustive tuning, I opted for creating a separate function for each method. The function <span class='pydef'>RandomTuning</span> invokes the randomized tuning search, prints the results of the tuning search, and then also sets the highest ranked hyper-parameter setting as the parameters for each model (by updating the _modD_ dictionary).

```
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
```

Without setting any parameters, the tuning search for each model is defaulted to use half of the dataset (parameter _fraction=0.5) for the tuning, 6 iterations (parameter _nIterSearch_=6), and to print out the top 3 results (parameter _n_top_=3). For each regressor, the hyper-parameters for the best tuning are retrieved. If the regressor model had any initial hyper-parameters set in the _modD_ dictionary they are added, and the tuned hyper-parameters are then set as the parameter+value pairs in _modD_.

The <span class='pydef'>RandomTuning</span> function uses the Scikit learn randomized tuning function <span class='package'>RandomizedSearchCV</span>, that you must add to the imports at the beginning of the module. Then you will also need to import functions for creating ranges of randomized numbers.

```
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_randreal
from sklearn.model_selection import RandomizedSearchCV
```

You also have to call <span class='pydef'>RandomTuning</span> from the \_\_main\_\_ section
```
regmods.RandomTuning()
```
If you want to increase the search iterations to 12, and the print out the top 6 results, but keep the fraction of the dataset at 0.5, add that to the call.
```
regmods.RandomTuning(0.5,12,6)
```

Then you have to create the values for the parameter _param_distributions_ used in <span class='pydef'>RandomTuning</span> (_param_distributions=self.paramDist_). The values to send to _param_distributions_ are defined in the variable _self.paramDist_, and defines both which hyper-parameters to tune, and what values each hyper-parameter is allowed to take. The only parameter (in _self.paramDist_) that is the same across all regressors is the _criterion_ to use for evaluate the performance of each combination of hyper-parameters in the tuning search. You can either set the _criterion_, or accept the the default. You have to look at the individual Scikit learn pages to get a grip on the hyper-parameters you want to tune, and what ranges/alternatives you can/want to set. The principle for setting the ranges/parameters differs for the different types of parameters.

* Integers: sp_randint(min, max) or predefined set (i, j, k, ...)
* Real: sp_randreal(min, max) or predefined set (i.j, k.l, m.n, ...)
* Alternatives: ['alt1', 'alt2', 'alt3', ...]
* Boolean: [True, False]

#### KNeighborsRegressor randomized tuning

The <span class='package'>KNeighborsRegressor</span> ('KnnRegr') regressor has fewer hyper-parameters compared to the other non-linear regressors used in here (see above). The code snippet below defines the randomized tuning for the 'KnnRegr' hyper-parameters _n_neighbors_, _leaf_size_, _weight_, _p_ and _algorithm_.
```
    def RandomTuningParams(self):
        # specify parameters and distributions to sample from
        for m in self.models:
            name,mod = m
            print ('name'), (name), (mod.get_params().keys())
            if name == 'KnnRegr':
                self.paramDist = {"n_neighbors": sp_randint(4, 12),
                              'leaf_size': sp_randint(10, 50),
                              'weights': ('uniform','distance'),
                              'p': (1,2),
                              'algorithm': ('auto','ball_tree', 'kd_tree', 'brute')}
```

Run the Pyton module to get the tuned hyper-parameter for 'KnnRegr'. As the process uses a randomizer, the results varies each time you run it, but should resemble the results shown below.

```
Model with rank: 1
Mean validation score: 0.373 (std: 0.158)
Parameters: {'n_neighbors': 7, 'weights': 'distance', 'leaf_size': 28, 'algorithm': 'auto', 'p': 1}

Model with rank: 2
Mean validation score: 0.371 (std: 0.136)
Parameters: {'n_neighbors': 10, 'weights': 'distance', 'leaf_size': 24, 'algorithm': 'ball_tree', 'p': 1}

Model with rank: 2
Mean validation score: 0.371 (std: 0.136)
Parameters: {'n_neighbors': 10, 'weights': 'distance', 'leaf_size': 15, 'algorithm': 'auto', 'p': 1}
```
Transferring the best result from the tuning above ("Model with rank: 1") to the model, the full model hyper-parameter settings for the tuned 'KnnRegr' is shwon below.
```
name KnnRegr {'n_neighbors': 7, 'n_jobs': 1, 'algorithm': 'auto', 'metric': 'minkowski', 'metric_params': None, 'p': 1, 'weights': 'distance', 'leaf_size': 28}
```

#### DecisionTreeRegressor randomized tuning

For the <span class='package'>DecisionTreeRegressor</span> ('KnnRegr') regressor I opted for tuning _max_depth_, _min_samples_split_ and _min_samples_leaf_.
```
            elif name =='DecTreeRegr':
                self.paramDist[name] = {"max_depth": [3, None],
                              "min_samples_split": sp_randint(2, 6),
                              "min_samples_leaf": sp_randint(1, 4)}
```

With the following results:

```
Model with rank: 1
Mean validation score: 0.479 (std: 0.162)
Parameters: {'min_samples_split': 4, 'max_depth': None, 'min_samples_leaf': 3}

Model with rank: 2
Mean validation score: 0.367 (std: 0.206)
Parameters: {'min_samples_split': 3, 'max_depth': 3, 'min_samples_leaf': 3}

Model with rank: 3
Mean validation score: 0.367 (std: 0.206)
Parameters: {'min_samples_split': 5, 'max_depth': 3, 'min_samples_leaf': 3}
```

#### SVR randomized tuning

For the <span class='package'>SVR</span> ('SVR') regressor I opted for tuning _kernel_, _epsilon_, and _C_. Rather than using a randomizer I hardcoded the values open for _epsilon_ and _C_ (with more values the processing takes a very long time).
```
            elif name =='SVR':
                self.paramDist[name] = {"kernel": ['linear', 'rbf'],
                              "epsilon": (0.05, 0.1, 0.2),
                              "C": (1, 2, 5, 10)}
```

With the following results:

```
Model with rank: 1
Mean validation score: 0.724 (std: 0.083)
Parameters: {'kernel': 'linear', 'C': 1, 'epsilon': 0.2}

Model with rank: 2
Mean validation score: 0.714 (std: 0.126)
Parameters: {'kernel': 'linear', 'C': 5, 'epsilon': 0.1}

Model with rank: 3
Mean validation score: 0.041 (std: 0.021)
Parameters: {'kernel': 'rbf', 'C': 10, 'epsilon': 0.05}
```
Note the large difference in validation score between highest ranked ranked models ('linear' _kernel_), and the 3rd model with the 'rbf' _kernel_. The latter also has the largest allowed value of the _C_ hyper-parameter.

#### RandomForestRegressor randomized tuning

For the <span class='package'>RandomForestRegressor</span> ('RandForRegr') regressor I opted for tuning _max_depth_, _n_estimators_, _max_features_, _min_samples_split_ , _min_samples_leaf_ and _bootstrap_.
```
            elif name =='RandForRegr':
                self.paramDist = {"max_depth": [3, None],
                              "n_estimators": sp_randint(10, 50),
                              "max_features": sp_randint(1, max_features),
                              "min_samples_split": sp_randint(2, up_samples_split),
                              "min_samples_leaf": sp_randint(1, up_samples_leaf),
                              "bootstrap": [True,False]}
```

With the following results:

```
Model with rank: 1
Mean validation score: 0.744 (std: 0.075)
Parameters: {'bootstrap': False, 'min_samples_leaf': 3, 'n_estimators': 17, 'min_samples_split': 2, 'max_features': 9, 'max_depth': None}

Model with rank: 2
Mean validation score: 0.727 (std: 0.116)
Parameters: {'bootstrap': True, 'min_samples_leaf': 2, 'n_estimators': 31, 'min_samples_split': 4, 'max_features': 3, 'max_depth': None}

Model with rank: 3
Mean validation score: 0.727 (std: 0.073)
Parameters: {'bootstrap': False, 'min_samples_leaf': 4, 'n_estimators': 13, 'min_samples_split': 5, 'max_features': 6, 'max_depth': 3}
```

#### Randomized Model Predictions

The model setup, using the _modD_ dictionary, allows the tuned hyper-parameters to be set directly to each model. The hyper-parameters of each regressor are updated as part of the function <span class='pydef'>RandomTuning</span>. To invoke the tuned hyper-parameters, you have to reset the models in the \_\_main\_\_ section, and then call either <span class='pydef'>RegrModTrainTest</span> or <span class='pydef'>RegrModKFold</span> function, or both, to run the tuned models for your dataset.

```
  #Reset the models with the tuned hyper-parameters
  regmods.ModelSelectSet()
  #Run the models
  regmods.RegrModTrainTest()
  regmods.RegrModKFold()
```

#### Randomized tuning results

<figure class="half">
	<a href="{{ site.commonurl }}/images/{{ site.data.images[page.figure1A].source }}"><img src="{{ site.commonurl }}/images/{{ site.data.images[page.figure1A].file }}" alt="image"></a>
  <a href="{{ site.commonurl }}/images/{{ site.data.images[page.figure1B].source }}"><img src="{{ site.commonurl }}/images/{{ site.data.images[page.figure1B].file }}" alt="image"></a>

  <figcaption>Hyper-parameter tuned prediction using k nearest neigbhor regression: left, dataset split into training and test subsets; right, folded cross validation.</figcaption>
</figure>

<figure class="half">
	<a href="{{ site.commonurl }}/images/{{ site.data.images[page.figure2A].source }}"><img src="{{ site.commonurl }}/images/{{ site.data.images[page.figure2A].file }}" alt="image"></a>
  <a href="{{ site.commonurl }}/images/{{ site.data.images[page.figure2B].source }}"><img src="{{ site.commonurl }}/images/{{ site.data.images[page.figure2B].file }}" alt="image"></a>

  <figcaption>Hyper-parameter tuned prediction using decision tree regression: left, dataset split into training and test subsets; right, folded cross validation.</figcaption>
</figure>

<figure class="half">
	<a href="{{ site.commonurl }}/images/{{ site.data.images[page.figure3A].source }}"><img src="{{ site.commonurl }}/images/{{ site.data.images[page.figure3A].file }}" alt="image"></a>
  <a href="{{ site.commonurl }}/images/{{ site.data.images[page.figure3B].source }}"><img src="{{ site.commonurl }}/images/{{ site.data.images[page.figure3B].file }}" alt="image"></a>

  <figcaption>Hyper-parameter tuned prediction using support vector machine regression: left, dataset split into training and test subsets; right, folded cross validation.</figcaption>
</figure>

<figure class="half">
	<a href="{{ site.commonurl }}/images/{{ site.data.images[page.figure4A].source }}"><img src="{{ site.commonurl }}/images/{{ site.data.images[page.figure4A].file }}" alt="image"></a>
  <a href="{{ site.commonurl }}/images/{{ site.data.images[page.figure4B].source }}"><img src="{{ site.commonurl }}/images/{{ site.data.images[page.figure4B].file }}" alt="image"></a>

  <figcaption>Hyper-parameter tuned prediction using random forest regression: left, dataset split into training and test subsets; right, folded cross validation.</figcaption>
</figure>

### Exhaustive tuning

The exhaustive tuning (or exhaustive grid search) is provided by the Scikit learn function <span class='package'>GridSearchCV</span>. The function exhaustively generates candidates from a grid of hyper-parameter values specified with the _param_grid_ parameter. Compared to the randomized grid search, you can specify the search space in more detail, but you need to narrow the search space down as the processes otherwise will take long. If your ranodmized tuning indicates that a hyper-parameter value can be set to a constant value that is not the default value, it is better to define that particular hyper-parameter in the initial model definition (_modD_) and omit it from the tuning search. If it is the default value of the hyper-parameter that can be held constant, all you have to do is to omit it from tuning search.

Import the <span class='package'>GridSearchCV</span> at the beginning of the module.

```
from sklearn.model_selection import GridSearchCV
```
And create the function <span class='pydel'>ExhaustiveTuning</span> under the class <span class='pydef'>RegressionModels</span>.

```
    def ExhaustiveTuning(self, fraction=0.5, n_top=3):
        # run exhaustive search
        for m in self.models:
            #Retrieve the model name and the model itself
            name,mod = m
            print name, self.paramGrid[name]
            search = GridSearchCV(mod, param_grid=self.paramGrid[name])
            X_train, X_test, y_train, y_test = model_selection.train_test_split(self.X, self.y, test_size=(1-fraction))
            search.fit(X_train, y_train)
            self.ReportSearch(search.cv_results_,n_top)
            #Append items from the initial modD dictionary
            best = np.flatnonzero(search.cv_results_['rank_test_score'] == 1)
            tuneModD=search.cv_results_['params'][best[0]]
            #Set the highest ranked hyper-parameter
            for key in self.modD[name]:
                tuneModD[key] = self.modD[name][key]
            regmods.modD[name] = tuneModD
```


When setting the exhaustive tunings below, I have glanced as the top ranked results from the randomized tuning for each regressor. For some regressor models I also chose to set some of the tuned hyper-parameters from the randomized tuning search as initial hyper-parameters and omit them from the exhaustive tuning search.

#### KNeighborsRegressor exhaustive tuning

Add the function for setting the exhaustive search tuning parameters for each model to test. The code also contains the search setting for 'KnnRegr'.

```
    def ExhaustiveTuningParams(self):
        # specify parameters and distributions to sample from
        self.paramGrid = {}
        for m in self.models:
            name,mod = m
            print ('name'), (name), (mod.get_params())
            if name == 'KnnRegr':
                self.paramGrid[name] = [{"n_neighbors": [6,7,8,9,10],
                                   'algorithm': ('ball_tree', 'kd_tree'),
                                   'leaf_size': [15,20,25,30,35]},
                                {"n_neighbors": [6,7,8,9,10],
                                  'algorithm': ('auto','brute')}
                                   ]
```

The hyper-parameter _leaf_size_ in 'KnnRegr' only has relevance when the hyper-parameter _algorithm_ is set either to _ball_tree_ or _kd_tree_. The search is thus divided into two blocks (each defined as a dictionary), the first block for _ball_tree_ and _kd_tree_ also includes _leaf_size_, whereas the second block (for the algorithms _auto_ and _brute_) does not. In the randomized tuning search for 'KnnRegr', the three top ranked results all had the hyper-parameter _p_ (power parameter for the [default] Minkowski metric) value of 1, whereas the default value is 2. Also the hyper-parameter _weights_ have a constant value (_distance_) in the top ranked randomized tunings, and this is also not the default value. When initially formulating the 'KnnRegr' model (in the \_\_main\_\_ section), I thus set the hyper-parameters _p_ to 1, and _weights_ to _'distance'_ and omit them from the exhaustive tuning.
```
regmods.modD['KnnRegr'] = {'weights':'distance','p':1}
```
The results from the exhaustive search with these settings are similar to the results from the randomized search. And the regressors appears to be insensitive to most of the hyper-parameters, as indicated from the five equally ranked parameter settings below.
```
Model with rank: 1
Mean validation score: 0.620 (std: 0.024)
Parameters: {'n_neighbors': 6, 'leaf_size': 15, 'algorithm': 'ball_tree'}

Model with rank: 1
Mean validation score: 0.620 (std: 0.024)
Parameters: {'n_neighbors': 6, 'leaf_size': 25, 'algorithm': 'ball_tree'}

Model with rank: 1
Mean validation score: 0.620 (std: 0.024)
Parameters: {'n_neighbors': 6, 'leaf_size': 15, 'algorithm': 'kd_tree'}

Model with rank: 1
Mean validation score: 0.620 (std: 0.024)
Parameters: {'n_neighbors': 6, 'algorithm': 'auto'}

Model with rank: 1
Mean validation score: 0.620 (std: 0.024)
Parameters: {'n_neighbors': 6, 'algorithm': 'brute'}
```

#### DecisionTreeRegressor exhaustive tuning

The 'DecTreeRegr' regressor does not have any hyper-parameter that is parameterized using a second hyper-parameter, and there is thus only a single search grid block for the exhaustive search.

```
            elif name =='DecTreeRegr':
                self.paramGrid =[{
                              "min_samples_split": [2,3,4,5,6],
                              "min_samples_leaf": [1,2,3,4]}]
```
```
Model with rank: 1
Mean validation score: 0.783 (std: 0.008)
Parameters: {'min_samples_split': 6, 'min_samples_leaf': 1}

Model with rank: 2
Mean validation score: 0.776 (std: 0.047)
Parameters: {'min_samples_split': 3, 'min_samples_leaf': 1}

Model with rank: 3
Mean validation score: 0.754 (std: 0.038)
Parameters: {'min_samples_split': 5, 'min_samples_leaf': 1}
```

#### SVR exhaustive tuning

The SVR regressor can be set with different kernels (hyper-parameter _kernel_), with different additional hyper-parameters used for defining the behaviour of different kernels. The SVR regressor is computationally demanding, and you must be careful when setting the tuning options unless you have a very powerful machine, or lots of time (or a small dataset).
```
            elif name =='SVR':                
                self.paramGrid = [{"kernel": ['linear'],
                              "epsilon": (0.05, 0.1, 0.2),
                              "C": (1, 10, 100)},
                              {"kernel": ['rbf'],
                               'gamma': [0.001, 0.0001],
                              "epsilon": (0.05, 0.1, 0.2),
                              "C": (1, 10, 100)},
                              {"kernel": ['poly'],
                               'gamma': [0.001, 0.0001],
                               'degree':[2,3],
                              "epsilon": (0.05, 0.1, 0.2),
                              "C": (0.5, 1, 5, 10, 100)}]
```

All the highest ranked models have the hyper paramteer _kernel_ set to 'linear', with results insensitive to both the hyper-parameters _C_ and _epsilon_ within the ranges set in the exhaustive tuning search.

```
Model with rank: 1
Mean validation score: 0.604 (std: 0.020)
Parameters: {'epsilon': 0.2, 'C': 1, 'kernel': 'linear'}

Model with rank: 2
Mean validation score: 0.602 (std: 0.021)
Parameters: {'epsilon': 0.2, 'C': 2, 'kernel': 'linear'}

Model with rank: 3
Mean validation score: 0.600 (std: 0.016)
Parameters: {'epsilon': 0.1, 'C': 1, 'kernel': 'linear'}
```

#### RandomForestRegressor exhaustive tuning

The 'RandForRegr' has plenty of hyper-parameters that can be set. I opted for only tuning a few, including _n_estimators_, _min_samples_split_, _min_samples_leaf_ and _bootstrap_.

```
            elif name =='RandForRegr':    
                self.paramGrid[name] = [{
                              "n_estimators": (20,30),
                              "min_samples_split": (2, 3, 4, 5),
                              "min_samples_leaf": (2, 3, 4),
                              "bootstrap": [True,False]}]
```

The results of the random forest regressor varies between different runs. This happens because the initial branching (how data is split in the growing trees) determines later branching and the forest can look vary different in different runs.  

```
Model with rank: 1
Mean validation score: 0.848 (std: 0.071)
Parameters: {'min_samples_split': 2, 'n_estimators': 30, 'bootstrap': True, 'min_samples_leaf': 2}

Model with rank: 2
Mean validation score: 0.848 (std: 0.071)
Parameters: {'min_samples_split': 3, 'n_estimators': 30, 'bootstrap': True, 'min_samples_leaf': 2}

Model with rank: 3
Mean validation score: 0.842 (std: 0.063)
Parameters: {'min_samples_split': 4, 'n_estimators': 20, 'bootstrap': True, 'min_samples_leaf': 2}
```

#### Exhausted tuning results

<figure class="half">
	<a href="{{ site.commonurl }}/images/{{ site.data.images[page.figure5A].source }}"><img src="{{ site.commonurl }}/images/{{ site.data.images[page.figure5A].file }}" alt="image"></a>
  <a href="{{ site.commonurl }}/images/{{ site.data.images[page.figure5B].source }}"><img src="{{ site.commonurl }}/images/{{ site.data.images[page.figure5B].file }}" alt="image"></a>

  <figcaption>Hyper-parameter tuned prediction using k nearest neigbhor regression: left, dataset split into training and test subsets; right, folded cross validation.</figcaption>
</figure>

<figure class="half">
	<a href="{{ site.commonurl }}/images/{{ site.data.images[page.figure6A].source }}"><img src="{{ site.commonurl }}/images/{{ site.data.images[page.figure6A].file }}" alt="image"></a>
  <a href="{{ site.commonurl }}/images/{{ site.data.images[page.figure6B].source }}"><img src="{{ site.commonurl }}/images/{{ site.data.images[page.figure6B].file }}" alt="image"></a>

  <figcaption>Hyper-parameter tuned prediction using decision tree regression: left, dataset split into training and test subsets; right, folded cross validation.</figcaption>
</figure>

<figure class="half">
	<a href="{{ site.commonurl }}/images/{{ site.data.images[page.figure7A].source }}"><img src="{{ site.commonurl }}/images/{{ site.data.images[page.figure7A].file }}" alt="image"></a>
  <a href="{{ site.commonurl }}/images/{{ site.data.images[page.figure7B].source }}"><img src="{{ site.commonurl }}/images/{{ site.data.images[page.figure7B].file }}" alt="image"></a>

  <figcaption>Hyper-parameter tuned prediction using support vector machine regression: left, dataset split into training and test subsets; right, folded cross validation.</figcaption>
</figure>

<figure class="half">
	<a href="{{ site.commonurl }}/images/{{ site.data.images[page.figure8A].source }}"><img src="{{ site.commonurl }}/images/{{ site.data.images[page.figure8A].file }}" alt="image"></a>
  <a href="{{ site.commonurl }}/images/{{ site.data.images[page.figure8B].source }}"><img src="{{ site.commonurl }}/images/{{ site.data.images[page.figure8B].file }}" alt="image"></a>

  <figcaption>Hyper-parameter tuned prediction using random forest regression: left, dataset split into training and test subsets; right, folded cross validation.</figcaption>
</figure>

The complete Python module is availabe on [Karttur's repository on Github](https://github.com/karttur/machinelearning/tree/gh-pages/docs/).

## Resources

[Tuning the hyper-parameters of an estimator](http://scikit-learn.org/stable/modules/grid_search.html), Scikit learn.

[RandomizedSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV), Scikit learn.

[GridSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html), Scikit learn.

[What is the Difference Between a Parameter and a Hyperparameter?](https://machinelearningmastery.com/difference-between-a-parameter-and-a-hyperparameter/) by Jason Brownlee

[Completed python module](https://github.com/karttur/machinelearning/tree/gh-pages/docs/) on GitHub.
