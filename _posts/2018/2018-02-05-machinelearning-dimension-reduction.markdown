---
layout: post
title: Machine learning 5 - Dimension reduction
modified: '2018-02-04 21:29'
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
figure1A: machinelearning5_knnregr_ward-random_tt
figure1B: machinelearning5_dectreeregr_ward-random_tt
figure1C: machinelearning5_svr_ward-random_tt
figure1D: machinelearning5_randforregr_ward-random_tt
figure2A: machinelearning5_knnregr_pca-random_tt
figure2B: machinelearning5_dectreeregr_pca-random_tt
figure2C: machinelearning5_svr_pca-random_tt
figure2D: machinelearning5_randforregr_pca-random_tt
date: '2018-02-05 23:08'
comments: true
share: true
---
<script src="https://karttur.github.io/common/assets/js/karttur/togglediv.js"></script>
**Contents**
	\- [introduction](#introduction)
	\- [Prerequistits](#prerequistits)
	\- [Skeleton](#module-skeleton)
	\- [Methods for dimension reduction](#methods-for-dimension-reduction)
		\- [Agglomeration](#agglomeration)
		\- [Model hyper-parameterization](#model-hyper-parameterization)
		\- [Principal Component Analysis (PCA)](#principal-component-analysis-pca)
	\- [Resources](#resources)

## introduction

In earlier posts in this series on machine learning, you have applied both linear and non-linear regressors for predictive modelling. In the post on [feature selection](../machinelearning-feature-selection/) different methods for discriminating among the independent features aiming at retaining only the most adequate is applied. In this post you will instead rearrange the independent variables and create a novel set of independent variables reducing the dimensionality (number of columns) while retaining more of the variation. The methods presented in this post are considered more powerful than simply discriminating among the independent variables.

You can copy and paste the code in the post, but you might have to edit some parts: indentations does not always line up, quotations marks can be of different kinds and are not always correct when published/printed as html text, and the same also happens with underscores and slashes.

The complete code is also available [here](https://github.com/karttur/machinelearning/tree/gh-pages/docs/).

## Prerequistits

The prerequisites are the same as in the previous posts in this series: a Python environment with numpy, pandas, sklearn (Scikit learn) and matplotlib installed.

## Module Skeleton

The module skeleton code is under the button.

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
matplotlib.use('TkAgg')
from matplotlib import pyplot

class RegressionModels:
    '''Machinelearning using regression models
    '''
    def \_\_init\_\_(self, columns,target):
        '''creates an instance of RegressionModels
        '''
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

## Methods for dimension reduction

This post covers two different methods for reducing the dimensions in the independent variables:

* Feature agglomeration
* Principle Component Analysis (PCA)

Import the required packages from Scikit learn.
```
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import PCA
```

For the feature agglomeration you are going to use a _pipeline_ approach for setting up a selection model, and then sending the model to the grid search module <span class='package'>GridSearchCV</span> that you used in the [previous post](../machinelearning-parameter-tuning/). For that you also need to import the Scikit learn modules for <span class='package'>BayesianRidge</span> (the model to use for agglomeration), <span class='package'>Pipeline</span> and <span class='package'>Memory</span>. And then you also need to import <span class='package'>GridSearchCV</span> and <span class='package'>RandomizedSearchCV</span>.

```
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import Pipeline
from sklearn.externals.joblib import Memory
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
```

### Agglomeration

Agglomeration aims at reducing the dimensionality (number of columns) of the independent (_X_) data by merging features that show similar variation patterns. The clustering function in Scikit learn <span class='package'>FeatureAgglomeration</span> uses the Ward hierarchical cluster analysis, and clusters the original _X_ dimension to _n_clusters_. Add the function <span class='pydef'>WardClustering</span> to the <span class='pydef'>RegressionsModels</span> class.


```
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
```

The function resets the class _X_ (_self.X_) variable, and all subsequent processing (regression modelling) will use the clustered data instead of the original data. Your models will then have fewer independent features to sieve through. The modelling will thus be faster, but with only a small loss in predictive power, and without redundancy among the independent features.

When calling the <span class='pydef'>WardClustering</span> function you have to give the number of clusters you want to merge the original into. To use the function, call it from the \_\_main\_\_ section.

```
    nClusters = 5
    regmods.WardClustering(nClusters)
    #Processes called after the clustering will use the clustered X dataset for modelling
    #Setup regressors to test
    regmods.modD['KnnRegr'] = {}
    regmods.modD['DecTreeRegr'] = {}
    regmods.modD['SVR'] = {}
    regmods.modD['RandForRegr'] = {}

    #Invoke the models
    regmods.ModelSelectSet()

    #set the random tuning parameters
    regmods.RandomTuningParams()
    #Run the tuning
    regmods.RandomTuning()
    #Reset the models with the tuned hyper-parameters
    regmods.ModelSelectSet()
    #Run the models
    regmods.RegrModTrainTest()
    regmods.RegrModKFold()   
```

With the \_\_main\_\_ section set as above the module:
* agglomerates the _X_ data to 5 clusters,
* sets up 4 regressors ('KnnRegr, 'DecTreeRegr', 'SVR' and 'RandForRegr'),
* uses a randomized tuning for setting the model hyper-paramters for each regressor, and
* test the predictive powers of each regressor using both train+test and cross validation.

It will take a while.

#### Tuning the number of clusters

In the section above we set an arbitrary number (5) defining the number of clusters that we wanted our dataset to be merged into. To tune an optimal number of clusters we could (manually) change the parameter _nClusters_ and check the result for each trial. But it would be much better to set up a process using a grid search evaluating different alternative agglomerations. In the [previous post](../machinelearning-paramter-tuning/) you used <span class='package'>GridSearchCV</span> for finding the best hyper-parameters.  <span class='package'>GridSearchCV</span> can also be used for identifying the optimal number of clusters. But you must set it up so that <span class='package'>GridSearchCV</span> has some _criterion_ on which to base the search for the optimal _nClusters_.

What is needed for tuning the optimal number of clusters is an estimator (regressor) that evaluates the effects of different _nClusters_. You thus need a process that:

* iteratively changes _nClusters_,
* agglomerates _X_ into _nClusters_ using <span class='pydef'>WardClustering</span>,
* sends the clustered _X_ dataset to an estimator, and
* evaluates the results from the estimator.

In Scikit learn this can be setup using a pipeline (<span class='package'>Pipeline</span>) and <span class='package'>GridSearchCV</span>. <span class='package'>Pipeline</span> defines the functions to link, and <span class='package'>GridSearchCV</span> defines the cluster sizes to test and iterates the process. The example below uses the the [Bayesian linear regressor](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html) as the estimator, and is taken from a [Scikit learn page on Feature agglomeration](http://scikit-learn.org/stable/auto_examples/cluster/plot_feature_agglomeration_vs_univariate_selection.html).

```
    def TuneWardClustering(self, nClustersL, kfolds=2):
        print 'Cluster agglomereations to test',nClustersL
        cv = KFold(kfolds)  # cross-validation generator for model selection
        ridge = BayesianRidge()
        cachedir = tempfile.mkdtemp()
        mem = Memory(cachedir=cachedir)
        ward = FeatureAgglomeration(n_clusters=6, memory=mem)
        clf = Pipeline([('ward', ward), ('ridge', ridge)])
        # Select the optimal number of parcels with grid search
        clf = GridSearchCV(clf, {'ward__n_clusters': nClustersL}, n_jobs=1, cv=cv)
        clf.fit(self.X, self.y)  # set the best parameters
        print 'initial Clusters',iniClusters
        #report the top three results
        self.ReportSearch(clf.cv_results_,3)
        #rerun with the best cluster agglomeration result
        return (clf.best_params_['ward__n_clusters'])
```

The function <span class='pydef'>TuneWardClustering</span> requires a list (_nClustersL_) containing the sizes of the clusters you want to test (for example, to test clustering the X data to between 4 and 10 cluster, the list would be [4,5,6,7,8,9,10]). You can also set the number of folds (_kfolds_) to use in the cross validation. The function returns a single number, the number of clusters that resulted in the highest score of the _criterion_ used in <span class='package'>GridSearchCV</span>. If you do not set a _criterion_ the inbuilt default will be used.


You then also need to add the reporting function for the results of the pipeline clustering.
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

When running the module you can choose to explore the results, or you can just send the best clustering results (the returned parameter from <span class='pydef'>TuneWardClustering</span>) to the function <span class='pydef'>WardClustering</span>.

To test the agglomeration function for exploring the results of the feature agglomeration, update the \_\_main\_\_ section.

```
    #nClusters = 5
    #Agglomerate the X data
    nClustersL = [4,5,6,7,8,9,10,11]
    nClusters = regmods.TuneWardClustering(nClustersL)
    regmods.WardClustering(nClusters)
    '''
    #Processes called after the clustering will use the clustered X dataset for modelling
    #Setup regressors to test
    regmods.modD['KnnRegr'] = {}
    regmods.modD['DecTreeRegr'] = {}
    regmods.modD['SVR'] = {}
    regmods.modD['RandForRegr'] = {}

    #Invoke the models
    regmods.ModelSelectSet()

    #set the random tuning parameters
    regmods.RandomTuningParams()
    #Run the tuning
    regmods.RandomTuning()
    #Reset the models with the tuned hyper-parameters
    regmods.ModelSelectSet()
    #Run the models
    regmods.RegrModTrainTest()
    regmods.RegrModKFold()
    '''
```
If you kept the suggested parameters, the results should be that the best option is to use 6 clusters.
```
Model with rank: 1
Mean validation score: 0.441 (std: 0.100)
Parameters: {'ward__n_clusters': 6}

Model with rank: 2
Mean validation score: 0.435 (std: 0.029)
Parameters: {'ward__n_clusters': 9}

Model with rank: 3
Mean validation score: 0.415 (std: 0.075)
Parameters: {'ward__n_clusters': 7}
```

To run the models with the suggested number of clusters, just remove the commented section invoking the models (the triple quotations \'\'\'). The module then runs both the training+test model predictions <span class='pydef'>RegrModTrainTest</span>) and the folded cross validation predictions <span class='pydef'>RegrModKFold</span>). All models are tuned before actually running the predictions, hence it will take a while for the model formulations to finish and the first plot to appear.

<figure class="half">
	<a href="{{ site.commonurl }}/images/{{ site.data.images[page.figure1A].source }}"><img src="{{ site.commonurl }}/images/{{ site.data.images[page.figure1A].file }}" alt="image"></a>
  <a href="{{ site.commonurl }}/images/{{ site.data.images[page.figure1B].source }}"><img src="{{ site.commonurl }}/images/{{ site.data.images[page.figure1B].file }}" alt="image"></a>
  <a href="{{ site.commonurl }}/images/{{ site.data.images[page.figure1C].source }}"><img src="{{ site.commonurl }}/images/{{ site.data.images[page.figure1C].file }}" alt="image"></a>
  <a href="{{ site.commonurl }}/images/{{ site.data.images[page.figure1D].source }}"><img src="{{ site.commonurl }}/images/{{ site.data.images[page.figure1D].file }}" alt="image"></a>

  <figcaption>Comparison of predictions from various regressors after feature agglomeration (to 6 clusters) and randomized hyper-parameter tuning.</figcaption>
</figure>

### Principal Component Analysis (PCA)

PCA is a linear transformation that places a set of n-vectors (called eigen-vectors) sequentially oriented orthogonally with regard to previously defined vectors, while seeking an orientation that maximizes the explanation of the remaining variation. The maximum number of vectors that can be constructed equals the number of input features, whereafter all the variation in the original data is explained by the vectors. The information content decreases with each vector, and usually the 3 to 4 first components carry almost all information from the original dataset.

You already imported the <span class='pydef'>PCA</span> function from Scikit learn above, to implement the PCA dimension reduction add the function <span class='pydef'>PCAdecompose</span> to the <span class='pydef'>RegressionModels</span> class.
```
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
```

By default <span class='pydef'>PCAdecompose</span> reduces the input array to three principal components. Alternatively you can either set a threshold for the ratio of the total variation that a component  must explain to be accepted (_minExplainRatio_), or set the number of components to be constructed (_nComps_). For the latter to be used, you must set the former to zero (0). To run your regressors using eigen-vectors from PCA as the independent variables, just replace the agglomeration with PCA in the  \_\_main\_\_ section.
```
    '''
    #Agglomerate the X data
    nClustersL = [4,5,6,7,8,9,10,11]
    nFeatures = regmods.TuneWardClustering(nClustersL)
    regmods.WardClustering(nClusters)
    '''
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
```

<figure class="half">
	<a href="{{ site.commonurl }}/images/{{ site.data.images[page.figure2A].source }}"><img src="{{ site.commonurl }}/images/{{ site.data.images[page.figure2A].file }}" alt="image"></a>
  <a href="{{ site.commonurl }}/images/{{ site.data.images[page.figure2B].source }}"><img src="{{ site.commonurl }}/images/{{ site.data.images[page.figure2B].file }}" alt="image"></a>
  <a href="{{ site.commonurl }}/images/{{ site.data.images[page.figure2C].source }}"><img src="{{ site.commonurl }}/images/{{ site.data.images[page.figure2C].file }}" alt="image"></a>
  <a href="{{ site.commonurl }}/images/{{ site.data.images[page.figure2D].source }}"><img src="{{ site.commonurl }}/images/{{ site.data.images[page.figure2D].file }}" alt="image"></a>

  <figcaption>Comparison of predictions from various regressors after dimension reduction with PCA (3 eigen-vectors used as independent variable) and randomized hyper-parameter tuning.</figcaption>
</figure>

The complete code of the module that you created in this post is available at [GitHub](https://github.com/karttur/machinelearning/tree/gh-pages/docs/).

## Resources

[Unsupervised dimensionality reduction](http://scikit-learn.org/stable/modules/unsupervised_reduction.html), Scikit learn.

[FeatureAgglomeration](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.FeatureAgglomeration.html), Scikit learn.

[Decomposing signals in components](http://scikit-learn.org/stable/modules/decomposition.html), Scikit learn.

[PCA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html), Scikit learn.

[Completed python module](https://github.com/karttur/machinelearning/tree/gh-pages/docs/) on GitHub.
