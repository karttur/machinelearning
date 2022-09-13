---
layout: post
title: Machine learning 6 - Fit polynomials
modified: '2022-09-11 21:29'
categories: blog
excerpt: Feature aggregation and dimension reduction using Machine Learning
tags:
  - OSSL
  - soil organic carbon
  - prediction
sensors:
  - TRMM
regions:
  - sweden
image: avg-trmm-3b43v7-precip_3B43_trmm_2001-2016_A
date: '2022-09-11 23:08'
comments: true
share: true
---
<script src="https://karttur.github.io/common/assets/js/karttur/togglediv.js"></script>

## introduction

This post is an attempt to test Machine Learning for predicting soil properties from spectra data.

## Prerequistits

The prerequisites are the same as in the previous posts in this series: a Python environment with numpy, pandas, sklearn (Scikit learn) and matplotlib installed.

## Module Skeleton

The module skeleton code will be made avaialble.

<button id= "toggle01btn" onclick="hiddencode('toggle01')">Hide/Show module skeleton</button>
<div id="toggle01" style="display:none">
{% capture text-capture %}
{% raw %}
```
CODE
```
{% endraw %}
{% endcapture %}
{% include widgets/toggle-code.html  toggle-text=text-capture  %}
</div>

The complete code of the module that you created in this post is available at [GitHub](https://github.com/karttur/machinelearning/tree/gh-pages/docs/).

## Material and methods

The data used is from OSSL covering south central Sweden.

## Results

### NIR broadband models

#### Soil Organic Carbon (SOC)

##### 1 degree train/test

<figure class="half">
	<a href="../../images/SOC_OLS-tt_no_no_1_no_no_11_620-1020.png"><img src="../../images/SOC_OLS-tt_no_no_1_no_no_11_620-1020.png"></a>

  <a href="../../images/SOC_theilsen-tt_no_no_1_no_no_11_620-1020.png"><img src="../../images/SOC_theilsen-tt_no_no_1_no_no_11_620-1020.png"></a>

  <a href="../../images/SOC_huber-tt_no_no_1_no_no_11_620-1020.png"><img src="../../images/SOC_huber-tt_no_no_1_no_no_11_620-1020.png"></a>

  <a href="../../images/SOC_knnregr-tt_no_no_1_no_no_11_620-1020.png"><img src="../../images/SOC_knnregr-tt_no_no_1_no_no_11_620-1020.png"></a>

  <a href="../../images/SOC_dectreeregr-tt_no_no_1_no_no_11_620-1020.png"><img src="../../images/SOC_dectreeregr-tt_no_no_1_no_no_11_620-1020.png"></a>

  <a href="../../images/SOC_svr-tt_no_no_1_no_no_11_620-1020.png"><img src="../../images/SOC_svr-tt_no_no_1_no_no_11_620-1020.png"></a>

  <a href="../../images/SOC_randforregr-tt_no_no_1_no_no_11_620-1020.png"><img src="../../images/SOC_randforregr-tt_no_no_1_no_no_11_620-1020.png"></a>


  <figcaption>Comparison of train/test (0.7/0.3) SOC predictions from various regressors using 11 raw bands in the NIR region, 620 - 1020 nm.</figcaption>
</figure>

##### 1 degree kfold (n=10)

<figure class="half">
	<a href="../../images/SOC_OLS-kfold_no_no_1_no_no_11_620-1020.png"><img src="../../images/SOC_OLS-kfold_no_no_1_no_no_11_620-1020.png"></a>

  <a href="../../images/SOC_theilsen-kfold_no_no_1_no_no_11_620-1020.png"><img src="../../images/SOC_theilsen-kfold_no_no_1_no_no_11_620-1020.png"></a>

  <a href="../../images/SOC_huber-kfold_no_no_1_no_no_11_620-1020.png"><img src="../../images/SOC_huber-kfold_no_no_1_no_no_11_620-1020.png"></a>

  <a href="../../images/SOC_knnregr-kfold_no_no_1_no_no_11_620-1020.png"><img src="../../images/SOC_knnregr-kfold_no_no_1_no_no_11_620-1020.png"></a>

  <a href="../../images/SOC_dectreeregr-kfold_no_no_1_no_no_11_620-1020.png"><img src="../../images/SOC_dectreeregr-kfold_no_no_1_no_no_11_620-1020.png"></a>

  <a href="../../images/SOC_svr-kfold_no_no_1_no_no_11_620-1020.png"><img src="../../images/SOC_svr-kfold_no_no_1_no_no_11_620-1020.png"></a>

  <a href="../../images/SOC_randforregr-kfold_no_no_1_no_no_11_620-1020.png"><img src="../../images/SOC_randforregr-kfold_no_no_1_no_no_11_620-1020.png"></a>

  <figcaption>Comparison of kfold (n=10) SOC predictions from various regressors using 11 raw bands in the NIR region, 620 - 1020 nm.</figcaption>
</figure>


##### 2 degree train/test

The 2-degree models, created by using [sklearn.preprocessing.PolynomialFeatures](#) get 77 co-variables upon expansion of the origianl 11 bands. These models are thus massively overparameterised.

<figure class="half">
	<a href="../../images/SOC_OLS-tt_no_no_2_no_no_77_620-1020.png"><img src="../../images/SOC_OLS-tt_no_no_2_no_no_77_620-1020.png"></a>

  <a href="../../images/SOC_theilsen-tt_no_no_2_no_no_77_620-1020.png"><img src="../../images/SOC_theilsen-tt_no_no_2_no_no_77_620-1020.png"></a>

  <a href="../../images/SOC_huber-tt_no_no_2_no_no_77_620-1020.png"><img src="../../images/SOC_huber-tt_no_no_2_no_no_77_620-1020.png"></a>

  <a href="../../images/SOC_knnregr-tt_no_no_2_no_no_77_620-1020.png"><img src="../../images/SOC_knnregr-tt_no_no_2_no_no_77_620-1020.png"></a>

  <a href="../../images/SOC_dectreeregr-tt_no_no_2_no_no_77_620-1020.png"><img src="../../images/SOC_dectreeregr-tt_no_no_2_no_no_77_620-1020.png"></a>

  <a href="../../images/SOC_svr-tt_no_no_2_no_no_77_620-1020.png"><img src="../../images/SOC_svr-tt_no_no_2_no_no_77_620-1020.png"></a>

  <a href="../../images/SOC_randforregr-tt_no_no_2_no_no_77_620-1020.png"><img src="../../images/SOC_randforregr-tt_no_no_2_no_no_77_620-1020.png"></a>


  <figcaption>Comparison of train/test (0.7/0.3) SOC predictions from various regressors using an expanded polynomial of 77 covariates derived from 11 original bands in the NIR region, 620 - 1020 nm.</figcaption>
</figure>

##### 2 degree kfold

The 2-degree models, created by using [sklearn.preprocessing.PolynomialFeatures](#) get 77 co-variables upon expansion of the origianl 11 bands. These models are thus massively overparameterised.

<figure class="half">
	<a href="../../images/SOC_OLS-kfold_no_no_2_no_no_77_620-1020.png"><img src="../../images/SOC_OLS-kfold_no_no_2_no_no_77_620-1020.png"></a>

  <a href="../../images/SOC_theilsen-kfold_no_no_2_no_no_77_620-1020.png"><img src="../../images/SOC_theilsen-kfold_no_no_2_no_no_77_620-1020.png"></a>

  <a href="../../images/SOC_huber-kfold_no_no_2_no_no_77_620-1020.png"><img src="../../images/SOC_huber-kfold_no_no_2_no_no_77_620-1020.png"></a>

  <a href="../../images/SOC_knnregr-kfold_no_no_2_no_no_77_620-1020.png"><img src="../../images/SOC_knnregr-kfold_no_no_2_no_no_77_620-1020.png"></a>

  <a href="../../images/SOC_dectreeregr-kfold_no_no_2_no_no_77_620-1020.png"><img src="../../images/SOC_dectreeregr-kfold_no_no_2_no_no_77_620-1020.png"></a>

  <a href="../../images/SOC_svr-kfold_no_no_2_no_no_77_620-1020.png"><img src="../../images/SOC_svr-kfold_no_no_2_no_no_77_620-1020.png"></a>

  <a href="../../images/SOC_randforregr-kfold_no_no_2_no_no_77_620-1020.png"><img src="../../images/SOC_randforregr-kfold_no_no_2_no_no_77_620-1020.png"></a>

  <figcaption>Comparison of kfold (n=10) SOC predictions from various regressors using an expanded polynomial of 77 covariates derived from 11 original bands in the NIR region, 620 - 1020 nm.</figcaption>
</figure>

##### 1 degree kfold with after feature selection

Using 11 consecutive bands probably causes overfitting as the bands are closely correlated. To reduce the number of covariates (bands) a simple method to use is to only retain those covariates that have a variance above a given threshold. In this example I have uses the variance threshold feature selection method and removed 8 bands and only retained the 3 bands with the highest variance for model development.

<figure class="half">
	<a href="../../images/SOC_OLS-kfold_vt_no_1_no_no_3_620-1020.png"><img src="../../images/SOC_OLS-kfold_vt_no_1_no_no_3_620-1020.png"></a>

  <a href="../../images/SOC_theilsen-kfold_vt_no_1_no_no_3_620-1020.png"><img src="../../images/SOC_theilsen-kfold_vt_no_1_no_no_3_620-1020.png"></a>

  <a href="../../images/SOC_huber-kfold_vt_no_1_no_no_3_620-1020.png"><img src="../../images/SOC_huber-kfold_vt_no_1_no_no_3_620-1020.png"></a>

  <a href="../../images/SOC_knnregr-kfold_vt_no_1_no_no_3_620-1020.png"><img src="../../images/SOC_knnregr-kfold_vt_no_1_no_no_3_620-1020.png"></a>

  <a href="../../images/SOC_dectreeregr-kfold_vt_no_1_no_no_3_620-1020.png"><img src="../../images/SOC_dectreeregr-kfold_vt_no_1_no_no_3_620-1020.png"></a>

  <a href="../../images/SOC_svr-kfold_vt_no_1_no_no_3_620-1020.png"><img src="../../images/SOC_svr-kfold_vt_no_1_no_no_3_620-1020.png"></a>

  <a href="../../images/SOC_randforregr-kfold_vt_no_1_no_no_3_620-1020.png"><img src="../../images/SOC_randforregr-kfold_vt_no_1_no_no_3_620-1020.png"></a>

  <figcaption>Comparison of kfold (n=10) SOC predictions from various regressors using 3 selected raw bands in the NIR region, 620 - 1020 nm.</figcaption>
</figure>

##### 2 degree kfold with after feature selection

Using 11 consecutive bands probably causes overfitting as the bands are closely correlated. To reduce the number of covariates (bands) a simple method to use is to only retain those covariates that have a variance above a given threshold. In this example I have uses the variance threshold feature selection method and removed 8 bands and only retained the 3 bands with the highest variance for model development.

<figure class="half">
	<a href="../../images/SOC_OLS-kfold_vt_no_2_no_no_9_620-1020.png"><img src="../../images/SOC_OLS-kfold_vt_no_2_no_no_9_620-1020.png"></a>

  <a href="../../images/SOC_theilsen-kfold_vt_no_2_no_no_9_620-1020.png"><img src="../../images/SOC_theilsen-kfold_vt_no_2_no_no_9_620-1020.png"></a>

  <a href="../../images/SOC_huber-kfold_vt_no_2_no_no_9_620-1020.png"><img src="../../images/SOC_huber-kfold_vt_no_2_no_no_9_620-1020.png"></a>

  <a href="../../images/SOC_knnregr-kfold_vt_no_2_no_no_9_620-1020.png"><img src="../../images/SOC_knnregr-kfold_vt_no_2_no_no_9_620-1020.png"></a>

  <a href="../../images/SOC_dectreeregr-kfold_vt_no_2_no_no_9_620-1020.png"><img src="../../images/SOC_dectreeregr-kfold_vt_no_2_no_no_9_620-1020.png"></a>

  <a href="../../images/SOC_svr-kfold_vt_no_2_no_no_9_620-1020.png"><img src="../../images/SOC_svr-kfold_vt_no_2_no_no_9_620-1020.png"></a>

  <a href="../../images/SOC_randforregr-kfold_vt_no_2_no_no_9_620-1020.png"><img src="../../images/SOC_randforregr-kfold_vt_no_2_no_no_9_620-1020.png"></a>

  <figcaption>Comparison of kfold (n=10) SOC predictions from various regressors using polynomial expansions of 3 selected raw bands in the NIR region, 620 - 1020 nm.</figcaption>
</figure>

##### 1 degree kfold with after KBest feature selection

Here I have used KBest to retain the 2 bands with highest score. The retained bands (660 and 700 nm) differ from the bands with the highest variance (940, 980, and 1020 nm).

<figure class="half">
	<a href="../../images/SOC_OLS-kfold_kb_no_1_no_no_2_620-1020.png"><img src="../../images/SOC_OLS-kfold_kb_no_1_no_no_2_620-1020.png"></a>

  <a href="../../images/SOC_theilsen-kfold_kb_no_1_no_no_2_620-1020.png"><img src="../../images/SOC_theilsen-kfold_kb_no_1_no_no_2_620-1020.png"></a>

  <a href="../../images/SOC_huber-kfold_kb_no_1_no_no_2_620-1020.png"><img src="../../images/SOC_huber-kfold_kb_no_1_no_no_2_620-1020.png"></a>

  <a href="../../images/SOC_knnregr-kfold_kb_no_1_no_no_2_620-1020.png"><img src="../../images/SOC_knnregr-kfold_kb_no_1_no_no_2_620-1020.png"></a>

  <a href="../../images/SOC_dectreeregr-kfold_kb_no_1_no_no_2_620-1020.png"><img src="../../images/SOC_dectreeregr-kfold_kb_no_1_no_no_2_620-1020.png"></a>

  <a href="../../images/SOC_svr-kfold_kb_no_1_no_no_2_620-1020.png"><img src="../../images/SOC_svr-kfold_kb_no_1_no_no_2_620-1020.png"></a>

  <a href="../../images/SOC_randforregr-kfold_kb_no_1_no_no_2_620-1020.png"><img src="../../images/SOC_randforregr-kfold_kb_no_1_no_no_2_620-1020.png"></a>

  <figcaption>Comparison of kfold (n=10) SOC predictions from various regressors using 2 KBest selected raw bands in the NIR region, 620 - 1020 nm.</figcaption>
</figure>

##### 2 degree kfold with after KBest feature selection


<figure class="half">
	<a href="../../images/SOC_OLS-kfold_kb_no_2_no_no_5_620-1020.png"><img src="../../images/SOC_OLS-kfold_kb_no_2_no_no_5_620-1020.png"></a>

  <a href="../../images/SOC_theilsen-kfold_kb_no_2_no_no_5_620-1020.png"><img src="../../images/SOC_theilsen-kfold_kb_no_2_no_no_5_620-1020.png"></a>

  <a href="../../images/SOC_huber-kfold_kb_no_2_no_no_5_620-1020.png"><img src="../../images/SOC_huber-kfold_kb_no_2_no_no_5_620-1020.png"></a>

  <a href="../../images/SOC_knnregr-kfold_kb_no_2_no_no_5_620-1020.png"><img src="../../images/SOC_knnregr-kfold_kb_no_2_no_no_5_620-1020.png"></a>

  <a href="../../images/SOC_dectreeregr-kfold_kb_no_2_no_no_5_620-1020.png"><img src="../../images/SOC_dectreeregr-kfold_kb_no_2_no_no_5_620-1020.png"></a>

  <a href="../../images/SOC_svr-kfold_kb_no_2_no_no_5_620-1020.png"><img src="../../images/SOC_svr-kfold_kb_no_2_no_no_5_620-1020.png"></a>

  <a href="../../images/SOC_randforregr-kfold_kb_no_2_no_no_5_620-1020.png"><img src="../../images/SOC_randforregr-kfold_kb_no_2_no_no_5_620-1020.png"></a>

  <figcaption>Comparison of kfold (n=10) SOC predictions from various regressors using 2 KBest selected raw bands with polynomial expansion to 5 covariates in the NIR region, 620 - 1020 nm.</figcaption>
</figure>

##### 1 degree kfold after RFE feature selection

<figure class="half">
	<a href="../../images/SOC_OLS-kfold_rfe_no_1_no_no_3_620-1020.png"><img src="../../images/SOC_OLS-kfold_rfe_no_1_no_no_3_620-1020.png"></a>

  <a href="../../images/SOC_theilsen-kfold_rfe_no_1_no_no_3_620-1020.png"><img src="../../images/SOC_theilsen-kfold_rfe_no_1_no_no_3_620-1020.png"></a>

  <a href="../../images/SOC_huber-kfold_rfe_no_1_no_no_3_620-1020.png"><img src="../../images/SOC_huber-kfold_rfe_no_1_no_no_3_620-1020.png"></a>

  <a href="../../images/SOC_dectreeregr-kfold_rfe_no_1_no_no_3_620-1020.png"><img src="../../images/SOC_dectreeregr-kfold_rfe_no_1_no_no_3_620-1020.png"></a>

  <a href="../../images/SOC_randforregr-kfold_rfe_no_1_no_no_3_620-1020.png"><img src="../../images/SOC_randforregr-kfold_rfe_no_1_no_no_3_620-1020.png"></a>

  <figcaption>Comparison of kfold (n=10) SOC predictions from various regressors using 3 RFE selected raw bands in the NIR region, 620 - 1020 nm.</figcaption>
</figure>

##### 1 degree kfold after RFECV feature selection

RFECV inlcudes an internal function for selecting an optimal set of covariates. The selection depends on the model used and thus the nr of feature (covariates) varies.

<figure class="half">
	<a href="../../images/SOC_OLS-kfold_rfecv_no_1_no_no_2_620-1020.png"><img src="../../images/SOC_OLS-kfold_rfecv_no_1_no_no_2_620-1020.png"></a>

  <a href="../../images/SOC_theilsen-kfold_rfecv_no_1_no_no_2_620-1020.png"><img src="../../images/SOC_theilsen-kfold_rfecv_no_1_no_no_2_620-1020.png"></a>

  <a href="../../images/SOC_huber-kfold_rfecv_no_1_no_no_11_620-1020.png"><img src="../../images/SOC_huber-kfold_rfecv_no_1_no_no_11_620-1020.png"></a>

  <a href="../../images/SOC_dectreeregr-kfold_rfecv_no_1_no_no_10_620-1020.png"><img src="../../images/SOC_dectreeregr-kfold_rfecv_no_1_no_no_10_620-1020.png"></a>

  <a href="../../images/SOC_randforregr-kfold_rfecv_no_1_no_no_10_620-1020.png"><img src="../../images/SOC_randforregr-kfold_rfecv_no_1_no_no_10_620-1020.png"></a>

  <figcaption>Comparison of kfold (n=10) SOC predictions from various regressors using RFECV selection on model specifik raw bands in the NIR region, 620 - 1020 nm.</figcaption>
</figure>

## Resources

[Polynomial Regression in Python using scikit-learn](https://data36.com/polynomial-regression-python-scikit-learn/), by Tamas Ujhelyi.
