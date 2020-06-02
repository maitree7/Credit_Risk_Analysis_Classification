# Classification & Logistic Regression Modeling - Risky Business

![Credit Risk](Images/credit-risk.jpg)

## Background

Auto loans, mortgages, student loans, debt consolidation ... these are just a few examples of credit and loans that people are seeking online. Peer-to-peer lending services such as LendingClub or Prosper allow investors to loan other people money without the use of a bank. However, investors always want to mitigate risk, so you have been asked by a client to help them use machine learning techniques to predict credit risk.

Here we will build and evaluate various machine-learning models to predict credit risk using LendingClub data. Also, credit risk is an inherently imbalanced classification problem (the number of good loans is much higher than the number of at-risk loans). We will employ different imbalanced-learn and Scikit libraries for training and evaluating models with imbalanced class using the following 2 techniques:


1. [Resampling](#Resampling)
2. [Ensemble Learning](#Ensemble-Learning)

- - -

### Files

[Resampling Starter Notebook](Starter_Code/credit_risk_resampling.ipynb)

[Ensemble Starter Notebook](Starter_Code/credit_risk_ensemble.ipynb)

[Lending Club Loans Data](Instructions/Resources/LoanStats_2019Q1.csv.zip)

- - -

#### Resampling

Use of the [imbalanced learn](https://imbalanced-learn.readthedocs.io) library to resample the LendingClub data and build and evaluate logistic regression classifiers using the resampled data.

1. Oversampling of the data using the `Naive Random Oversampler` and `SMOTE` algorithms.
```
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=1)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
```

```
from imblearn.over_sampling import SMOTE
X_resampled, y_resampled = SMOTE(random_state=1, sampling_strategy=1.0).fit_resample(
    X_train, y_train)
```
2. Undersampling of the data using the `Cluster Centroids` algorithm.
```
from imblearn.under_sampling import ClusterCentroids
undersample = ClusterCentroids(random_state=1, n_jobs=1)
X_resampled, y_resampled = undersample.fit_resample(X_train, y_train)
```
3. Over- and under-sampling using a combination `SMOTEENN` algorithm.
```
from imblearn.combine import SMOTEENN
sm = SMOTEENN(random_state=1)
X_resampled, y_resampled = sm.fit_resample(X_train, y_train)
```

Following steps were performed for each of the algorithm:

1. Train a `logistic regression classifier` from `sklearn.linear_model` using the resampled data.
```
from sklearn.linear_model import LogisticRegression
model_ros = LogisticRegression(solver='liblinear', random_state=1)
model_ros.fit(X_resampled, y_resampled)
```

2. Calculate the `balanced accuracy score` from `sklearn.metrics` 
3. Calculate the `confusin matrix` from `sklearn.metrics`
4. Print the `imbalanced classification report` from `imblearn.metrics`


Models                      |Accuracy Score          |Confusion Matrix                  | Classification Report 
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
Random OverSampling |0.744894078392872 |<img src="Images/ros_cm.PNG" width="200" />|<img src="Images/ros_class_rept.PNG" width="200" />
SMOTE OverSampling| 0.7130672345765914|<img src="Images/smote_cm.PNG" width="200" />|<img src="Images/smote_class_rept.PNG" width="200" />
ClusterCentroids UnderSampling| 0.646955954950032 | <img src="Images/us_cc_cm.PNG" width="200" />|<img src="Images/cc_us_class_rept.PNG" width="200" />
SMOTEENN Combination Sampling |0.6917587455658569 | <img src="Images/sm_cm.PNG" width="200" />|<img src="Images/sm_class_rept.PNG" width="200" />


Use the above to answer the following:

> Which model had the best balanced accuracy score?
`Naive Random OverSampler`
> Which model had the best recall score?
`Naive Random OverSampler`
> Which model had the best geometric mean score?
`Naive Random OverSampler` 

#### Ensemble Learning

Use of the `balanced random forest classifier` and the `easy ensemble AdaBoost classifier` to predict loan risk and evaluate each model.

Following Steps were performed:

1. Train the model using the quarterly data from LendingClub provided in the `Resource` folder.
2. Calculate the balanced accuracy score from `sklearn.metrics`.
3. Print the confusion matrix from `sklearn.metrics`.
4. Generate a classification report using the `imbalanced_classification_report` from imbalanced learn.


5. For the balanced random forest classifier only, print the feature importance sorted in descending order (most important feature to least important) along with the feature score.

Use the above to answer the following:

> Which model had the best balanced accuracy score?
>
> Which model had the best recall score?
>
> Which model had the best geometric mean score?
>
> What are the top three features?

- - -


