# -*- coding: utf-8 -*-
"""
Created on Mon May  5 14:30:55 2025

@author: nicol
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
import shap

## In earlier stages of this project a predictive model built over causal
## conclusions was planned, but since these do not match medical knowledge, 
## a regular predictive model will be utilized.

np.random.seed(42)


clinic_data = pd.read_pickle("myfilepath")


        ## Dividing data into train/test
        
x_train, x_test, y_train, y_test = train_test_split(
    clinic_data.drop(columns=['Recurrence'])
    , clinic_data['Recurrence'], test_size=0.3, 
    stratify=clinic_data["Recurrence"], random_state=42)

print("Train/test dimensions:\n", "Train:", x_train.shape[0],
      "Test:", x_test.shape[0]) ## 1295 train to 556 test.

## Here 30% of the dataset will be used to test, and the function is stratified
## to account for the selection bias experienced in the causal section of the
## project.



        ## Cross validation for parameter selection:
    
## The following models will use balanced classes to account for selection bias.
## The parameter selection will be done through a randomized grid search, since
## every model needs several tuned parameters.
## Finally, the measure of interest is recall, as false negatives are very
## significant and dangerous on this situation.
    
## Decision tree:
    
dt_model = DecisionTreeClassifier(class_weight='balanced')
    
dt_param_grid = {
    'max_depth':[5, 10, 15, None],
    'min_samples_split':[2, 5, 10],
    'min_samples_leaf':[1, 2, 4, 8],
    'max_features':['sqrt', 'log2', None]}

dt_strat_k = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

dt_random_search = RandomizedSearchCV(dt_model, dt_param_grid,
                                      n_iter=20, scoring='recall',
                                      cv=dt_strat_k, random_state=42)

dt_random_search.fit(x_train, y_train)

print('Best parameters for decision tree:', dt_random_search.best_params_)
## Best parameters for decision tree: {'min_samples_split': 5, 'min_samples_leaf'
## : 8, 'max_features': 'log2', 'max_depth': 15}
print('Best score for decision tree:', dt_random_search.best_score_) ## 0.55 recall

dt_final = dt_random_search.best_estimator_


## Random forest:
    
rf_model = RandomForestClassifier(class_weight='balanced')

rf_param_grid = {
    "n_estimators": [50, 100, 200, 500],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": ["sqrt", "log2", None],
    "max_samples": [0.5, 0.7, 0.9, None]}

rf_strat_k = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rf_random_search = RandomizedSearchCV(rf_model, rf_param_grid,
                                      n_iter=20, scoring='recall',
                                      cv=rf_strat_k, random_state=42)

rf_random_search.fit(x_train, y_train)

print('Best parameters for random forest:', rf_random_search.best_params_)
## Best parameters for random forest: {'n_estimators': 500, 'min_samples_split'
## : 5, 'min_samples_leaf': 8, 'max_samples': None, 'max_features': 'log2', 
## 'max_depth': 10}
print('Best score for random forest:', rf_random_search.best_score_) ## 0.46 recall

rf_final = rf_random_search.best_estimator_

## XGBoost:
    
pos = x_train.loc[y_train == 1, :].shape[0]
neg = x_train.loc[y_train == 0, :].shape[0]
    
xgb_model = XGBClassifier(scale_pos_weight=neg/pos)

xgb_param_grid = {
    "n_estimators": [50, 100, 200, 500],
    "learning_rate": [0.01, 0.05, 0.1, 0.3],
    "max_depth": [5, 10, 15, None],
    "min_child_weight": [1, 3, 5],
    "subsample": [0.5, 0.7, 0.9, 1],
    "gamma": [0, 0.1, 0.3, 1],
    "scale_pos_weight": [1, 3, 5, 10]}

xgb_strat_k = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

xgb_random_search = RandomizedSearchCV(xgb_model, xgb_param_grid,
                                       n_iter=20, scoring='recall',
                                       cv=xgb_strat_k, random_state=42)

xgb_random_search.fit(x_train, y_train)

print('Best parameters for XGBoost:', xgb_random_search.best_params_)
## Best parameters for XGBoost: {'subsample': 0.5, 'scale_pos_weight': 10, 
## 'n_estimators': 100, 'min_child_weight': 3, 'max_depth': 5, 
## 'learning_rate': 0.05, 'gamma': 0}
print('Best score for XGBoost:', xgb_random_search.best_score_) ## 0.95 recall

xgb_final = xgb_random_search.best_estimator_



## Logistic regression:
    
lr_model = LogisticRegression(solver='liblinear', class_weight='balanced',
                              max_iter = 500)

lr_param_grid = {
    'C':[0.01, 0.1, 1, 10, 100],
    'penalty':['l1', 'l2']}

lr_strat_k = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

lr_random_search = RandomizedSearchCV(lr_model, lr_param_grid,
                                      n_iter=20, scoring='recall',
                                      cv=lr_strat_k, random_state=42)

lr_random_search.fit(x_train, y_train)

print('Best parameters for logistic regression:', lr_random_search.best_params_)
## Best parameters for logistic regression: {'penalty': 'l1', 'C': 1}
print('Best score for logistic regression:', lr_random_search.best_score_) ## 0.55 recall


lr_final = lr_random_search.best_estimator_



        ## Testing with train/test split
        
## Binary predictions:

dt_predict = dt_final.predict(x_test)
rf_predict = rf_final.predict(x_test)
xgb_predict = xgb_final.predict(x_test)
lr_predict = lr_final.predict(x_test)


## dt metrics and separation test:
    
print(confusion_matrix(y_test, dt_predict)) ## 200 TP, 158 FP, 92 FN, 106 TN
print(classification_report(y_test, dt_predict)) ## Recurrence class: 0.54 recall
## Accuracy: 0.55

print('ROC-AUC Score:', roc_auc_score(y_test, dt_predict)) ## 0.54


## rf metrics and separation test:

print(confusion_matrix(y_test, rf_predict)) ## 240 TP, 118 FP, 112 FN, 86 TN
print(classification_report(y_test, rf_predict)) ## Recurrence class: 0.43 recall
## Accuracy: 0.59

print('ROC-AUC Score:', roc_auc_score(y_test, rf_predict)) ## 0.55


## xgb metrics and separation test:

print(confusion_matrix(y_test, xgb_predict)) ## 31 TP, 327 FP, 10 FN, 188 TN
print(classification_report(y_test, xgb_predict)) ## Recurrence class: 0.95 recall
## Accuracy: 0.39. Important note: non-recurrence class: 0.09 recall.

print('ROC-AUC Score:', roc_auc_score(y_test, xgb_predict)) ## 0.51

## lr metrics and separation test:

print(confusion_matrix(y_test, lr_predict)) ## 199 TP, 159 FP, 86 FN, 112 TN
print(classification_report(y_test, lr_predict)) ## Recurrence class: 0.57 recall
## Accuracy: 0.56

print('ROC-AUC Score:', roc_auc_score(y_test, lr_predict)) ## 0.56

## These evaluations show that the simplified recall is misleading in the
## apparently best model. The XGBoost model has a recall of 95% on the recurrence class, 
## but the confusion matrix reveals that it is simply predicting recurrence in the
## vast majority of cases. This makes the model good at predicting when a patient
## will be recurring but condemns healthy patients to more severe treatments,
## making it unreliable in a real medical setting.

## From all of the metrics, the model that best balances the important
## statistics is the logistic regression model. It unites the best relationship
## between recall and accuracy, as well as the highest ROC-AUC score from the 
## 4 models.

## Even though none of the models performed as expected, the logistic regression
## model is chosen as the best option.


## Summary of the models:
    
## Recall metrics may differ, since earlier the measure was the recall for 
## the recurrence class, and the df contains the recall for of all the data.
    
summary = pd.DataFrame({
    'Model':['DT', 'RF', 'XGB', 'LR'],
    'Recall':[0.545, 0.56, 0.52, 0.565],
    'Accuracy':[0.56, 0.59, 0.39, 0.56],
    'Precision':[0.545, 0.56, 0.565, 0.555],
    'ROC-AUC':[0.54, 0.55, 0.51, 0.56],
    'False Positives':[158, 118, 327, 159],
    'False Negatives':[92, 112, 10, 86]})

print(summary)

        ## Re training model:
            
            
## Since 30% of the data was used for testing, the model was not trained on that
## partition. This, unless the model is retrained including that section, means
## that the model is not being optimized, since there is still data it can learn
## from.


new_train_x = clinic_data.drop(columns=['Recurrence'])
new_train_y = clinic_data['Recurrence']

lr_final.fit(new_train_x, new_train_y)



        ## SHAP values and model decision explanation (logistic regression):
    
lr_explainer = shap.Explainer(lr_final, new_train_x)

lr_shap = lr_explainer(new_train_x)

shap.summary_plot(lr_shap, new_train_x)

## Important features: Age at Diagnosis, ER Status and positive lymph nodes.

shap.plots.waterfall(lr_shap[0])
## This plot shows that for a single specific patient, the age at diagnosis is
## what pushes the model the most to conclude on recurrence. Opposite to general
## expectations, tumor size has an inverse relationship with recurrence for this
## predictive model.

## Saving model for use:
    
joblib.dump(lr_final, 'LR model')



## From this file it can be observed that none of the models reunite all of the
## requisites to be useful in a real setting, but they are still optimized
## to best capabilities that the data allowed. The chosen model is availiable
## for use through its joblib file.


