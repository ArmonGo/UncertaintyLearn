# Learning from Uncertainty: Improving Churning Prediction using Conformal Confidence Intervals


This repository contains the implementation for the models included in the experimental comparison as presented in:

Learning from Uncertainty: Improving Churning Prediction using Conformal Confidence Intervals Yameng Guo, Seppe vanden Broucke

## Data set 

For the data sets used in the paper, see

D1: https://huggingface.co/datasets/scikit-learn/churn-prediction

D2: https://www.kaggle.com/datasets/jpacse/datasets-for-churn-telecom

D3 & D4: https://www.kaggle.com/datasets/varshapandey/assignment-data 

## Implementation 
The implementation details are in folder 'code'.

the tuned hyperparameters is shown as follows:

| Estimator |  Hyperparameter Grid |
| --- | --- |

 Logistic Regression  |  C: [0.01, 0.1, 1, 10, 30,...110], class_weight: [balanced] (for class weights approach only)  | 
        
Decision Tree   |  min_samples_split: [2, 3, 5], min_samples_leaf: [3, 5, 10], class_weight: [balanced] (for class weights approach only) | 

K-NN |  n_neighbors : [5, 15,...95],  weights: [uniform, distance]| 

Random Forest   |  min_samples_split: [2, 3, 5], min_samples_leaf: [3, 5, 10], class_weight: [balanced] (for class weights approach only) | 

LightGBM   |  reg_alpha: [0, 0.1, 0.2,...1], reg_lambda: [0, 0.1, 0.2,...1], learning_rate: [0.1, 0.01, 0.005], class_weight: [balanced] (for class weights approach only)  | 

XGBoost    |  reg_alpha: [0, 0.1, 0.2,...1],  reg_lambda: [0, 0.1, 0.2,...1], learning_rate: [0.1, 0.01, 0.005], weight: [True] (for class weights approach only) | 

RUSBoost    |  learning_rate: [0, 0.1,...1.0], sampling_strategy: [all, majority, 0.5, 0.6,...1.0{]}, replacement: [True] | 

Balanced RF  |  min_samples_split: [2, 3, 5], min_samples_leaf: [3, 5, 10], sampling_strategy: [all, majority, 0.5, 0.6,...1.0] replacement: [True] | 


## Result
You can find the result in the original paper [waiting for a link], besides we provide extra lift curve for top 30% ranking instances of each data set. You can find the complete curve in folder 'lift curve'.

