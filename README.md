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

| Logistic Regression  |  C: [0.01, 0.1, 1, 10, 30,...110], class_weight: [balanced] (for class weights approach only)  |

## Result
You can find the result in the original paper [waiting for a link], besides we provide extra lift curve for top 30% ranking instances of each data set. You can find the complete curve in folder 'lift curve'.

