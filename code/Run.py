
from DataProcess import * 
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from imblearn.ensemble import BalancedRandomForestClassifier, RUSBoostClassifier
import pickle
from imblearn.over_sampling import SMOTE, ADASYN, KMeansSMOTE
from imblearn.combine import SMOTETomek
from sklearn.preprocessing import StandardScaler
from Strategy import *
import warnings
warnings.filterwarnings('ignore')
from Predictor import CWXGBoost

# set the seed
params = {"base_ci_unaware": {
                        "DecisionTreeClassifier" : (DecisionTreeClassifier, 
                                                   {"min_samples_split" :  [2,3,5],
                                                   "min_samples_leaf" :[3,5,10],
                                                   "class_weight": [None], 
                                                   # "random_state" : [1996] # set the seed for the estimator
                                                   }),

                       "LogisticRegression" : (LogisticRegression, 
                                                {"C" : [ 0.01, 0.1, 1 ] + list(range(10,120,20)), 
                                               "class_weight": [None]
                                              }),
                        "LGBMClassifier" : (LGBMClassifier,
                                     {"reg_alpha" : np.arange(0, 1.1, 0.1),
                                     "reg_lambda" : np.arange(0, 1.1, 0.1),
                                    "learning_rate" : [0.1, 0.01, 0.005],
                                    "verbose": [-100], 
                                     "class_weight": [None]
                                    #"random_state": [1996]
                                    }
                                    ),
                          "RandomForestClassifier" : (RandomForestClassifier,
                                    {"min_samples_split" : [2,3,5],
                                    "min_samples_leaf" : [3,5,10],
                                     "class_weight": [None]
                                    #"random_state": [1996]
                                    }
                                    ),
                         "KNeighborsClassifier" : (KNeighborsClassifier,
                                {"n_neighbors" : np.arange(5, 105, 10),  # all default setting
                                 "weights" :["uniform", "distance"]
                                }
                                ),
                        "XGBClassifier" : (CWXGBoost,
                               {"weight" : [False],
                                "learning_rate" : [0.1, 0.01, 0.005],
                                "reg_alpha" : np.arange(0, 1.1, 0.1),
                                "reg_lambda" : np.arange(0, 1.1, 0.1)
                               }
                               ),

                        "BalancedRandomForestClassifier" :(BalancedRandomForestClassifier,
                                                          {"min_samples_split" : [2,3,5],
                                                 "min_samples_leaf" : [3,5,10],
                                                  "sampling_strategy": ["all", "majority", 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                                                  "replacement" :  [True],
                                                  "class_weight" : [None]
                                                    # "random_state": [1996]
                                                }) , 
                        "RUSBoostClassifier" : (RUSBoostClassifier, 
                                                          {
                                                          "learning_rate" : np.arange(0.1, 1.1,0.1),
                                                           "sampling_strategy": ["all", "majority", 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                                                          "replacement" :  [True]
                                                          # "random_state": [1996]
                                                          })
                       
                        },
                        
        "base_ci_aware": {
                        "DecisionTreeClassifier" : (DecisionTreeClassifier, 
                                                   {"min_samples_split" :  [2,3,5],
                                                   "min_samples_leaf" :[3,5,10],
                                                   "class_weight": ["balanced"], 
                                                   # "random_state" : [1996] # set the seed for the estimator
                                                   }),

                       "LogisticRegression" : (LogisticRegression, 
                                                {"C" : [ 0.01, 0.1, 1 ] + list(range(10,120,20)), # if c is too small, we have the error 
                                               "class_weight": ["balanced"]
                                              }),
                        "LGBMClassifier" : (LGBMClassifier,
                                     {"reg_alpha" : np.arange(0, 1.1, 0.1),
                                     "reg_lambda" : np.arange(0, 1.1, 0.1),
                                    "learning_rate" : [0.1, 0.01, 0.005],
                                    "verbose": [-100], 
                                     "class_weight": ["balanced"]
                                    #"random_state": [1996]
                                    }
                                    ),
                          "RandomForestClassifier" : (RandomForestClassifier,
                                    {"min_samples_split" : [2,3,5],
                                    "min_samples_leaf" : [3,5,10],
                                     "class_weight": ["balanced"]
                                    #"random_state": [1996]
                                    }
                                    ),
                        "XGBClassifier" : (CWXGBoost,
                               {"weight" : [True],
                                "learning_rate" : [0.1, 0.01, 0.005],
                                "reg_alpha" : np.arange(0, 1.1, 0.1),
                                "reg_lambda" : np.arange(0, 1.1, 0.1)
                               }
                               )
                        
                        },
       

        "samplers": {"SMOTE": ("SMOTE", SMOTE), 
                     "ADASYN": ("ADASYN", ADASYN), 
                     "SMOTETomek": ("SMOTETomek", SMOTETomek), 
                     "KMeansSMOTE":("KMeansSMOTE", KMeansSMOTE), 
                     },
        "alpha" : [0.01, 0.02, 0.03, 0.04] + list(np.arange(0.05, 0.55, 0.05).round(2)), 
        "test_size" : 0.3, 
        "scaler" : StandardScaler,
        "cp_score": ["abs", "gamma"]}

datapaths =  ['YOUR_DATA_PATH/D1.csv',  
              'YOUR_DATA_PATH/D2.csv',  
              'YOUR_DATA_PATH/D3.csv',  
              'YOUR_DATA_PATH/T1.csv',  
              ]
func =  [load_data_D1,
        load_data_D2, 
        load_data_D3, 
        load_data_T1_temporal
        ]
        
for i in range(len(func)):
    p = datapaths[i]
    f = func[i]
    engine = experiment(params)
    res = engine.run(p, f)
    with open("YOUR_RESULT_PATH/result_dataset_" + str(p[-6: -4]) + '.pkl', 'wb') as fp:
        pickle.dump(res, fp)
        print('data index ' + str(p[-6: -4]) + "has been finished")
    with open("YOUR_RESULT_PATH/best_param_" + str(p[-6: -4]) + '.pkl', 'wb') as fp:
        pickle.dump(engine.best_param, fp)