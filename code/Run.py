
from DataProcess import * 
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
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

params = {"base_ci_unaware": {
                        "DecisionTreeClassifier" : (DecisionTreeClassifier, 
                                                   {"min_samples_split" :  [2,3,5],
                                                   "min_samples_leaf" :[3,5,10],
                                                   "class_weight": [None], 
                                                  
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
                                    
                                    }
                                    ),
                          "RandomForestClassifier" : (RandomForestClassifier,
                                    {"min_samples_split" : [2,3,5],
                                    "min_samples_leaf" : [3,5,10],
                                     "class_weight": [None]
                                    
                                    }
                                    ),
                         "KNeighborsClassifier" : (KNeighborsClassifier,
                                {"n_neighbors" : np.arange(5, 105, 10), 
                                 "weights" :["uniform", "distance"]
                                }
                                ),
                        "XGBClassifier" : (CWXGBoost,
                               {"weight" : [False],
                                "learning_rate" : [0.1, 0.01, 0.005],
                                "reg_alpha" : np.arange(0, 1.1, 0.1),
                                "reg_lambda" : np.arange(0, 1.1, 0.1)
                               }
                               )
                       
                        },
                        
        "base_ci_aware": {
                        "DecisionTreeClassifier" : (DecisionTreeClassifier, 
                                                   {"min_samples_split" :  [2,3,5],
                                                   "min_samples_leaf" :[3,5,10],
                                                   "class_weight": ["balanced"]
                                                
                                                   }),

                       "LogisticRegression" : (LogisticRegression, 
                                                {"C" : [ 0.01, 0.1, 1 ] + list(range(10,120,20)),
                                               "class_weight": ["balanced"]
                                              }),
                        "LGBMClassifier" : (LGBMClassifier,
                                     {"reg_alpha" : np.arange(0, 1.1, 0.1),
                                     "reg_lambda" : np.arange(0, 1.1, 0.1),
                                    "learning_rate" : [0.1, 0.01, 0.005],
                                    "verbose": [-100], 
                                     "class_weight": ["balanced"]
                                    
                                    }
                                    ),
                          "RandomForestClassifier" : (RandomForestClassifier,
                                    {"min_samples_split" : [2,3,5],
                                    "min_samples_leaf" : [3,5,10],
                                     "class_weight": ["balanced"]
                                    
                                    }
                                    ),
                        "XGBClassifier" : (CWXGBoost,
                               {"weight" : [True],
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
                                                }) ,
                        "RUSBoostClassifier" : (RUSBoostClassifier, 
                                                          {
                                                          "learning_rate" : np.arange(0.1, 1.1,0.1),
                                                           "sampling_strategy": ["all", "majority", 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                                                          "replacement" :  [True]
                                                         
                                                          })
                        },
       

        "samplers": {"SMOTE": ("SMOTE", SMOTE), 
                     "ADASYN": ("ADASYN", ADASYN), 
                     "SMOTETomek": ("SMOTETomek", SMOTETomek), 
                     "KMeansSMOTE":("KMeansSMOTE", KMeansSMOTE), 
                     },
        "alpha" : list(np.arange(0.05, 0.55, 0.05).round(2)), 
        "test_size" : 0.3, 
        "scaler" : StandardScaler,
        "cp_score": ["abs", "gamma"]}

# your own data path
datapaths =  ['./D1.csv', 
              './D2.csv', 
              './D3.csv', 
              './D4.csv', 
              ]
func =  [load_data_D1,
        load_data_D2, 
        load_data_D3, 
        load_data_D4_temporal
        ]
        
for i in range(len(func)):
    p = datapaths[i]
    f = func[i]
    engine = experiment(params)
    res = engine.run(p, f)
    with open("YOUR_SAVE_PATH/result_dataset_" + str(p[-6: -4]) + '.pkl', 'wb') as fp:
        pickle.dump(res, fp)
        print('data index ' + str(p[-6: -4]) + "has been finished")
    with open("YOUR_SAVE_PATH/best_param_" + str(p[-6: -4]) + '.pkl', 'wb') as fp:
        pickle.dump(engine.best_param, fp)