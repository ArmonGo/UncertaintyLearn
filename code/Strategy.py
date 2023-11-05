
# experiment 

from Predictor import  * 
from DataProcess import * 
from Gridsearch import *
import pickle
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

class experiment:
    def __init__(self, params):
        self.base_ci_unaware = params["base_ci_unaware"]
        self.base_ci_aware = params["base_ci_aware"]
        self.samplers = params["samplers"]
        self.alpha = params["alpha"]
        self.test_size = params["test_size"]
        self.scaler = params["scaler"]
        self.best_param = {"base_ci_unaware": {},
                            "base_ci_aware": {} }
        self.cp_score = params["cp_score"]

    def get_data(self, path, f): # data index
        self.datacode = str(path[-6: -4])
        self.retrain = Path("YOUR_SAVE_PATH/best_param_" + self.datacode + '.pkl').is_file()
        if self.retrain: # the text file exist
            # load the train and test data 
            with open ("YOUR_SAVE_PATH/test_df_" + self.datacode + '.pkl', 'rb') as fp:
                dataset = pickle.load(fp)
            # load the best param
            with open ("YOUR_SAVE_PATH/best_param_" + self.datacode + '.pkl', 'rb') as fp:
                self.best_param = pickle.load(fp)
            self.datas = dataset

        else: 
            if self.datacode == "T1": # temporal data
                X_train, X_test, y_train, y_test =  f(path, scaler = self.scaler)

            else:
                df = f(path)
                f = stand_split_df
                X_train, X_test, y_train, y_test = f(df, test_size = self.test_size, 
                                                                scaler = self.scaler)
            self.datas ={}
            self.datas["ori"] = (X_train, y_train)
            self.datas["test"] = (X_test, y_test)
            for s in self.samplers.keys():
                X_res, y_res = sampling_data(X_train, y_train, sampler = self.samplers[s])
                self.datas[s] = (X_res, y_res)
            with open("YOUR_SAVE_PATH/test_df_" + str(path[-6: -4]) + '.pkl', 'wb') as fp:
                pickle.dump(self.datas, fp)
                print("test data has been saved")
        return self.datas

    # different strategies
    def base_estimator(self, param_choice):
        pred_set = {}
        X_train, y_train = self.datas["ori"] 
        X_test, y_test = self.datas["test"] 
        if not self.retrain:
            for m_n in param_choice.keys():
                print("model: ", m_n)
                m, m_params = param_choice[m_n]
                best_m, best_param = cross_val(m(), m_params, X_train, y_train)
                if param_choice == self.base_ci_unaware:
                    self.best_param["base_ci_unaware"][m_n]  = best_param
                elif param_choice == self.base_ci_aware:
                    self.best_param["base_ci_aware"][m_n]  = best_param

                pred_set[m_n] = {'prob': best_m.predict_proba(X_test), 
                                    'class': best_m.predict(X_test),
                                    'train_prob': best_m.predict_proba(X_train), 
                                    'train_class': best_m.predict(X_train)}
        else: 
            for m_n in param_choice.keys():
                print("model: ", m_n)
                best_m, _ = param_choice[m_n]
                if param_choice == self.base_ci_unaware:
                    params = self.best_param["base_ci_unaware"][m_n]
                elif param_choice == self.base_ci_aware:
                    params = self.best_param["base_ci_aware"][m_n]
                best_m = best_m()
                best_m.set_params(**params) # set the best param
                best_m.fit(X_train, y_train)
        # update the prediction     
                pred_set[m_n] = {'prob': best_m.predict_proba(X_test), 
                                'class': best_m.predict(X_test),
                                'train_prob': best_m.predict_proba(X_train), 
                                'train_class': best_m.predict(X_train)}
        return pred_set
    
    def cp(self, param_choice, cp_score):
        pred_set = {}
        X_train, y_train = self.datas["ori"] 
        X_test, y_test = self.datas["test"] 
        for m_n in param_choice.keys():
            print("model: ", m_n)
            m, _ = param_choice[m_n]
            if param_choice == self.base_ci_unaware:
                cp_m = CP(m(), self.best_param["base_ci_unaware"][m_n], self.alpha, cp_score )
            elif param_choice == self.base_ci_aware:
                cp_m = CP(m(), self.best_param["base_ci_aware"][m_n], self.alpha, cp_score)
            cp_m.fit(X_train, y_train)
            pred_set[m_n] = cp_m.predict(X_test)
            train_res=  cp_m.predict(X_train)
            pred_set[m_n]["train_prob"] = train_res["prob"]
            pred_set[m_n]["train_threshold"] = train_res["threshold"]
        return pred_set


    def sampling(self, s_n):
        pred_set = {}
        X_test, y_test = self.datas["test"] 
        if not self.retrain:
            print("samplers: ", s_n)
            X_train, y_train = self.datas[s_n]
            for m_n in self.base_ci_unaware.keys():
                print("model: ", m_n)
                m, m_params = self.base_ci_unaware[m_n]
                best_m, best_param = cross_val(m(), m_params, X_train, y_train)
                self.best_param["base_ci_unaware"][m_n + "_" + s_n]   = best_param
                # update the prediction
                pred_set[m_n] = {'prob': best_m.predict_proba(X_test), 
                                    'class': best_m.predict(X_test), 
                                    'train_prob': best_m.predict_proba(X_train), 
                                    'train_class': best_m.predict(X_train)}
        else:
            print("samplers: ", s_n)
            X_train, y_train = self.datas[s_n]
            for m_n in self.base_ci_unaware.keys():
                print("model: ", m_n)
                best_m, _ = self.base_ci_unaware[m_n]
                params = self.best_param["base_ci_unaware"][m_n + "_" + s_n]
                best_m = best_m()
                best_m.set_params(**params) # set the best param
                best_m.fit(X_train, y_train)
        # update the prediction     
                pred_set[m_n] = {'prob': best_m.predict_proba(X_test), 
                                    'class': best_m.predict(X_test),
                                    'train_prob': best_m.predict_proba(X_train), 
                                    'train_class': best_m.predict(X_train)}
        return pred_set

    
    def run(self, path, f):
        self.get_data( path, f)
        result_set = {}
        # base estimator
        print("base_ci_unaware begins")
        result_set["base_ci_unaware"] =  self.base_estimator( param_choice = self.base_ci_unaware)
        print("base_ci_aware begins")
        result_set["base_ci_aware"] =  self.base_estimator( param_choice = self.base_ci_aware)
        # cp
        for c in self.cp_score:
            print("cp_ci_unaware_"+ c + " begins")
            result_set["cp_ci_unaware_" + c ] =  self.cp( param_choice = self.base_ci_unaware, cp_score=c)
            print("cp_ci_aware_"+ c + " begins")
            result_set["cp_ci_aware_" + c] =  self.cp( param_choice = self.base_ci_aware, cp_score=c)

        # sampling
        print("sampling begins")
        for s_n in self.samplers.keys():
            result_set["sampling" + '-' + s_n] = self.sampling(s_n) 
        return result_set
    
def combine_score(res, y_train):
    for k in res.keys():
        if k.split('_')[0] == "cp":
            for m in res[k].keys():
                res[k][m]["prob_comb"] = {}
                for a in res[k][m]["threshold"].keys():
                    scaler = MinMaxScaler()
                    lgs = LogisticRegression()
                    a = round(a, 2)
                    
                    X = (np.concatenate([(res[k][m]["train_threshold"][a] [:,1] -  res[k][m]["train_threshold"][a] [:,0]),
                                    res[k][m]["train_prob"][a]]).reshape(2, -1)).T
                
                    scaler.fit(X)
                    X = scaler.transform(X)
                    lgs.fit(X, y_train)
                    X_t = (np.concatenate([(res[k][m]["threshold"][a] [:,1] -  res[k][m]["threshold"][a] [:,0]),
                                    res[k][m]["prob"][a]]).reshape(2, -1)).T
                    res[k][m]["prob_comb"][a]= lgs.predict_proba(X_t)[:, 1]
    return res


    





