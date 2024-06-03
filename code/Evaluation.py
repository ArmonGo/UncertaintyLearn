from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy


#---------------------------------------------------------------- precision and recall and performance metrics
def lift_score(y_true, y_prob, baseline, top = 0.1):
    ix = np.argsort(-np.array(y_prob))# decreasing 
    rank_y_test = np.array(y_true)[ix[: int(len(ix) * top)]] # top n% hitting rate
    s = sum(rank_y_test)/len(rank_y_test) / baseline
    return s

def evaluation_calculate(res, y_test, score_choice = 'prob_comb', cal_score = 'pr_auc'):
    score_ls = {"strategies" : [],
                   "model" : [],
                   "alpha": [],
                   "score":[],
                   "category": []
                   }
    mapping_key =  {'base_ci_aware' : "Class Weight",
                    'base_ci_unaware' : "Base",
                    'cp_ci_unaware_abs' : "CP",
                    'cp_ci_unaware_gamma' : "CP",
                    'sampling-ADASYN' :  "Resampling",
                    'sampling-KMeansSMOTE' :  "Resampling",
                    'sampling-SMOTE' :  "Resampling",
                    'sampling-SMOTETomek' :  "Resampling"}


    for k in res.keys():
        if k.split('_')[0] != "cp":
            for m in res[k].keys():
                # precision - recall
                score_ls["strategies"].append(k)
                score_ls["model"].append(m)
                score_ls["alpha"].append(None)
                y_prob = res[k][m]["prob"][:, 1]
                if cal_score == 'pr_auc':
                    pre, re, thresholds = precision_recall_curve(y_test, y_prob)
                    score_ls["score"].append((pre, re))
                else:
                    score_ls["score"].append(lift_score(y_test, y_prob, baseline= y_test.sum()/len(y_test)))
                
                score_ls["category"].append(mapping_key[k])
                
                

        if k.split('_')[0] == "cp":
            for m in res[k].keys():
                for a in res[k][m]["threshold"].keys():
                    # precision - recall
                    score_ls["strategies"].append(k)
                    score_ls["model"].append(m)
                    score_ls["alpha"].append(a)
                   
                    y_prob = res[k][m][score_choice][a]
                    if cal_score == 'pr_auc':
                        pre, re, thresholds = precision_recall_curve(y_test, y_prob)
                        score_ls["score"].append((pre, re))
                    else:
                        score_ls["score"].append(lift_score(y_test, y_prob, baseline= y_test.sum()/len(y_test)))
                   
                    score_ls["category"].append(mapping_key[k])

    return score_ls

def mapping_score_tb(score_ls, cal_score = 'pr_auc'): # only consider the base and the lower bound prob
    score_tb = {"strategies" : [],
              "model" : [],
              "alpha": [],
              "score":[],
              "category": []}
    for i in range(len(score_ls["strategies"])):
        score_tb["strategies"].append(score_ls["strategies"][i])
        score_tb["model"].append(score_ls["model"][i])
        score_tb["alpha"].append(score_ls["alpha"][i])
        score_tb["category"].append(score_ls["category"][i])
        if cal_score == 'pr_auc':
            score_tb["score"].append(auc(score_ls["score"][i][1], 
                                            score_ls["score"][i][0])) # recall precision
        else:
            score_tb["score"].append(score_ls['score'][i])
    score_tb = pd.DataFrame.from_dict(score_tb)
    
    score_tb["type"] = score_tb["strategies"].map({'base_ci_aware' : "",
                                                    'base_ci_unaware' : "",
                                                    'cp_ci_unaware_abs' : "abs",
                                                    'cp_ci_unaware_gamma' : "gamma",
                                                    'sampling-ADASYN' :  "ADASYN",
                                                    'sampling-KMeansSMOTE' :  "KMeansSMOTE",
                                                    'sampling-SMOTE' :  "SMOTE",
                                                    'sampling-SMOTETomek' :  "SMOTETomek"})

    return score_tb


def vis_score_tb(auc_df):
    auc_df_rank = auc_df.sort_values(by = ["model",  "score", "category", "strategies"], ascending = False).groupby(["model",  "category"], as_index = False).head(1)
    auc_df_rank["score"] = auc_df_rank["score"].round(3).astype(str) +  " (" + auc_df["alpha"].round(3).astype(str).replace("nan", "") +'-'+ auc_df["type"] + ')'
    auc_df_rank["score"] = auc_df_rank["score"].str.replace("(-)", "")
    auc_df_param = copy.deepcopy(auc_df_rank)
    auc_df_rank = auc_df_rank.drop('alpha',  axis=1)
    auc_df_rank = auc_df_rank.drop('type',  axis=1)
    auc_df_rank = auc_df_rank.pivot(index="model", columns="category", values="score")
    auc_df_rank = auc_df_rank.reset_index()
    best_alpha_set = dict(zip(map(lambda a, b : (a,b), list(auc_df_param["strategies"]), list(auc_df_param["model"])), list(auc_df_param["alpha"]) ))
    return auc_df_rank, best_alpha_set


def plot_best_precision_recall_curve(precision_recall, best_alpha_set):
    df = pd.DataFrame.from_dict(precision_recall)
    plot_models = list(set(list(df["model"])))
    plot_strategies = list(set(list(df["strategies"])))
    row = 4 # one row
    col =  2 # fix

    fig = plt.figure(figsize=(15, 20))
    fig.subplots_adjust(hspace=0.2, wspace=0.2)

    for i in range(1, len(plot_models)+1): # the position of subgraph begins from 1
        p_m = plot_models[i-1]
        ax = fig.add_subplot(row, col, i)
        for p_s in plot_strategies:
            k = (p_s, p_m)
            if k in best_alpha_set.keys():
                p_a = best_alpha_set[k]
                if not np.isnan(p_a):
                    pair = df[(df["model"] == p_m) & (df["strategies"] == p_s) & (df["alpha"] == p_a)]["score"].item()
                    l = df[(df["model"] == p_m) & (df["strategies"] == p_s) & (df["alpha"] == p_a)]["category"].item()
                    label = l + "-alpha " + str(round(p_a, 3))
                else:
                    pair = df[(df["model"] == p_m) & (df["strategies"] == p_s) &(df["alpha"].isna())]["score"].item()
                    l = df[(df["model"] == p_m) & (df["strategies"] == p_s) &(df["alpha"].isna())]["category"].item()
                    if l == "Resampling":
                        l = l + "-" + p_s.split("-")[1] # sampling policy 
                    label = l 
                
                ax.plot( pair[1],pair[0], label=label) # drop the last node 
                plt.legend()
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision') 
        ax.set_title(p_m) # model name 
    fig.suptitle("Precision and recall curve", fontsize = 40)
    plt.show()



#---------------------------------------------------------------- top n

def rank_two_list(prob, y_test):
    ix = np.argsort(-np.array(prob))# decreasing 
    rank_y_test = np.array(y_test)[ix]
    return rank_y_test


def top_n_error_rate(res, y_test, top_percent = list(np.arange(0.01, 0.31, 0.01).round(2)), score_choice = 'prob_comb', baseline = None):
    top_n_error_tb = {"strategies" : [],
              "model" : [],
              "alpha": [],
              "category" : []
             }
    top_n_error_tb.update({"score_for_rate_" + str(r) : [] for r in top_percent })
    mapping_label = {'base_ci_aware' : "Class Weight",
                    'base_ci_unaware' : "Base",
                    'cp_ci_unaware_abs' : "CP",
                    'cp_ci_unaware_gamma' : "CP",
                    'sampling-ADASYN' :  "Resampling",
                    'sampling-KMeansSMOTE' :  "Resampling",
                    'sampling-SMOTE' :  "Resampling",
                    'sampling-SMOTETomek' :  "Resampling"}


    for k in res.keys():
        if k.split('_')[0] != "cp":
            for m in res[k].keys():
                top_n_error_tb["strategies"].append(k)
                top_n_error_tb["model"].append(m)
                top_n_error_tb["alpha"].append(np.nan)
                top_n_error_tb["category"].append(mapping_label[k])
                y_prob = res[k][m]["prob"][:, 1]
                rank_y_test = rank_two_list( y_prob, y_test)
                
                for p in top_percent:
                    cases = int(len(rank_y_test) * p )
                    if baseline is not None:
                        error_rate = sum(rank_y_test[: cases])/len(rank_y_test[: cases]) / baseline
                    else:
                        error_rate = sum(rank_y_test[: cases])/len(rank_y_test[: cases]) 
                    top_n_error_tb["score_for_rate_" + str(p)].append(error_rate)


        if k.split('_')[0] == "cp":
            for m in res[k].keys():
                for a in res[k][m]["threshold"].keys():
                    a = round(a, 3)
                    top_n_error_tb["strategies"].append(k)
                    top_n_error_tb["model"].append(m)
                    top_n_error_tb["alpha"].append(a)
                    top_n_error_tb["category"].append(mapping_label[k])
                    y_prob = res[k][m][score_choice][a]
                    rank_y_test = rank_two_list(y_prob, y_test)
                    for p in top_percent:
                        cases = int(len(y_test) * p )
                        if baseline is not None:
                            error_rate = sum(rank_y_test[: cases])/len(rank_y_test[: cases]) / baseline
                        else:
                            error_rate = sum(rank_y_test[: cases])/len(rank_y_test[: cases]) 
                        top_n_error_tb["score_for_rate_" + str(p)].append(error_rate)
                       
    return top_n_error_tb


def order_label(or_ls, aim_ls, extra_ls):
    or_ix = list(map(lambda x : aim_ls.index(x.split("-")[0]), or_ls))
    goal_pos = np.argsort(or_ix)
    or_ls = [or_ls[x] for x in goal_pos ]
    extra_ls = [extra_ls[x] for x in goal_pos ]
    return or_ls, extra_ls
     

def plot_best_topn(top_n_error_tb, best_alpha_set = None, strategies = None, save_name = None):
    df = pd.DataFrame.from_dict(top_n_error_tb)
    if strategies is not None:
        df =  df[df["strategies"].isin(strategies)]
    # plot_models = list(set(list(df["model"])))
    plot_models = ["LogisticRegression","DecisionTreeClassifier",
                    "KNeighborsClassifier","RandomForestClassifier",
                    "LGBMClassifier", "XGBClassifier",
                    "RUSBoostClassifier", "BalancedRandomForestClassifier"]

    plot_strategies = list(set(list(df["strategies"])))

    row = 4 # one row
    col =  2 
    fig = plt.figure(figsize=(16, 25))
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    color_set = {"Base": "#26577C",
                "CP": "#CD5C08",
                "Resampling": "#B4B4B3",	
                "Class Weight" : "#26577C"}
    line_set = {"Base": "solid",
                "CP": "solid",
                "Resampling": "dashdot",	
                "Class Weight" : "dashed"}
    m_set = {"LogisticRegression" : "Logistic Regression",
            "DecisionTreeClassifier" : "Decision Tree",
            "KNeighborsClassifier" : "K-NN",
            "RandomForestClassifier" : "Random Forest",
            "LGBMClassifier": "LightGBM", 
            "XGBClassifier" : "XGBoost",
            "RUSBoostClassifier" : "RUSBoost",
            "BalancedRandomForestClassifier" : "Balanced RF"}

    # get the x 
    x = list(filter(lambda x: x.startswith("score_for_rate_"), list(df.columns)))
    x = list(map(lambda x: float(x.split('_')[-1]), x))
        
    for i in range(1, len(plot_models)+1): # the position of subgraph begins from 1
        ax = fig.add_subplot(row, col, i)
       
        ax.set_ylim(0., 2.8) # for d1 is 0.65, the rest is 0.75
        p_m = plot_models[i-1] # take the model name 
        for p_s in plot_strategies:
            k = (p_s, p_m)
            if k in best_alpha_set.keys():
                p_a = best_alpha_set[k]
                if not np.isnan(p_a):
                    y = df[(df["model"] == p_m) & (df["strategies"] == p_s) & (df["alpha"] == p_a)].iloc[0, -len(x):]
                    l = df[(df["model"] == p_m) & (df["strategies"] == p_s) & (df["alpha"] == p_a)]["category"].item()
                    label = l + "-alpha " + str(round(p_a, 2))
                else:
                    y = df[(df["model"] == p_m) & (df["strategies"] == p_s) & (df["alpha"].isna())].iloc[0, -len(x):]
                    l = df[(df["model"] == p_m) & (df["strategies"] == p_s) & (df["alpha"].isna())]["category"].item()
                    if l == "Resampling":
                        label = l + "-" + p_s.split("-")[1]
                    else:
                        label = l 
                # print("l", l)
                ax.plot(x, y, label=label, color = color_set[l], linestyle = line_set[l]) # drop the last node 
                # legends sort
        handles, labels = ax.get_legend_handles_labels()
        aim_ls = ["CP", "Class Weight & CP", "Base", "Class Weight", "Resampling"]
        labels2, handles2 = order_label(labels, aim_ls, handles)
        ax.legend(handles2, labels2, loc = "upper right", fontsize= 13)
        ax.set_xlabel('top n rate', fontsize= 15)
        ax.set_ylabel('lift', fontsize= 15) 
        ax.set_title(m_set[p_m], fontsize= 15) # model name 
    # fig.suptitle("top n error rate", fontsize = 15)
    if save_name is not None:
        plt.savefig(save_name, format="pdf", bbox_inches="tight") 
    # show the plot
    plt.show()


def data_des(X_train, X_test, y_train, y_test):
    #Dataset & \#Features  & \#Churning Rate\footnotemark[1] & \#Instances & \#Split Method\\
    features_num = X_train.shape[1]
    instances_num = len(X_train) +  len(X_test)
    churning_rate = round((sum(y_train) +  sum(y_test))/(len(y_train) +  len(y_test)),3)
    train_r = round(sum(y_train)/len(y_train),3)
    test_r = round(sum(y_test)/len(y_test),3)
    return print("features_num {},  churning_rate {}, train_r {},  test_r {}, instances_num {}".format(features_num, churning_rate,train_r, test_r, instances_num))


def rescadule_df(df):
    df_re = copy.deepcopy(df[[ "model", "Base","CP","Resampling","Class Weight"]])
    df_re['m_order'] = pd.Categorical(df['model'], ["LogisticRegression","DecisionTreeClassifier",
                                                    "KNeighborsClassifier","RandomForestClassifier",
                                                    "LGBMClassifier", "XGBClassifier",
                                                    "RUSBoostClassifier", "BalancedRandomForestClassifier"])

    df_re = df_re.sort_values("m_order") 
    return   df_re     

    