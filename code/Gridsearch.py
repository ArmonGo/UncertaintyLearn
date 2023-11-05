from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import fbeta_score, make_scorer

def my_custom_loss_func(y_true, y_prob):
    pr, re, _ = precision_recall_curve(y_true, y_prob)
    s = auc(re, pr)
    return s

SCORE = make_scorer(my_custom_loss_func, greater_is_better=True, needs_proba = True)


def cross_val(estimator, grid_params, X_train, y_train, 
                scoring = SCORE,
                refit = True, 
                cv = 5, return_train_score = False):
    grid_model = GridSearchCV(estimator=estimator,
                            param_grid=grid_params,
                            scoring=scoring,
                            cv=cv,
                            refit= refit, 
                            return_train_score = return_train_score)
    grid_model.fit(X_train, y_train)
    return grid_model, grid_model.best_params_

