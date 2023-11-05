from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, clone
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.multiclass import unique_labels
import numpy as np
import warnings
from mapie.regression import MapieRegressor
from sklearn.model_selection import KFold
from mapie.conformity_scores import GammaConformityScore, AbsoluteConformityScore
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier 
from sklearn.utils import class_weight

class ClassifierRegressorCalib(BaseEstimator, RegressorMixin):
    def __init__(self, estimator=None, cp_score=None):
        self.estimator = estimator
        self.cp_score = cp_score

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        X, y = self._validate_data(
            X,
            y,
            accept_sparse="csr",
            dtype=[np.float64, np.float32],
            order="C",
        )
        if self.estimator is None:
            warnings.warn("You did not pass an estimator, falling back to LinearRegression()")
            self.fitted_estimator_ = LinearRegression().fit(X, y)
        else:
            y = (y > .5).astype(int)
            self.classes_ = unique_labels(y)
            self.fitted_estimator_ = clone(self.estimator)
            self.calib_est_ = CalibratedClassifierCV(estimator=self.fitted_estimator_, cv = 5)
            self.calib_est_.fit(X, y)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self)
        if isinstance(self.calib_est_, ClassifierMixin):
            pred_prob = self.calib_est_.predict_proba(X)
            if self.cp_score is None or self.cp_score =="abs": # default abs
                return pred_prob[:, 1]
            elif self.cp_score =="gamma": # 
                return pred_prob[:, 1]+ 0.000001 # make sure there is no 0
            else:
                return NameError
        else:
            return self.calib_est_.predict(X)

class CP: 
    def __init__(self, estimator, params, alpha, cp_score):  # calib or not 
        self.estimator = estimator
        self.params = params
        self.estimator.set_params(**params)
        self.cp_score = cp_score
        self.kf = KFold(n_splits=5, shuffle=True,random_state=np.random.randint(100)) 
        if self.cp_score == "abs":
            self.mapie_r = MapieRegressor(ClassifierRegressorCalib(self.estimator,
                                                                    cp_score= self.cp_score), 
                                                                    method="plus", cv = self.kf,
                                                                    conformity_score= AbsoluteConformityScore())
        elif self.cp_score == "gamma":
            self.mapie_r = MapieRegressor(ClassifierRegressorCalib(self.estimator, 
                                                                    cp_score= self.cp_score), 
                                                                    method="plus", cv = self.kf,
                                                                    conformity_score= GammaConformityScore())
        else:
            return NameError
        self.alpha = alpha
    
    def fit(self, X, y):
        if self.cp_score == "abs":
            y_add = np.array(y)
        elif self.cp_score == "gamma":
            y_add = np.array(y)+ 0.000001 
        self.mapie_r.fit(X, y_add) # make sure it is positive for gamma score

    def predict(self, X):
        y_std_pred_r, y_cp_score_r = self.mapie_r.predict(X, alpha=self.alpha)
        y_cp_pred = {"threshold" : {},
                     "prob" : {}
                     }
        for a in range(len(self.alpha)):
            y_cp_pred["threshold"][self.alpha[a]] = y_cp_score_r[:, :, a] 
            y_cp_pred["prob"][self.alpha[a]] = y_std_pred_r
        return y_cp_pred


class CWXGBoost(BaseEstimator, ClassifierMixin):
    def __init__(self, weight=False, learning_rate=0.1, reg_alpha = 1e-5, reg_lambda=1e-05):
        self.weight = weight 
        self.learning_rate = learning_rate
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.estimator_ = XGBClassifier()
        self.estimator_.set_params(**{"learning_rate": self.learning_rate, 
                                    "reg_alpha": self.reg_alpha, 
                                    "reg_lambda": self.reg_lambda})
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        if self.weight:
            classes_weights = class_weight.compute_sample_weight(
                                        class_weight='balanced',
                                        y=y
                                    )
            self.estimator_.fit(X, y, sample_weight=classes_weights)
        else:
            self.estimator_.fit(X, y)
        
    
    def predict(self, X):
        check_is_fitted(self)
        return self.estimator_.predict(X)
    
    def predict_proba(self, X):
        check_is_fitted(self)
        return self.estimator_.predict_proba(X)
