import pdb
import requests
import warnings
import numpy as np
from sklearn.base import clone
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split, cross_val_score
from sklearn.ensemble.partial_dependence import partial_dependence
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.datasets import make_classification
from time import time
from scipy.stats import randint as sp_randint




class Classifiers(object):


    def __init__(self, model, param_dist):
        self.param_dist = param_dist
        self.model = model 

    def best_parameters(self, n_iter_search, X_train, y_train):
        n_iter_search = 20
        random_search = RandomizedSearchCV(self.model(), 
                                        param_distributions=self.param_dist,
                                        n_iter=n_iter_search, 
                                        cv=5, 
                                        iid=False)
        start = time()
        random_search.fit(X_train, y_train)
        print("RandomizedSearchCV took %.2f seconds for %d candidates"
            " parameter settings." % ((time() - start), n_iter_search))
        self.model = self.model(**random_search.best_params_)
        
        

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        self.y_pred = self.model.predict(X)
        
        
    def feature_importance(self):
        if self.model.__class__.__name__ == 'AdaBoostClassifier' or self.model.__class__.__name__ == 'GradientBoostingClassifier' or self.model.__class__.__name__ == 'RandomForestClassifier':
            self.feature_importances_ = self.model.feature_importances_

        if self.model.__class__.__name__ == 'LogisticRegression':
            self.feature_importances_ = self.model.coef_

    def cross_val(self, X_train, y_train, nfolds):
        ''' Takes an instantiated model (estimator) and returns the average
            mean square error (mse) and coefficient of determination (r2) from
            kfold cross-validation.
            Parameters: estimator: model object
                        X_train: 2d numpy array
                        y_train: 1d numpy array
                        nfolds: the number of folds in the kfold cross-validation
            Returns:  mse: average mean_square_error of model over number of folds
                    r2: average coefficient of determination over number of folds
        
            There are many possible values for scoring parameter in cross_val_score.
            http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
            kfold is easily parallelizable, so set n_jobs = -1 in cross_val_score
        '''
        mse = cross_val_score(self.model, X_train, y_train, 
                            scoring='neg_mean_squared_error',
                            cv=nfolds, n_jobs=-1) * -1
        # mse multiplied by -1 to make positive
        r2 = cross_val_score(self.model, X_train, y_train, 
                            scoring='r2', cv=nfolds, n_jobs=-1)
        mean_mse = mse.mean()
        mean_r2 = r2.mean()
        name = self.model.__class__.__name__
        print("{0:<25s} Train CV | MSE: {1:0.3f} | R2: {2:0.3f}".format(name,
                                                            mean_mse, mean_r2))
        return mean_mse, mean_r2


    def misclassification_rate(self, y, y_pred):
        '''
        INPUT:
            - y: numpy array, true labels
            - y_pred: numpy array, predicted labels
        '''
        return float((y_pred != y).sum())/len(y)

    def stage_score_plot(self, X_train, X_test, y_train, y_test):
        '''
            Parameters: estimator: GradientBoostingClassifier or 
                                AdaBoostClassifier
                        X_train: 2d numpy array
                        y_train: 1d numpy array
                        X_test: 2d numpy array
                        y_test: 1d numpy array
            Returns: A plot of the number of iterations vs the misclassification rate 
            for the model for both the training set and test set.
        '''

        name = self.model.__class__.__name__.replace('Classifier', '')
        label_str = name
        if "Gradient" in name: 
            md = self.model.max_depth
            label_str += ", max depth: {0}".format(md)
        
        # initialize 
        test_scores = np.zeros((self.model.n_estimators,), dtype=np.float64)
        # Get test score from each boost
        for i, y_test_pred in enumerate(self.model.staged_predict(X_test)):
            test_scores[i] = self.misclassification_rate(y_test, y_test_pred)
        plt.plot(test_scores, alpha=.5, label=label_str, ls = '-', marker = 'o', ms = 3)
        plt.ylabel('Misclassification rate', fontsize=14)
        plt.xlabel('Iterations', fontsize=14)
        plt.show()





if __name__ == '__main__':
    X, y = make_classification(n_samples=1000, n_features=4,
                            n_informative=2, n_redundant=0,
                            random_state=0, shuffle=False)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # params = {'n_estimators': 100, 'random_state': 2}
    # ada = Classifiers(AdaBoostClassifier, params)
    # ada.fit(X_train, y_train)
    # ada.predict(X_test)
    # ada.feature_importance()
    # ada.cross_val(X_train, y_train, 4)


    param_dist = {"n_estimators": [5, 10, 300, 100, 500, 1000],
              "max_depth": [3, 2, None],
              "max_leaf_nodes": [3, 4],
              "min_samples_split": sp_randint(2, 11),
              "random_state": [1, 2, 3]}
    # params = {'n_estimators': 1000, 'max_leaf_nodes': 4, 'max_depth': None, 'random_state': 2, 'min_samples_split': 5}
    gbc = Classifiers(GradientBoostingClassifier, param_dist)
    gbc.best_parameters(20, X_train, y_train)
    gbc.fit(X_train, y_train)
    gbc.predict(X_test)
    gbc.feature_importance()
    gbc.stage_score_plot(X_train, X_test, y_train, y_test)
    # gbc.cross_val(X_train, y_train, 4)

    # params = {'n_estimators':100, 'max_depth':2, 'random_state':0}
    # param_dist = {"n_estimators": [5, 10, 20, 40, 60],
    #           "max_depth": [3, None],
    #           "max_features": sp_randint(1, 4),
    #           "min_samples_split": sp_randint(2, 11),
    #           "bootstrap": [True, False],
    #           "criterion": ["gini", "entropy"]}
    # rf = Classifiers(RandomForestClassifier, param_dist)
    # rf.best_parameters(20, X_train, y_train)
    # rf.fit(X_train, y_train)
    # rf.predict(X_test)
    # rf.feature_importance()
    # rf.cross_val(X_train, y_train, 4)


    

    # n_iter_search = 20
    # random_search = RandomizedSearchCV(RandomForestClassifier(), param_distributions=param_dist,
    #                                n_iter=n_iter_search, cv=5, iid=False)

    # start = time()
    # random_search.fit(X_train, y_train)
    # print("RandomizedSearchCV took %.2f seconds for %d candidates"
    #     " parameter settings." % ((time() - start), n_iter_search))
    # print(RandomForestClassifier(**random_search.best_params_))
    # params = {}
    # clf = Classifiers(LogisticRegression, params)
    # clf.fit(X_train, y_train)
    # clf.predict(X_test)
    # clf.feature_importance()
    # clf.cross_val(X_train, y_train, 4)


# >>> X, y = make_classification(n_samples=1000, n_features=4,
# ...                            n_informative=2, n_redundant=0,
# ...                            random_state=0, shuffle=False)
# >>> clf = AdaBoostClassifier(n_estimators=100, random_state=0)
# >>> clf.fit(X, y)  
# AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
#         learning_rate=1.0, n_estimators=100, random_state=0)
# >>> clf.feature_importances_  
# array([0.28..., 0.42..., 0.14..., 0.16...])
# >>> clf.predict([[0, 0, 0, 0]])
# array([1])
# >>> clf.score(X, y)  
# 0.983...