# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 20:32:32 2022

@author: 81901
"""

from abc import ABC, abstractmethod
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

class BaseModel(ABC):
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.name = type(model).__name__
        self.train_time = None
        self.run_time = None

    def predict(self, X, **kwargs):
        return self.model.predict(X, **kwargs)

    def fit(self, X, Y, **kwargs):
        self.model.fit(X, Y, **kwargs)
        self.save()





class PUAdapter(object):
    def __init__(self, estimator, hold_out_ratio=0.1, precomputed_kernel=False):
        self.estimator = estimator
        self.c = 1.0
        self.hold_out_ratio = hold_out_ratio
        if precomputed_kernel:
            self.fit = self.__fit_precomputed_kernel
        else:
            self.fit = self.__fit_no_precomputed_kernel
        self.estimator_fitted = False

    def __str__(self):
        return 'Estimator:' + str(self.estimator) + '\n' + 'p(s=1|y=1,x) ~= ' + str(self.c) + '\n' + \
            'Fitted: ' + str(self.estimator_fitted)

    def __fit_precomputed_kernel(self, X, y):
        positives = np.where(y == 1.)[0]
        #hold_out_size = np.ceil(len(positives) * self.hold_out_ratio)
        hold_out_size = int(np.ceil(len(positives) * self.hold_out_ratio))
        if len(positives) <= hold_out_size:
            raise('Not enough positive examples to estimate p(s=1|y=1,x). Need at least ' + str(hold_out_size + 1) + '.')
        np.random.shuffle(positives)
        hold_out = positives[:hold_out_size]
        #Hold out test kernel matrix
        X_test_hold_out = X[hold_out]
        keep = list(set(np.arange(len(y))) - set(hold_out))
        X_test_hold_out = X_test_hold_out[:,keep]
        #New training kernel matrix
        X = X[:, keep]
        X = X[keep]
        y = np.delete(y, hold_out)
        self.estimator.fit(X, y)
        hold_out_predictions = self.estimator.predict_proba(X_test_hold_out)
        try:
            hold_out_predictions = hold_out_predictions[:,1]
        except:
            pass
        c = np.mean(hold_out_predictions)
        self.c = c
        self.estimator_fitted = True

    def __fit_no_precomputed_kernel(self, X, y):
        positives = np.where(y == 1.)[0]
        #hold_out_size = np.ceil(len(positives) * self.hold_out_ratio)
        hold_out_size = int(np.ceil(len(positives) * self.hold_out_ratio))
        if len(positives) <= hold_out_size:
            raise('Not enough positive examples to estimate p(s=1|y=1,x). Need at least ' + str(hold_out_size + 1) + '.')
        np.random.shuffle(positives)
        hold_out = positives[:int(hold_out_size)]
        X_hold_out = X[hold_out]
        X = np.delete(X, hold_out,0)
        y = np.delete(y, hold_out)
        self.estimator.fit(X, y)
        hold_out_predictions = self.estimator.predict_proba(X_hold_out)
        try:
            hold_out_predictions = hold_out_predictions[:,1]
        except:
            pass
        c = np.mean(hold_out_predictions)
        self.c = c
        self.estimator_fitted = True
        

    def predict_proba(self, X):
        if not self.estimator_fitted:
            raise Exception('The estimator must be fitted before calling predict_proba(...).')
        probabilistic_predictions = self.estimator.predict_proba(X)
        try:
            probabilistic_predictions = probabilistic_predictions[:,1]
        except:
            pass
        return probabilistic_predictions / self.c

    def predict(self, X, treshold=0.5):
        if not self.estimator_fitted:
            raise Exception('The estimator must be fitted before calling predict(...).')
        return np.array([1. if p > treshold else -1. for p in self.predict_proba(X)])



class PUAdapterWrapper(BaseModel):
    def __init__(self, config, precomputed_kernel=False):
        hparms = {'n_estimators': 10, 'criterion': "entropy", 'bootstrap': True, 'n_jobs': -1}
        estimator = RandomForestClassifier(**hparms)
        model = PUAdapter(estimator, precomputed_kernel=precomputed_kernel)
        #wrapped_pu_estimator = PUAdapterWrapper(pu_adaptor, config)
        super().__init__(model, config)

    def save(self, **kwargs):
        pu_saver = {'estimator': self.model.estimator, 'c': self.model.c}
        with open(self.config.output_dir + "pu_estimator.pickle", 'wb') as pu_estimator_file:
            pickle.dump(pu_saver, pu_estimator_file)
        

    def load(self, **kwargs):
        with open(self.config.output_dir + "pu_estimator.pickle", 'rb') as pu_estimator_file:
            pu_saver = pickle.load(pu_estimator_file)
            estimator = pu_saver['estimator']
            pu_estimator = PUAdapter(estimator)
            pu_estimator.c = pu_saver['c']
            pu_estimator.estimator_fitted = True
            self.model = pu_estimator




