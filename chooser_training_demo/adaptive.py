#!/usr/bin/env python
# -*- coding: utf-8 -*-

## Abstracts:
from sklearn.svm.base import BaseSVC

## Classifiers
from sklearn import svm
from sklearn import linear_model
import warnings

# Utilities:
import numpy as np
from sklearn.base import is_classifier

class AdaptiveC(BaseSVC): # BaseSVC might be a base class, but if something funky starts going on, remove it!!!
    def __init__(self, budgets = None, chooser = None, cls = None):
      """
        cls format: [(cls0, cost0), (cls1, cost1), ...], where cost is 1/accuracy, energy, delay (anything that is 'the smaller - the better')
        """

        self.budgets = budgets
        self.chooser = chooser
        
        self.classifiers = cls

        if self.chooser is None:
            self.chooser = svm.SVC(kernel='linear', C = 1., decision_function_shape='ovr', verbose = True, class_weight = 'balanced')
        if self.classifiers is None:
            self.classifiers = [
                (svm.SVC(kernel='linear', probability=True),1),
                (svm.SVC(kernel='poly', degree=5, probability=True),3),
                (svm.SVC(kernel='rbf', probability=True),10)
            ]

        if not isinstance(self.chooser, (svm.LinearSVC, linear_model.LinearRegression)):
            # raise RuntimeWarning('The change in utilization is only supported in LinearSVC. That might change in the future')
            warnings.warn('The change in utilization is only supported in LinearSVC or LinearRegression. That might change in the future')

        ## Not very efficient, creates copies -- hopefully not a lot of classifiers:
        # self.classifiers = sorted(self.classifiers, key = lambda x: x[1]) # Sort by cost!

        self._chooser_order = len(self.classifiers)
        # TODO: Make sure to check the datatypes for the inputs
        # if not is_classifier(self.chooser):
        #     raise TypeError('Chooser function has to be a Classifier! %s received'%type(self.chooser))
        # if not isinstance(self.classifiers, (tuple, list)):
        #     raise TypeError('List of classifiers has to be of type "tuple" or "list"! %s received'%type(self.classifiers))
        # for cls in self.classifiers:
        #     if not is_classifier(cls):
        #         raise TypeError('Non-classifer found in the array of classifiers: %s!'%(type(cls)))
        # print ("Passed!")

    def fit(self, X, y, cross_val = None):
      ## Step 1: Train all other classifiers
      for cls in self.classifiers:
        cls[0].fit(X, y)

      ## Step 2: Train the chooser
      if cross_val is None:
        cross_val = (X, y)

