#!/usr/bin/env python

"""
Ignore that file for now
"""

import pickle

with open('file.pickle', 'r') as f:
    obj = pickle.load(f)

X = obj[0]
y = obj[1] - 1
    
from sklearn import svm

lin = svm.LinearSVC(tol = 1e-16)
lin.fit(X, y)

print lin.predict(X)
