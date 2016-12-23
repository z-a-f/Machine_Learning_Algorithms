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
    def __init__(self, budget = None, chooser = None, cls = None):
        """
        cls format: [(cls0, cost0), (cls1, cost1), ...], where cost is 1/accuracy, energy, delay (anything that is 'the smaller - the better')
        """
        
        self.budget = budget
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
        self.classifiers = sorted(self.classifiers, key = lambda x: x[1]) # Sort by cost!

        

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

    def fit(self, X, y, cross_val = None, budget = None):
        ## TODO: Currently, cross_val is of the form (X, y), where cross_val[0] = X,
        ## cross_val[1] = y. This needs to be changed and standardized 
        
        ## Step 1: Fit all the classifiers
        for idx in xrange(self._chooser_order):
            print "Training", idx
            self.classifiers[idx][0].fit(X, y)
        ## Create all the pseudo-labels to fit the chooser functionvi
        if cross_val is None:
            cross_val = (X, y)
        y_hats = [cls[0].predict(cross_val[0]) for cls in self.classifiers]

        ## Start from the most expensive classifier:
        l = np.zeros(len(cross_val[1]))
        for idx in xrange(self._chooser_order-1, -1, -1):
            # l[np.where(cross_val[1] == y_hats[idx])] = idx
            l[np.where(cross_val[1] == y_hats[idx])] = idx
        # print np.unique(l, return_counts = True)
        # import matplotlib.pyplot as plt
        # plt.scatter(cross_val[0][:, 0], cross_val[0][:, 1], c=l/2., cmap=plt.cm.BrBG)
        # plt.show()
        # import pickle
        # with open('file.pickle', 'w') as f:
        #     pickle.dump([cross_val[0], l], f)
        
        self.chooser.fit(cross_val[0], l)
        print self.chooser.classes_
        # print self.chooser.score(cross_val[0], cross_val[1])
        return self

    def _predict_proba(self, X):
        chosen = self.chooser.predict(X)
        # return [self.classifiers[ch][0]._predict_proba(X[ch]) for ch in chosen.astype(int)]
        prel_res = []
        result = [[None]*len(X) for _ in xrange(self._chooser_order)]## Need the length of X
        for idx in xrange(self._chooser_order):
            result[idx] = self.classifiers[idx][0]._predict_proba(X)
        # return np.choose(chosen.astype(int), result)
        res = np.array([ result[int(chs)][idx] for idx, chs in enumerate(chosen) ])
        
        return res
        
    def predict(self, X):
        return np.array(map(np.argmax, self._predict_proba(X)))

    def utilization(self, X):
        chosen = self.chooser.predict(X)
        # print np.unique(chosen)
        res = np.zeros(self._chooser_order)
        for el in chosen:
            res[int(el)] += 1
            
        # print _, repeats
        return res / len(X)

    def bias(self, b):
        ## See  http://scikit-learn.org/stable/modules/svm.html#svc
        pass
    
if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import make_moons, make_circles, make_classification

    # X, y = make_moons(n_samples = 1000, noise = 0.1)
    X, y = make_classification(n_samples = 100000, n_features=2, n_redundant=0, n_informative=2,
                           random_state=0, n_clusters_per_class=1)
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    linearly_separable = (X, y)

    # It is usually a good idea to scale the data for SVM training.
    # We are cheating a bit in this example in scaling all of the data,
    # instead of fitting the transformation on the training set and
    # just applying it on the test set.
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)
    X_xval, X_test, y_xval, y_test = train_test_split(X_test, y_test, test_size = .4)
    
    #print np.shape(X_train), np.shape(y_train)
    #print np.shape(y_train)
    ac = AdaptiveC(chooser = svm.SVC(kernel='linear'))
    # ac.fit(X_train, y_train, cross_val = (X_xval, y_xval))
    ac.fit(X_train, y_train, (X_xval, y_xval))

    # print np.shape(X_test)
    X_print = X_train
    y_print = y_train
    
    # print ac.score(X_print, y_print)
    # print ac.classifiers[0][0].score(X_print, y_print)
    # print ac.classifiers[1][0].score(X_print, y_print)
    # print ac.classifiers[2][0].score(X_print, y_print)
    # print ac.utilization(X_xval)

    # print ac.chooser.intercept_
    # ac.chooser.intercept_ = ac.chooser.intercept_*[0.1, 0.5, -1]
    # print ac.chooser.intercept_

    # print "===After change==="
    # print ac.score(X_print, y_print)
    # print ac.classifiers[0][0].score(X_print, y_print)
    # print ac.classifiers[1][0].score(X_print, y_print)
    # print ac.classifiers[2][0].score(X_print, y_print)
    # print ac.utilization(X_xval)


    import matplotlib.pyplot as plt
    h = 0.02
    clf = ac
    # clf = ac.classifiers[2][0]
    clf = ac.classifiers[1][0]
    titles = ['AC', '0', '1', '2']
    for i, clf in enumerate((ac, ac.classifiers[0][0], ac.classifiers[1][0], ac.classifiers[2][0])):
        # create a mesh to plot in
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        plt.subplot(3, 2, i + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
        
        # Plot also the training points
        plt.scatter(X_print[:, 0], X_print[:, 1], c=y_print, cmap=plt.cm.Paired)
        plt.xlabel('Sepal length')
        plt.ylabel('Sepal width')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title(titles[i])

    clf = ac.chooser
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    plt.subplot(3, 2, i + 2)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    
    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    # plt.xlabel('Sepal length')
    # plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title('choose')
    
    plt.show()
