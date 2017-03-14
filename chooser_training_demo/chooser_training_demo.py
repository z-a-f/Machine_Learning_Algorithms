#!/usr/bin/env python

import scipy.io as sio
import numpy as np

import matplotlib.pyplot as plt

# def repmat(a, m, n):
#  return np.kron(np.ones((m,n)), a)

def repmat(matrixA, rowFinal, colFinal):
  return [matrixA[i] * colFinal for i in range(len(matrixA))] * rowFinal

def logistic_regression_backtracking(xtrain, ytrain, w = None, max_it = None, eps = None):
  if w is None:
    w = np.ones((1, xtrain.shape[1]))
    # w = np.random.random((1, xtrain.shape[1]))
  if max_it is None:
    max_it = 100
  if eps is None:
    eps = 1e-6

  if len(np.unique(ytrain)) == 1:
    alpha = zeros(1, xtrain.shape[0])
    alpha[-1] = np.unique(ytrain)
    return alpha

  alpha = np.zeros((1, xtrain.shape[0]))
  for it in xrange(max_it):
    L = np.log(1.+np.exp(-ytrain*np.dot(alpha,xtrain)))
    # nan = np.isnan(L).any()
    # if nan:
    #   L[nan]=-ytrain[0,nan]*np.dot(alpha,xtrain[:,nan]);
    L = np.sum(L)
    # print xtrain
    dL=np.dot(
      xtrain, 
      (
        -w*ytrain / 
        (1+np.exp(ytrain*np.dot(alpha,xtrain)))
      ).conj().T
    )

    dL2weight=w/(2+np.exp(ytrain*np.dot(alpha,xtrain))+np.exp(-ytrain*np.dot(alpha,xtrain)));
    dL2=np.dot(xtrain*repmat(dL2weight,xtrain.shape[0],1),xtrain.conj().T);
    # print repmat(dL2weight,xtrain.shape[0],1).shape
    grad_dir=np.dot(-np.linalg.pinv(dL2),dL);
    # grad_dir=-(dL2\dL)

    backtrack_counter=0;
    t=1;
    a=.1;
    b=.9;
    crit_fac=np.dot(np.dot(a,dL.conj().T), grad_dir);
    temp_loss=np.log(1+np.exp(-ytrain*np.dot(alpha+t*grad_dir.conj().T,xtrain)))
    # temp_loss[np.isnan(temp_loss)]=-ytrain[0,np.isnan(temp_loss)]*np.dot(alpha+np.dot(t,grad_dir.conj().T),xtrain[:,isnan(temp_loss)])
    # print np.sum(temp_loss), (L+t*crit_fac)
    while np.sum(temp_loss)>(L+t*crit_fac)[0][0] and backtrack_counter<50:
      t=b*t
      temp_loss=np.log(1+np.exp(-ytrain*np.dot(alpha+np.dot(t,grad_dir.conj().T),xtrain)))
      # temp_loss[np.isnan(temp_loss)]=-ytrain[0,np.isnan(temp_loss)]*np.dot(alpha+np.dot(t,grad_dir.conj().T),xtrain[:,isnan(temp_loss)])
      backtrack_counter=backtrack_counter+1
    
    alpha=alpha+(t*grad_dir).conj().T;
    if np.max(np.abs(t*grad_dir))<eps:
        break
  return alpha


pendigits = sio.loadmat('pendigits.mat')

# print pendigits.keys()
ytrain = pendigits['ytrain'].astype(float)
ytest = pendigits['ytest'].astype(float)
xtrain = pendigits['xtrain'].astype(float)
xtest = pendigits['xtest'].astype(float)

###

# print ytrain
ytrain = (-1)**(ytrain<5)
ytest = (-1)**(ytest<5)


d, ntrain = xtrain.shape
_, ntest = xtest.shape


## Train linear classifier
alpha = logistic_regression_backtracking(np.vstack((xtrain, np.ones((1, ntrain)))), ytrain)
alpha_offset = alpha[:,-1]
alpha = alpha[:,:-1]
yhatl_train = np.sign(np.dot(alpha, xtrain)+alpha_offset)
yhatl_test = np.sign(np.dot(alpha, xtest)+alpha_offset)

print 'Linear Classifier:'
print '\tTrain error: ' + '%.2f%%'%(np.mean(yhatl_train != ytrain)*100) 
print '\tTest error: ' + '%.2f%%'%(np.mean(yhatl_test != ytest)*100) 

###

## Polynomial classifier training
poly_degree = 3

# Add one more dimension for the bias
xtrainp = np.ones((d*(poly_degree)+1, ntrain))
xtestp = np.ones((d*(poly_degree)+1, ntest))

for deg in xrange(1, poly_degree+1):
  # print deg
  xtrainp[(deg-1)*d:deg*d,:] = np.power(xtrain.astype(float),deg)
  xtestp[(deg-1)*d:deg*d,:] = xtest**deg

alphap = logistic_regression_backtracking(xtrainp, ytrain, max_it = 1)

yhatp_train = np.sign(np.dot(alphap, xtrainp))
yhatp_test = np.sign(np.dot(alphap, xtestp))

print 'Polynomial-%d Classifier (%dx%d features):'%(poly_degree, xtrainp.shape[0] // xtrain.shape[0], xtrain.shape[0])
print '\tTrain error: ' + '%.2f%%'%(np.mean(yhatp_train != ytrain)*100) 
print '\tTest error: ' + '%.2f%%'%(np.mean(yhatp_test != ytest)*100) 

# xtrain = xtrain.conj().T
# ytrain = ytrain[0]
# xtest = xtest.conj().T
# ytest = ytest[0]

# from sklearn.linear_model import LogisticRegression
# sk_lin = LogisticRegression()
# sk_lin.fit(xtrain.conj().T, ytrain[0])

# print 'SciKit Linear Classifier:'
# print '\tTrain error: ' + '%.2f%%'%(100. - 100*sk_lin.score(xtrain.conj().T, ytrain[0])) 
# print '\tTest error: ' + '%.2f%%'%(100. - 100*sk_lin.score(xtest.conj().T, ytest[0])) 

# from sklearn.preprocessing import PolynomialFeatures

# poly = PolynomialFeatures(degree=poly_degree)

# xtrainp_sk = poly.fit_transform(xtrain.conj().T)
# xtestp_sk = poly.fit_transform(xtest.conj().T)

# sk_poly = LogisticRegression()
# sk_poly.fit(xtrainp_sk, ytrain[0])
# yhatp_train_sk = sk_poly.predict(xtrainp_sk)

# print 'SciKit Polynomial-%d Classifier (full polynomial, %d features):'%(poly_degree, poly.n_output_features_ )
# print '\tTrain error: ' + '%.2f%%'%(100. - 100*sk_poly.score(xtrainp_sk, ytrain[0])) 
# print '\tTest error: ' + '%.2f%%'%(100. - 100*sk_poly.score(xtestp_sk, ytest[0])) 

# poly_int = PolynomialFeatures(degree=poly_degree, interaction_only=True)

# xtrainp_sk_int = poly_int.fit_transform(xtrain.conj().T)
# xtestp_sk_int = poly_int.fit_transform(xtest.conj().T)

# sk_poly_int = LogisticRegression()
# sk_poly_int.fit(xtrainp_sk_int, ytrain)

# print 'SciKit Polynomial-%d Classifier (Interaction only polynomial, %d features):'%(poly_degree, poly_int.n_output_features_ )
# print '\tTrain error: ' + '%.2f%%'%(100. - 100*sk_poly_int.score(xtrainp_sk_int, ytrain)) 
# print '\tTest error: ' + '%.2f%%'%(100. - 100*sk_poly_int.score(xtestp_sk_int, ytest)) 


### chooser function

# Specify the budget model
n_budget_points = 40
low = 0.
high = .8
tradeoffs = np.linspace(low, high, n_budget_points)

# augment the data with budgets
xtrain_aug = np.vstack((
  np.kron(xtrain, np.ones((1, n_budget_points))),
  repmat(tradeoffs, ntrain, 1)
)).astype(float)

# define pseudo_labels for each point
pseudo_labels = -np.ones((1, ntrain))
pseudo_labels[np.logical_and(yhatp_train == ytrain, yhatl_train != ytrain)] = 1
pseudo_labels = np.kron(pseudo_labels, np.ones((1, n_budget_points)))

# importance factors
weights = np.logical_and(yhatl_train != ytrain, yhatp_train == ytrain)
weights = np.kron(weights, np.ones((1, n_budget_points))) + repmat(tradeoffs, ntrain, 1)


# train chooser
chooser = logistic_regression_backtracking(
  np.vstack((xtrain_aug, np.ones((1, xtrain_aug.shape[1])))), 
  pseudo_labels, 
  weights)[0]

chooser_offset = chooser[-1]
chooser = chooser[:-1]

# np.hstack((xtrain_aug, np.ones((1, xtrain_aug.shape[0])))).shape

# Evaluate

test_error = np.ones(n_budget_points)
test_cost= np.zeros(n_budget_points)

for k in xrange(n_budget_points):
  temp_labels = np.sign(np.dot(chooser[:-1], xtest)+np.dot(chooser[-1],tradeoffs[k])+chooser_offset)
  # print np.sum(temp_labels)
  test_cost[k] = np.mean(temp_labels==1)
  yhat_test = np.array(yhatl_test) # Deep copy
  yhat_test.T[temp_labels==1]=yhatp_test.T[temp_labels==1]
  test_error[k]=np.mean(yhat_test!=ytest)
  # print yhat_test.shape, temp_labels.shape
  print '%.2f\t%.2f\t%.2f'%(1.-tradeoffs[k],test_cost[k], test_error[k])

test_error=np.hstack((np.mean(yhatp_test!=ytest), test_error))
test_cost=np.hstack((1, test_cost))
# print np.sum(yhat_test!=ytest)

# plt.plot(1.-tradeoffs,test_error[1:]) # xlim([0 1]);xlabel('Fraction of Test Examples using Poly');ylabel('Test Error');
# plt.xlim([0, 1])
# plt.xlabel('Battery Level')
# plt.ylabel('Test Error')
# plt.title('Penbase (linspace: %.2f - %.2f)'%(1-high, 1-low))
# plt.show()
# print "Test Error:", test_error
# print np.sum(yhatp_test)

###
# Change in battery life
battery = np.hstack((
  [1.]*5,
  np.linspace(1., .4, 10),
  [.4]*2,
  np.linspace(.4, .1, 4),
  [.1]*2,
  np.linspace(.1, .9, 8),
  [.9]*3
))

test_error = np.ones(len(battery))
test_cost= np.zeros(len(battery))

for k in xrange(len(battery)):
  temp_labels = np.sign(np.dot(chooser[:-1], xtest)+np.dot(chooser[-1],battery[k])+chooser_offset)
  # print np.sum(temp_labels)
  test_cost[k] = np.mean(temp_labels==1)
  yhat_test = np.array(yhatl_test) # Deep copy
  yhat_test.T[temp_labels==1]=yhatp_test.T[temp_labels==1]
  test_error[k]=np.mean(yhat_test!=ytest)
  # print yhat_test.shape, temp_labels.shape


fig, ax1 = plt.subplots()
ax1.plot(range(len(battery)),battery) 

ax1.set_xlabel('Time')
ax1.set_ylabel('Battery Level')
ax1.grid()

ax2 = ax1.twinx()
ax2.plot(range(len(battery)),max(test_error) - test_error, 'r', label='Test Error')
ax2.plot(range(len(battery)),max(test_cost)-test_cost, 'g', label='HARD Usage')
ax2.set_ylabel('Ratio')
ax2.legend()


plt.tight_layout()
plt.show()
# print "Test Error:", test_error
# print np.sum(yhatp_test)
