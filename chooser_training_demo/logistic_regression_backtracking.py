
import numpy as np
import scipy
import matcompat

# if available import pylab (from matlibplot)
try:
    import matplotlib.pylab as plt
except ImportError:
    pass

def logistic_regression_backtracking(xtrain, ytrain, w, max_it, eps):

    # Local Variables: a, dL, dL2weight, grad_dir, it, temp_loss, max_it, eps, ytrain, dL2, b, t, w, crit_fac, xtrain, alpha, backtrack_counter, L
    # Function calls: isnan, repmat, log, max, sum, abs, logistic_regression_backtracking, nargin, length, ones, zeros, exp, unique, pinv, size
    #% xtrain: d x n matrix
    #% ytrain: 1 x n label vector {-1,1}
    #% output: 1 x d classification vector
    if nargin<3.:
        w = np.ones(1., matcompat.size(xtrain, 2.))
    
    
    if nargin<4.:
        max_it = 100.
    
    
    if nargin<5.:
        eps = 1e-6
    
    
    if length(np.unique(ytrain)) == 1.:
        alpha = np.zeros(1., matcompat.size(xtrain, 1.))
        alpha[int(0)-1] = np.unique(ytrain)
        return []
    
    
    alpha = np.zeros(1., matcompat.size(xtrain, 1.))
    for it in np.arange(1., (max_it)+(1.), 1.):
        L = np.sum(np.log((1.+np.exp((-ytrain*np.dot(alpha, xtrain))))))
        #% L(isnan(L))=-ytrain(1,isnan(L)).*(alpha*xtrain(:,isnan(L)));
        dL = np.dot(xtrain, (-w*ytrain/(1.+np.exp((ytrain*np.dot(alpha, xtrain))))).conj().T)
        dL2weight = w/(2.+np.exp((ytrain*np.dot(alpha, xtrain)))+np.exp((-ytrain*np.dot(alpha, xtrain))))
        dL2 = np.dot(xtrain*matcompat.repmat(dL2weight, matcompat.size(xtrain, 1.), 1.), xtrain.conj().T)
        grad_dir = np.dot(-linalg.pinv(dL2), dL)
        #%     grad_dir=-(dL2\dL);
        backtrack_counter = 1.
        t = 1.
        a = .1
        b = .9
        crit_fac = np.dot(np.dot(a, dL.conj().T), grad_dir)
        temp_loss = np.log((1.+np.exp((-ytrain*np.dot(alpha+np.dot(t, grad_dir.conj().T), xtrain)))))
        #% temp_loss(isnan(temp_loss))=-ytrain(1,isnan(temp_loss)).*((alpha+t*grad_dir')*xtrain(:,isnan(temp_loss)));
        while np.sum(temp_loss) > L+np.dot(t, crit_fac):
            if backtrack_counter >= 50.:
                break
            
            
            t = np.dot(b, t)
            temp_loss = np.log((1.+np.exp((-ytrain*np.dot(alpha+np.dot(t, grad_dir.conj().T), xtrain)))))
            temp_loss[int(np.isnan[int(temp_loss)-1])-1] = -ytrain[0,int(np.isnan(temp_loss))-1]*np.dot(alpha+np.dot(t, grad_dir.conj().T), xtrain[:,int(np.isnan(temp_loss))-1])
            backtrack_counter = backtrack_counter+1.
            
        alpha = alpha+np.dot(t, grad_dir).conj().T
        if matcompat.max(np.abs(np.dot(t, grad_dir)))<eps:
            break
        
        
        
    return [alpha]