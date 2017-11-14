#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 17:53:06 2017

@author: Onat1
"""

from matplotlib.pyplot import figure, plot, subplot, xlabel, ylabel, show, clim, savefig, suptitle
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import cross_validation
from toolbox_02450 import feature_selector_lr, bmplot
import numpy as np

# Load data from matlab file
mat_data = loadmat('murder.mat')
X = mat_data['X']
y = mat_data['y'].squeeze()
attributeNames = [name[0] for name in mat_data['attributeNames'][0]]
N, M = X.shape

# Normalize X
X_normalize = (X - np.ones((N,1))*X.mean(0))
for col_id in range(M):
    X_normalize[:,col_id] = X_normalize[:,col_id]/X[:,col_id].std(ddof=1)

X = X_normalize

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
CV = cross_validation.KFold(N,K,shuffle=True)

# Initialize variables
Features = np.zeros((M,K))
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_fs = np.empty((K,1))
Error_test_fs = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))

k=0
for train_index, test_index in CV:
    
    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]
    internal_cross_validation = 10
    
    # Compute squared error without using the input data at all
    Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum()/y_train.shape[0]
    Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum()/y_test.shape[0]
    
    # Compute squared error with all features selected (no feature selection)
    m = lm.LinearRegression(fit_intercept=True).fit(X_train, y_train)
    Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
    Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]
    
    # Compute squared error with feature subset selection
    #textout = 'verbose';
    textout = '';
    selected_features, features_record, loss_record = feature_selector_lr(X_train, y_train, internal_cross_validation,display=textout)
    
    Features[selected_features,k]=1
    if len(selected_features) is 0:
        print('No features were selected, i.e. the data (X) in the fold cannot describe the outcomes (y).' )
    else:
        m = lm.LinearRegression(fit_intercept=True).fit(X_train[:,selected_features], y_train)
        Error_train_fs[k] = np.square(y_train-m.predict(X_train[:,selected_features])).sum()/y_train.shape[0]
        Error_test_fs[k] = np.square(y_test-m.predict(X_test[:,selected_features])).sum()/y_test.shape[0]
        
        figure(k)
        suptitle("Cross-validation fold #{}".format(k+1), fontsize=12, fontweight='bold')
        subplot(1,2,1)
        plot(range(1,len(loss_record)), loss_record[1:])
        xlabel('Iteration')
        ylabel('Squared error (crossvalidation)')    
        
        subplot(1,3,3)
        bmplot(attributeNames, range(1,features_record.shape[1]), -features_record[:,1:])
        clim(-1.5,0)
        xlabel('Iteration')
        savefig("project-2-forward-selection-squared-error-{}-cv-fold.png".format(k+1), bbox_inches='tight')
    
    print('Cross validation fold {0}/{1}'.format(k+1,K))
    print('Train indices: {0}'.format(train_index))
    print('Test indices: {0}'.format(test_index))
    print('Features no: {0}'.format(selected_features.size))
    print("Weights: {}\n".format(m.coef_))

    k+=1
    
    
# Display results
print('\n')
print('Linear regression without feature selection:\n')
print('- Training error: {0}'.format(Error_train.mean()))
print('- Test error:     {0}'.format(Error_test.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
print('Linear regression with feature selection:\n')
print('- Training error: {0}'.format(Error_train_fs.mean()))
print('- Test error:     {0}'.format(Error_test_fs.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_fs.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}'.format((Error_test_nofeatures.sum()-Error_test_fs.sum())/Error_test_nofeatures.sum()))

figure(k, figsize=(10, 10))
subplot(1,3,2)
bmplot(attributeNames, range(1,Features.shape[1]+1), -Features)
clim(-1.5,0)
xlabel('Crossvalidation fold')
ylabel('Attribute')
savefig("project-2-selected-features-for-each-cv-fold.png", bbox_inches='tight')
show()

# Inspect selected feature coefficients effect on the entire dataset and
# plot the fitted model residual error as function of each attribute to
# inspect for systematic structure in the residual

#f=2 # cross-validation fold to inspect
for f in range(0, K):
        
    ff=Features[:,f].nonzero()[0]
    if len(ff) is 0:
        print('\nNo features were selected, i.e. the data (X) in the fold cannot describe the outcomes (y).' )
    else:
        m = lm.LinearRegression(fit_intercept=True).fit(X[:,ff], y)
        
        y_est= m.predict(X[:,ff])
        residual=y-y_est
        
        
        figure(k+1,figsize=(10, 10))
        suptitle('Residual error vs. Attributes for features selected in cross-validation fold {0}'.format(f+1), fontsize=12, fontweight='bold')
        for i in range(0,len(ff)):
           subplot(2,np.ceil(len(ff)/2.0),i+1)
           plot(X[:,ff[i]],residual,'.')
           xlabel(attributeNames[ff[i]])
           ylabel('residual error')
        
        savefig("project-2-residual-error-vs-attribute-{}-cv-fold.png".format(f+1), bbox_inches='tight')
        show()  

#%% ANN
import neurolab as nl
from sklearn import model_selection
from matplotlib.pyplot import (figure, plot, subplot, title, xlabel, ylabel, 
                               hold, contour, contourf, cm, colorbar, show,
                               legend)


X = mat_data['X']
y = mat_data['y']
attributeNames = [name[0] for name in mat_data['attributeNames'].squeeze()]
N, M = X.shape


# Parameters for neural network classifier
n_hidden_units = 10      # number of hidden units
n_train = 2             # number of networks trained in each k-fold

# These parameters are usually adjusted to: (1) data specifics, (2) computational constraints
learning_goal = 2.0     # stop criterion 1 (train mse to be reached)
max_epochs = 200        # stop criterion 2 (max epochs in training)

# K-fold CrossValidation (4 folds here to speed up this example)
K = 4
CV = model_selection.KFold(K,shuffle=True)

# Variable for classification error
errors = np.zeros(K)
error_hist = np.zeros((max_epochs,K))
bestnet = list()
k=0
for train_index, test_index in CV.split(X,y):
    print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))    
    
    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index,:]
    X_test = X[test_index,:]
    y_test = y[test_index,:]
    
    best_train_error = 1e100
    for i in range(n_train):
        # Create randomly initialized network with 2 layers
        ann = nl.net.newff([[0, 1], [0, 1]], [n_hidden_units, 1], [nl.trans.TanSig(),nl.trans.PureLin()])
        # train network
        train_error = ann.train(X_train, y_train, goal=learning_goal, epochs=max_epochs, show=round(max_epochs/8))
        if train_error[-1]<best_train_error:
            bestnet.append(ann)
            best_train_error = train_error[-1]
            error_hist[range(len(train_error)),k] = train_error
    
    y_est = bestnet[k].sim(X_test)
    y_est = (y_est>.5).astype(int)
    errors[k] = (y_est!=y_test).sum().astype(float)/y_test.shape[0]
    k+=1
    

# Print the average classification error rate
print('Error rate: {0}%'.format(100*np.mean(errors)))


# Display the decision boundary for the several crossvalidation folds.
# (create grid of points, compute network output for each point, color-code and plot).
grid_range = [-1, 2, -1, 2]; delta = 0.05; levels = 100
a = np.arange(grid_range[0],grid_range[1],delta)
b = np.arange(grid_range[2],grid_range[3],delta)
A, B = np.meshgrid(a, b)
values = np.zeros(A.shape)

figure(1,figsize=(18,9)); hold(True)
for k in range(4):
    subplot(2,2,k+1)
    cmask = (y==0).squeeze(); plot(X[cmask,0], X[cmask,1],'.r')
    cmask = (y==1).squeeze(); plot(X[cmask,0], X[cmask,1],'.b')
    title('Model prediction and decision boundary (kfold={0})'.format(k+1))
    xlabel('Feature 1'); ylabel('Feature 2');
    for i in range(len(a)):
        for j in range(len(b)):
            values[i,j] = bestnet[k].sim( np.mat([a[i],b[j]]) )[0,0]
    contour(A, B, values, levels=[.5], colors=['k'], linestyles='dashed')
    contourf(A, B, values, levels=np.linspace(values.min(),values.max(),levels), cmap=cm.RdBu)
    if k==0: colorbar(); legend(['Class A (y=0)', 'Class B (y=1)'])


# Display exemplary networks learning curve (best network of each fold)
figure(2); hold(True)
bn_id = np.argmax(error_hist[-1,:])
error_hist[error_hist==0] = learning_goal
for bn_id in range(K):
    plot(error_hist[:,bn_id]); xlabel('epoch'); ylabel('train error (mse)'); title('Learning curve (best for each CV fold)')

plot(range(max_epochs), [learning_goal]*max_epochs, '-.')


show()