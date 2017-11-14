#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 17:53:06 2017

@author: Onat1
"""
#%%
import math
from matplotlib.pyplot import figure, plot, subplot, xlabel, ylabel, show, clim, savefig, suptitle
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import cross_validation
from toolbox_02450 import feature_selector_lr, bmplot
import numpy as np
import neurolab as nl
from sklearn import model_selection
from matplotlib.pyplot import (figure, plot, subplot, title, xlabel, ylabel, 
                               hold, contour, contourf, cm, colorbar, show,
                               legend)
from matplotlib.colors import LinearSegmentedColormap

#%%

# Load data from matlab file
mat_data = loadmat('murder.mat')
X = mat_data['X']
y = mat_data['y']
attributeNames = [name[0] for name in mat_data['attributeNames'].squeeze()]
N, M = X.shape

# Parameters for neural network classifier
n_hidden_units = 10      # number of hidden units
n_train = 5              # number of networks trained in each k-fold

# These parameters are usually adjusted to: (1) data specifics, (2) computational constraints
learning_goal = 0.05     # stop criterion 1 (train mse to be reached)
max_epochs = 250         # stop criterion 2 (max epochs in training)

# K-fold CrossValidation
K = 10
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
        ann = nl.net.newff([
                [-100, 100]]*6, 
                [n_hidden_units, 1], 
                [nl.trans.TanSig(),nl.trans.PureLin()
            ])
        # train network
        train_error = ann.train(
                X_train, y_train, 
                goal=learning_goal, 
                epochs=max_epochs, 
                show=round(max_epochs/8)
            )
        if train_error[-1]<best_train_error:
            bestnet.append(ann)
            best_train_error = train_error[-1]
            error_hist[range(len(train_error)),k] = train_error
    
    y_est = bestnet[k].sim(X_test)
    errors[k] = ((y_est - y_test) * (y_est - y_test)).sum()/y_test.shape[0]
    e2 = ((y_est - y_test)).sum()/y_test.shape[0]
    print("train_error", train_error[-1], errors[k], e2)
    k+=1
    

# Print the average error
print('Average error: {0}%'.format(100*np.mean(errors)))

#%% Plotting

# Display the decision boundary for the several crossvalidation folds.
# (create grid of points, compute network output for each point, 
# color-code and plot).
for prop1_idx, prop2_idx in [[0, 4], [3, 5], [3, 4], [4, 5]]:
    grid_range = [X[:,prop1_idx].min(), X[:,prop1_idx].max(), X[:,prop2_idx].min(), X[:,prop2_idx].max()]; 
    a_delta = (X[:,prop1_idx].max() - X[:,prop1_idx].min()) * 1.0/100;  
    b_delta = (X[:,prop2_idx].max() - X[:,prop2_idx].min()) * 1.0/100;
    levels = 100
    a = np.arange(grid_range[0],grid_range[1],a_delta)
    b = np.arange(grid_range[2],grid_range[3],b_delta)
    A, B = np.meshgrid(a, b)
    values = np.zeros(A.shape)
    
    figure(1,figsize=(18,4*K)); hold(True)
    for k in range(K):
        subplot(math.ceil(K*0.5), 2, k+1)
        title('Model prediction and decision boundary (kfold={0})'.format(k+1))
        xlabel('{}'.format(attributeNames[prop1_idx])); 
        ylabel('{}'.format(attributeNames[prop2_idx]));
        for i in range(len(a)):
            for j in range(len(b)):
                sim_point = [
                    X[:,0].mean(), 
                    X[:,1].mean(), 
                    X[:,2].mean(), 
                    X[:,3].mean(), 
                    X[:,4].mean(), 
                    X[:,5].mean() ]
                sim_point[prop1_idx] = a[i]
                sim_point[prop2_idx] = b[j]
                values[j,i] = bestnet[k].sim( np.mat(sim_point))[0,0]

        lo = y.min()
        hi = y.max()
        mid = lo + (hi-lo) * 0.5
        
        
        #contour(A, B, values, levels=[lo, mid, hi], colors=['m', 'k', 'y'], linestyles='dashed')
        print("values:", values.min(), values.max(), lo, hi)
        contourf(A, B, values,
                 vmin=lo, vmax=hi,
                 #[lo, mid, hi],
                 #levels=np.linspace(values.min(),values.max(),levels)
                 cmap=cm.bwr
                 )
        # if k==0: colorbar(); legend(['Class A (y=0)', 'Class B (y=1)'])
        colorbar(); #legend(['Class A (y=0)', 'Class B (y=1)'])

    show()
    

# Display exemplary networks learning curve (best network of each fold)
figure(2); hold(True)
bn_id = np.argmax(error_hist[-1,:])
error_hist[error_hist==0] = learning_goal
for bn_id in range(K):
    plot(error_hist[:,bn_id]); xlabel('epoch'); ylabel('train error (mse)'); 
    title('Learning curve (best for each CV fold)')

plot(range(max_epochs), [learning_goal]*max_epochs, '-.')


show()