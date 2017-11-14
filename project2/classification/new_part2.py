# -*- coding: utf-8 -*-
"""
...
"""
import numpy as np
import xlrd
import matplotlib.pyplot as plt
from scipy.linalg import svd
from scipy.stats import zscore
from sklearn import tree,model_selection,cross_validation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from matplotlib.pylab import figure, plot, xlabel, ylabel, legend, ylim, show
import sklearn.linear_model as lm
from matplotlib.pyplot import (figure, plot, subplot, title, xlabel, ylabel, 
                               hold, contour, contourf, cm, colorbar, show,
                               legend)

#import neurolab as nl

# requires data from exercise 5.1.4



#import dot2tex
#file=open('tree_gini.gvz', 'r') 
#testgraph = file.read()
#
#texcode = dot2tex.dot2tex(testgraph, format='tikz', crop=True)
#% Shaping the data

# Load xls sheet with data
doc = xlrd.open_workbook('one-out-of-k-murder.xls').sheet_by_index(0)
ncols = doc.ncols
nrows = doc.nrows
ndatarows = nrows-1

# Extract attribute names (1st row, column 4 to 12)
attributeNames = doc.row_values(0, 0, ncols)


# Extract class names to python list,
# then encode with integers (dict)
#classLabels = doc.col_values(0, 1, nrows)
#classNames = sorted(set(classLabels))
#classDict = dict(zip(classNames, range(len(classNames))))

# Extract vector y, convert to NumPy matrix and transpose
#y = np.mat([classDict[value] for value in classLabels]).T

# Preallocate memory, then extract excel data to matrix X
X = np.mat(np.empty((ndatarows, ncols)))
for i, col_id in enumerate(range(ncols)):
    X[:, i] = np.mat(doc.col_values(col_id, 1, nrows)).T

X = np.delete(X, (24, 25, 26), axis=0) # rows 24 25 26 are dc
X = np.delete(X, (8), axis=1) # col 8 is dc
box_plot_cols = [53, 54, 55, 56]
attributeNames = np.delete(attributeNames, (8))
N = X.shape[0]
M = X.shape[1]

##%%
##use decision trees to find the which observation belongs to which state
#X_with_no_state=X[:,53:]
#y=np.kron(np.arange(1,51),np.ones((1,3))).T
#
#
#dtc = tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=4)
#dtc = dtc.fit(X_with_no_state,y)
#
## Export tree graph for visualization purposes:
## (note: you can use i.e. Graphviz application to visualize the file)
#out = tree.export_graphviz(dtc, out_file='tree_state_entropy.gvz', feature_names=attributeNames[53:])
#
##%%
##use decision trees to find the which observation has death penalty and which does not
#X_with_no_exec=np.concatenate((X[:,53:-3],X[:,-2:]),axis=1)
#y=(X[:,-3]>0.5).astype(int)
#
#
#dtc = tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=15)
#dtc = dtc.fit(X_with_no_exec,y)
#
## Export tree graph for visualization purposes:
## (note: you can use i.e. Graphviz application to visualize the file)
#out = tree.export_graphviz(dtc, out_file='tree_exec_entropy.gvz', feature_names=np.concatenate((attributeNames[53:-3],attributeNames[-2:]),axis=0))
#
##%% same with linear regression calssification
#ydim,xdim=y.shape
#ynew=np.zeros((ydim,))
#for i,v in enumerate(y):
#    ynew[i]=v
#y=ynew
#
## Fit logistic regression model
#model = lm.logistic.LogisticRegression()
#model = model.fit(X_with_no_exec,y)
#
## Classify wine as White/Red (0/1) and assess probabilities
#y_est = model.predict(X_with_no_exec)
#y_est_dmcrtvoterte_prob = model.predict_proba(X_with_no_exec)[:, 0] 
#
#
#
## Evaluate classifier's misclassification rate over entire training data
#misclass_rate = sum(np.abs(y_est - y)) / float(len(y_est))
#
## Display classification results
#
#print('\nOverall misclassification rate: {0:.3f}'.format(misclass_rate))
#
#f = figure(); f.hold(True)
#class0_ids = np.nonzero(y==0)[0].tolist()
#plot(class0_ids, y_est_dmcrtvoterte_prob[class0_ids],'.',color='lightgreen')
#class1_ids = np.nonzero(y==1)[0].tolist()
#plot(class1_ids, y_est_dmcrtvoterte_prob[class1_ids],'.k')
#xlabel('Data object (observation)'); ylabel('Predicted prob. of class dmcrt');
#legend(['no executions', 'executions'])
#ylim(-0.01,1.5)
#
#show()
#%%
#use decision trees to find the which observation votes more democratic or republican
X_with_no_dmcrtvoterte=X[:,50:-1]
y=(X[:,-1]>0.5).astype(int)
#print(y.shape)
#print(X_with_no_dmcrtvoterte)


dtc = tree.DecisionTreeClassifier(criterion='gini', min_samples_split=20)
dtc = dtc.fit(X_with_no_dmcrtvoterte,y)

# Export tree graph for visualization purposes:
# (note: you can use i.e. Graphviz application to visualize the file)
out = tree.export_graphviz(dtc, out_file='tree_dmcrtvote_entropy.gvz', feature_names=attributeNames[50:-1])
#==============================================================================
# 
#==============================================================================
#%% linear regression calssification ##########################################
ydim,xdim=y.shape
ynew=np.zeros((ydim,))
for i,v in enumerate(y):
    ynew[i]=v
y=ynew

# Fit logistic regression model
model = lm.logistic.LogisticRegression()
model = model.fit(X_with_no_dmcrtvoterte,y)

# Classify wine as White/Red (0/1) and assess probabilities
y_est = model.predict(X_with_no_dmcrtvoterte)
y_est_dmcrtvoterte_prob = model.predict_proba(X_with_no_dmcrtvoterte)[:, 0] 



# Evaluate classifier's misclassification rate over entire training data
misclass_rate = sum(np.abs(y_est - y)) / float(len(y_est))

# Display classification results

print('\nOverall misclassification rate: {0:.3f}'.format(misclass_rate))

f = figure(); f.hold(True)
class0_ids = np.nonzero(y==0)[0].tolist()
plot(class0_ids, y_est_dmcrtvoterte_prob[class0_ids], '.b')
class1_ids = np.nonzero(y==1)[0].tolist()
plot(class1_ids, y_est_dmcrtvoterte_prob[class1_ids], '.r')
xlabel('Data object (observation)'); ylabel('Predicted prob. of class dmcrt');
legend(['dmcrtvote', 'repvote'])
ylim(-0.01,1.5)
plt.savefig('lin_class_vote.png')
show()
#==============================================================================
# 
#==============================================================================
#%% KNN #############################################
# Maximum number of neighbors
L=40

CV = cross_validation.LeaveOneOut(N)
errors = np.zeros((N,L))
i=0
for train_index, test_index in CV:
    print('Crossvalidation fold: {0}/{1}'.format(i+1,N))    
    
    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]

    # Fit classifier and classify the test points (consider 1 to 40 neighbors)
    for l in range(1,L+1):
        knclassifier = KNeighborsClassifier(n_neighbors=l);
        knclassifier.fit(X_train, y_train);
        y_est = knclassifier.predict(X_test);
        errors[i,l-1] = np.sum(y_est[0]!=y_test[0])

    i+=1
    
# Plot the classification error rate
figure()
plot(100*sum(errors,0)/N)
xlabel('Number of neighbors')
ylabel('Classification error rate (%)')
plt.savefig('knn_decide_class_vote.png')
show()

#f = figure(); f.hold(True)
#for train_index, test_index in CV:
#    print('Crossvalidation fold: {0}/{1}'.format(i+1,N))    
#    
#    # extract training and test set for current CV fold
#    X_train = X[train_index,:]
#    y_train = y[train_index]
#    X_test = X[test_index,:]
#    y_test = y[test_index]
#
#    # Fit classifier and classify the test points (consider 1 to 40 neighbors)
#    for l in [7]:
#        knclassifier = KNeighborsClassifier(n_neighbors=l);
#        knclassifier.fit(X_train, y_train);
#        y_est = knclassifier.predict(X_test);
#        #errors[i,l-1] = np.sum(y_est[0]!=y_test[0])
#
#    class0_ids = np.nonzero(y_test<0.5)[0].tolist()
#    yplot0=[]
#    for i in y_est[class0_ids]:
#        if i==0:
#            yplot0.append(1)
#        else:
#            yplot0.append(0)
#    
#    plot([i], yplot0, '.b')
#    class1_ids = np.nonzero(y_test>0.5)[0].tolist()
#    yplot1=[]
#    for i in y_est[class1_ids]:
#        if i==0:
#            yplot1.append(0)
#        else:
#            yplot1.append(1)
#    plot([i], yplot1, '.r')
#    
#
#    i+=1
#xlabel('Data object (observation)'); ylabel('Correctly (1) or uncorrectly (0) classified');
#legend(['dmcrtvote', 'repvote'])
#ylim(-0.01,1.5)
#plt.savefig('knn_class_vote.png')
#show()   

#f = figure(); f.hold(True)
#class0_ids = np.nonzero(y_test==0)[0].tolist()
#yplot0=[]
#for i in y_est[class0_ids]:
#    if i==0:
#        yplot0.append(1)
#    else:
#        yplot0.append(0)
#
#plot(class0_ids, yplot0, '.b')
#class1_ids = np.nonzero(y_test==1)[0].tolist()
#yplot1=[]
#for i in y_est[class1_ids]:
#    if i==0:
#        yplot1.append(0)
#    else:
#        yplot1.append(1)
#plot(class1_ids, yplot1, '.r')
#xlabel('Data object (observation)'); ylabel('Correctly (1) or uncorrectly (0) classified');
#legend(['dmcrtvote', 'repvote'])
#ylim(-0.01,1.5)
#plt.savefig('knn_class_vote.png')
#show()
#==============================================================================
# 
#==============================================================================
##%% Naive Bayes ##############################################################
#classNames = ['dmcrtvote', 'repvote'];
#
#y = y.squeeze()
#
## Naive Bayes classifier parameters
#alpha = 1.0         # additive parameter (e.g. Laplace correction)
#est_prior = True   # uniform prior (change to True to estimate prior from data)
#
## K-fold crossvalidation
#K = 10
#CV = cross_validation.KFold(N,K,shuffle=True)
#
#errors = np.zeros(K)
#k=0
#for train_index, test_index in CV:
#    print('Crossvalidation fold: {0}/{1}'.format(k+1,K))    
#    
#    # extract training and test set for current CV fold
#    X_train = X[train_index,:]
#    y_train = y[train_index]
#    X_test = X[test_index,:]
#    y_test = y[test_index]
#    
#    nb_classifier = MultinomialNB(alpha=alpha, fit_prior=est_prior)
#    nb_classifier.fit(X_train, y_train)
#    y_est_prob = nb_classifier.predict_proba(X_test)
#    y_est = np.argmax(y_est_prob,1)
#    
#    errors[k] = np.sum(y_est!=y_test,dtype=float)/y_test.shape[0]
#    k+=1
#    
## Plot the classification error rate
#print('Error rate: {0}%'.format(100*np.mean(errors)))
#==============================================================================
# 
#==============================================================================
##%% ANN ####################################################################
#
#
#ynew=np.zeros((len(y),2))
#for i,v in enumerate(y):
#    if v==0:
#        ynew[i,:]=[1,0]
#    else:
#        ynew[i,:]=[0,1]
#y=ynew
#print(y)
#
#
## Parameters for neural network classifier
#n_hidden_units = 2      # number of hidden units
#n_train = 2             # number of networks trained in each k-fold
#
## These parameters are usually adjusted to: (1) data specifics, (2) computational constraints
#learning_goal = 2.0     # stop criterion 1 (train mse to be reached)
#max_epochs = 200        # stop criterion 2 (max epochs in training)
#
## K-fold CrossValidation (4 folds here to speed up this example)
#K = 4
#CV = model_selection.KFold(K,shuffle=True)
#
## Variable for classification error
#errors = np.zeros(K)
#error_hist = np.zeros((max_epochs,K))
#bestnet = list()
#k=0
#for train_index, test_index in CV.split(X,y):
#    print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))    
#    
#    # extract training and test set for current CV fold
#    X_train = X[train_index,:]
#    y_train = y[train_index,:]
#    X_test = X[test_index,:]
#    y_test = y[test_index,:]
#    print(y_train.shape)
#    print(X_train.shape)
#    best_train_error = 1e100
#    for i in range(n_train):
#        # Create randomly initialized network with 2 layers
#        ann = nl.net.newff([[0, 1], [0, 1]], [n_hidden_units, 1], [nl.trans.TanSig(),nl.trans.PureLin()])
#        # train network
#        train_error = ann.train(X_train, y_train, goal=learning_goal, epochs=max_epochs, show=round(max_epochs/8))
#        if train_error[-1]<best_train_error:
#            bestnet.append(ann)
#            best_train_error = train_error[-1]
#            error_hist[range(len(train_error)),k] = train_error
#    
#    y_est = bestnet[k].sim(X_test)
#    y_est = (y_est>.5).astype(int)
#    errors[k] = (y_est!=y_test).sum().astype(float)/y_test.shape[0]
#    k+=1
#    
#
## Print the average classification error rate
#print('Error rate: {0}%'.format(100*np.mean(errors)))
#
#
## Display the decision boundary for the several crossvalidation folds.
## (create grid of points, compute network output for each point, color-code and plot).
#grid_range = [-1, 2, -1, 2]; delta = 0.05; levels = 100
#a = np.arange(grid_range[0],grid_range[1],delta)
#b = np.arange(grid_range[2],grid_range[3],delta)
#A, B = np.meshgrid(a, b)
#values = np.zeros(A.shape)
#
#figure(1,figsize=(18,9)); hold(True)
#for k in range(4):
#    subplot(2,2,k+1)
#    cmask = (y==0).squeeze(); plot(X[cmask,0], X[cmask,1],'.r')
#    cmask = (y==1).squeeze(); plot(X[cmask,0], X[cmask,1],'.b')
#    title('Model prediction and decision boundary (kfold={0})'.format(k+1))
#    xlabel('Feature 1'); ylabel('Feature 2');
#    for i in range(len(a)):
#        for j in range(len(b)):
#            values[i,j] = bestnet[k].sim( np.mat([a[i],b[j]]) )[0,0]
#    contour(A, B, values, levels=[.5], colors=['k'], linestyles='dashed')
#    contourf(A, B, values, levels=np.linspace(values.min(),values.max(),levels), cmap=cm.RdBu)
#    if k==0: colorbar(); legend(['Class A (y=0)', 'Class B (y=1)'])
#
#
## Display exemplary networks learning curve (best network of each fold)
#figure(2); hold(True)
#bn_id = np.argmax(error_hist[-1,:])
#error_hist[error_hist==0] = learning_goal
#for bn_id in range(K):
#    plot(error_hist[:,bn_id]); xlabel('epoch'); ylabel('train error (mse)'); title('Learning curve (best for each CV fold)')
#
#plot(range(max_epochs), [learning_goal]*max_epochs, '-.')
#
#
#show()