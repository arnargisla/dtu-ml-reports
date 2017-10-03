# -*- coding: utf-8 -*-
"""
...
"""
import numpy as np
import xlrd
import matplotlib.pyplot as plt
from scipy.linalg import svd
from scipy.stats import zscore

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

# Compute values of N, M and C.
#N = len(y)
N = X.shape[0]
M = len(attributeNames)
#C = len(classNames)

#%% boxplots of data
fig = plt.figure(figsize=(10, 5))
box_plot_cols = [54, 55, 56, 57]
sub_plot_count = 1
for col_id in box_plot_cols:        
    ax = fig.add_subplot(1, len(box_plot_cols), sub_plot_count)
    #ax.title.set_text()
    plt.boxplot(X[:, col_id])
    plt.xlabel(attributeNames[col_id])
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off') # labels along the bottom edge are off
    sub_plot_count += 1
            
fig.suptitle('Boxplots of murder rate, execution rate, unemployment and democratic voting percentage \nfrom the data set. Before removing DC.')
#plt.tight_layout()
fig.tight_layout(rect=[0, 0.03, 1, 0.90])
#plt.show()
plt.savefig("project-1-box-plot-before-removing-dc.png", bbox_inches='tight')




#%% We can see that DC is creating outliers in the murder rate so we try removing them and check what we get
X = np.delete(X, (24, 25, 26), axis=0) # rows 24 25 26 are dc
X = np.delete(X, (8), axis=1) # col 8 is dc
box_plot_cols = [53, 54, 55, 56]
attributeNames = np.delete(attributeNames, (8))
N = X.shape[0]
M = X.shape[1]

#%% boxplots of data without DC
fig = plt.figure(figsize=(10, 5))
sub_plot_count = 1
for col_id in box_plot_cols:        
    ax = fig.add_subplot(1, len(box_plot_cols), sub_plot_count)
    #ax.title.set_text()
    plt.boxplot(X[:, col_id])
    plt.xlabel(attributeNames[col_id])
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off') # labels along the bottom edge are off
    sub_plot_count += 1
            
fig.suptitle('Boxplots of murder rate, execution rate, unemployment and democratic voting percentage \nfrom the data set. After removing DC.')
#plt.tight_layout()
fig.tight_layout(rect=[0, 0.03, 1, 0.90])
#plt.show()
plt.savefig("project-1-box-plot-after-removing-dc.png", bbox_inches='tight')

#%% Standardize

# Subtract mean value from data
Y = (X - np.ones((N,1))*X.mean(0))
# Standardize the data
for col_id in range(M):
    bottom = X[:,col_id].std(ddof=1)
    top =  Y[:,col_id]
    Y[:,col_id] = top/bottom
    

#%% boxplots standardized
fig = plt.figure(figsize=(10, 5))
sub_plot_count = 1
for col_id in box_plot_cols:        
    ax = fig.add_subplot(1, len(box_plot_cols), sub_plot_count)
    #ax.title.set_text()
    plt.boxplot(Y[:, col_id])
    plt.xlabel(attributeNames[col_id])
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off') # labels along the bottom edge are off
    sub_plot_count += 1
            
fig.suptitle('Standardized boxplots of murder rate, execution rate, unemployment and democratic voting percentage \nfrom the data set. After removing DC.')
#plt.tight_layout()
fig.tight_layout(rect=[0, 0.03, 1, 0.90])
#plt.show()
plt.savefig("project-1-standardized-box-plot-after-removing-dc.png", bbox_inches='tight')

#%% Standardized Scatter plot matrix
Attributes = box_plot_cols
NumAtr = len(Attributes)

plt.figure(figsize=(12,12))
plt.hold(True)

for m1 in range(NumAtr):
    for m2 in range(NumAtr):
        plt.subplot(NumAtr, NumAtr, m1*NumAtr + m2 + 1)
        #for c in range(C):
        #class_mask = (y==c)
        plt.plot(Y[:,Attributes[m2]], Y[:,Attributes[m1]], '.')
        if m1==NumAtr-1:
            plt.xlabel(attributeNames[Attributes[m2]])
        else:
            plt.xticks([])
        if m2==0:
            plt.ylabel(attributeNames[Attributes[m1]])
        else:
            plt.yticks([])
        #ylim(0,X.max()*1.1)
        #xlim(0,X.max()*1.1)
#plt.legend(attributeNames)
plt.suptitle('Standardized scatter plot matrix of murder rate, execution rate, unemployment and democratic voting percentage \nfrom the data set. After removing DC.')

plt.savefig("project-1-standardized-scatter-plot-matrix-after-removing-dc.png", bbox_inches='tight')

#%% Scatter plot matrix
Attributes = box_plot_cols
NumAtr = len(Attributes)

plt.figure(figsize=(12,12))
plt.hold(True)

for m1 in range(NumAtr):
    for m2 in range(NumAtr):
        plt.subplot(NumAtr, NumAtr, m1*NumAtr + m2 + 1)
        #for c in range(C):
        #class_mask = (y==c)
        plt.plot(X[:,Attributes[m2]], X[:,Attributes[m1]], '.')
        if m1==NumAtr-1:
            plt.xlabel(attributeNames[Attributes[m2]])
        else:
            plt.xticks([])
        if m2==0:
            plt.ylabel(attributeNames[Attributes[m1]])
        else:
            plt.yticks([])
        #ylim(0,X.max()*1.1)
        #xlim(0,X.max()*1.1)
#plt.legend(attributeNames)
plt.suptitle('Scatter plot matrix of murder rate, execution rate, unemployment and democratic voting percentage \nfrom the data set. After removing DC.')

plt.savefig("project-1-scatter-plot-matrix-after-removing-dc.png", bbox_inches='tight')


#%% Standard Histograms

plt.figure(figsize=(8,7))
u = np.floor(np.sqrt(len(box_plot_cols))); v = np.ceil(len(box_plot_cols)/u)
for i, col_id in enumerate(box_plot_cols):
    plt.subplot(u,v,i+1)
    plt.hist(Y[:,col_id], color=(0.2, 0.8-i*0.2, 0.4))
    plt.xlabel(attributeNames[col_id])
    plt.ylim(0,N/2)
 
plt.suptitle('Standardized historgrams of murder rate, execution rate, unemployment and democratic voting percentage \nfrom the data set. After removing DC.')

plt.savefig("project-1-standardized-histgrams-after-removing-dc.png", bbox_inches='tight')


#%% Histograms

plt.figure(figsize=(8,7))
u = np.floor(np.sqrt(len(box_plot_cols))); v = np.ceil(len(box_plot_cols)/u)
for i, col_id in enumerate(box_plot_cols):
    plt.subplot(u,v,i+1)
    plt.hist(X[:,col_id], color=(0.2, 0.8-i*0.2, 0.4))
    plt.xlabel(attributeNames[col_id])
    #plt.ylim(0,N/2)
    print(max(X[:,col_id]))
 
plt.suptitle('Historgrams of murder rate, execution rate, unemployment and democratic voting percentage \nfrom the data set. After removing DC.')

plt.savefig("project-1-histgrams-after-removing-dc.png", bbox_inches='tight')



#%% PCA
# PCA by computing SVD of Y
U,S,V = svd(Y,full_matrices=False)

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum()

#%% Plot pca

# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'o-')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.savefig("project-1-variance-explained-by-principal-compinents.png", bbox_inches='tight')

#%% projection onto pc1 and pc2
pc1 = V[:,0]
pc2 = V[:,1]

A = Y @ pc1
B = Y @ pc2

plt.figure()
plt.title('Projection of data onto pc1 and pc2');
plt.xlabel('pc1');
plt.ylabel('pc2');
plt.plot(A, B, 'bo', fillstyle='none')
plt.savefig("project-1-projection-onto-pc1-and-pc2.png", bbox_inches='tight')


#%% look at biggest contributors in pc1 and pc2
def getKey(pair):
    return pair[1]
for pc, pcname in [(pc1, "pc1"), (pc2, "pc2")]:
    plt.figure(figsize=(12, 3))
    k = 0
    for i, x in sorted(enumerate(np.abs(pc)), key=getKey, reverse=True):
        pass
        #print(attributeNames[i], i, x, pc[i])
    plt.plot(range(len(pc)), sorted(np.abs(pc), reverse=True))
    plt.xticks(range(len(pc)), [p[0] for p in sorted(zip(attributeNames, np.abs(pc)), key=getKey, reverse=True)], rotation=90)
    plt.savefig("project-1-absolute-value-of-{}-coefficients.png".format(pcname), bbox_inches='tight')


#%% PCA2 without states this time
# PCA by computing SVD of Y
Y2 = Y[:, box_plot_cols]
U,S,V = svd(Y2,full_matrices=False)

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum()

pc1 = V[:,0]
pc2 = V[:,1]

A = Y2 @ pc1
B = Y2 @ pc2

# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'o-')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.savefig("project-1-variance-explained-by-principal-compinents-without-states.png", bbox_inches='tight')


plt.figure()
plt.title('Projection of data onto pc1 and pc2');
plt.xlabel('pc1');
plt.ylabel('pc2');
plt.plot(A, B, 'bo', fillstyle='none')
plt.savefig("project-1-projection-onto-pc1-and-pc2-without-states.png", bbox_inches='tight')

#%% look at biggest contributors in pc1 and pc2 without states
attributeNames2 = attributeNames[box_plot_cols]
def getKey(pair):
    return pair[1]

plt.figure(figsize=(8, 3))
for pc, pcname in [(pc1, "pc1"), (pc2, "pc2")]:
    index = 1
    if pcname == "pc1":
        index = 2
    plt.subplot(1, 2, index)
    plt.title("Coefficients of {}".format(pcname))
    k = 0
    for i, x in sorted(enumerate(np.abs(pc)), key=getKey, reverse=True):
        pass
        #print(attributeNames[i], i, x, pc[i])
    plt.plot(range(len(pc)), sorted(np.abs(pc), reverse=True))
    plt.xticks(range(len(pc)), [p[0] for p in sorted(zip(attributeNames2, np.abs(pc)), key=getKey, reverse=True)], rotation=90)

plt.savefig("project-1-absolute-value-of-pc1-and-pc2-coefficients-without-states.png".format(pcname), bbox_inches='tight')


#%%
#plt.figure(figsize=(12,6))
#plt.title('Wine: Boxplot (standarized)')
#plt.boxplot(zscore(X, ddof=1), attributeNames)
#plt.xticks(range(1,M+1), attributeNames, rotation=45)

#%%
# We start with a box plot of each attribute
#plt.figure(figsize=(10, 6))
#plt.title('Murder data: Boxplot')
#plt.boxplot(X)
#plt.xticks(range(1,M+1), attributeNames, rotation=45)

#%% boxplots of data
#plt.figure()
#for col_id in box_plot_cols:
#    plt.subplot(1, ncols, col_id+1)
#    plt.boxplot(Y[:, col_id])
#    plt.tight_layout()
#plt.title('Boxplots before removing ouliers');
#plt.show()


#%% mask
#Y = np.asarray(Y)z
#outlier_mask = (Y[:,2]>4) | (Y[:,3]>100) | (Y[:,4]>3)
#valid_mask = np.logical_not(outlier_mask)

# Finally we will remove these from the data set
#Y = Y[valid_mask,:]

#%% boxplots of data
#plt.figure()
#plt.plot()
#plt.title('Boxplots after removing ouliers');
#for col_id in range(2,ncols):
#    plt.subplot(1, ncols-2, col_id+1-2)
#    plt.boxplot(Y[:, col_id])
#    plt.tight_layout()    
#    y_up = Y.max()+(Y.max()-Y.min())*0.1
#    y_down = Y.min()-(Y.max()-Y.min())*0.1
#    #ylim(y_down, y_up)

#splt.show()
