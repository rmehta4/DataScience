# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 23:58:32 2016

@author: rutvij
"""
import scipy.io
import sys
import numpy as np
from scipy import stats
from sklearn import tree

mat = scipy.io.loadmat('hw2.mat')


#############Q1
#print mat.keys()
#print (mat['X_train'].shape)



#####################b
#==============================================================================
# 
new_xtest =stats.zscore(mat['X_test'])
new_ytest =stats.zscore(mat['Y_test'])
# 
new_xtrain =stats.zscore(mat['X_train'])
new_ytrain =stats.zscore(mat['Y_train'])

# 
# print len(new_xtest)
# print len(new_ytest)
# print np.shape(new_xtrain)
# print np.shape(new_ytrain)
# 
#==============================================================================
###############################c

#### calculate co-variance

cov_x = np.cov(np.transpose(new_xtrain))
#cov_y= np.cov(np.transpose(new_ytrain))

print cov_x.shape
#print cov_y

eig_val_cov, eig_vec_cov = np.linalg.eig(cov_x)
print eig_val_cov
print eig_vec_cov
#eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)

srt = np.argsort( eig_val_cov )[::-1]
eig_vec_cov = np.matrix( eig_vec_cov[:,srt] )
eig_val_cov = eig_val_cov[srt]

s = '{:10.2f} '*20

eig_val_pca = eig_vec_cov[:,:20]

print "aaaaaaaaaaaaaaaaaaaaaaaaaaa"
print (eig_val_pca.shape)


reduced_data = new_xtrain * eig_val_pca
print reduced_data.shape


clf = tree.DecisionTreeClassifier()
clf = clf.fit(reduced_data, new_ytrain)

print clf
#==============================================================================
# 
# for i in range(0,len(mat['X_test'])):
#    np_array.append(np.mean(mat['X_test'][i]))
#    np_sd.append(np.std(mat['X_test'][i]))
#    
# 
# 
# for i in range(0,len(mat['X_test'])):
#     print np_sd[i]
#     
#     
# 
# print(len(np_array))
# print(len(np_sd))
# 
# z_score_x_test = []
#==============================================================================

