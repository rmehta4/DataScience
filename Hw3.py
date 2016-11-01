# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 15:38:03 2016

@author: rutvij
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 16:41:06 2016

@author: Team31
       : Rutvij Mehta
       : Sagar Gupta
       : Tanay Pandey
"""
import scipy.io
import numpy as np
from sklearn import svm , grid_search
from sklearn import metrics
from sklearn.cross_validation import train_test_split

mat = scipy.io.loadmat('hw3data.mat')
print mat.keys()
mat_Y = mat['Y_target']
print (mat['Y_target']).shape
print (mat['X']).shape
mat_X = mat['X'].T
result = []

for i in range(0,len(mat_Y[0])):
    if(mat_Y[0][i]==1):
        result.append(1)
    elif(mat_Y[1][i] == 1):
       result.append(2)
      
X_train, X_test, y_train, y_test = train_test_split(mat_X, result, test_size=0.4)

C_range = 10.0 ** np.arange(-2, 2)
gamma_range = 10.0 ** np.arange(-2, 2)

parameters = {'kernel':('linear','poly','sigmoid'),'gamma':gamma_range.tolist(), 'C':C_range.tolist()}
#'kernel':('linear','poly','rbf','sigmoid')
svr = svm.SVC()
clf = grid_search.GridSearchCV(svr,parameters)
clf.fit(X_train, y_train)
y_predicted=clf.predict(X_test)
print clf.score(X_test,y_test)

#print clf.best_estimator_


print "Classification report for %s" % clf
print
print metrics.classification_report(y_test, y_predicted)
print
print "Confusion matrix"
print metrics.confusion_matrix(y_test, y_predicted)
#==============================================================================
# np.random.shuffle(result)
# 
# training_set = np.array(result[:int(len(result)*0.6)])
# testing_set = np.array(result [:int(len(result)*0.4)])
# print (testing_set).shape
# 
# print(training_set).shape
# #training_set.reshape(training_set,1)
# Y = np.squeeze(np.asarray(training_set))
# print Y.shape
# #clf.fit(training_set,Y)
# 
# X = mat['X'].T
# print(X[:1200][0])
# clf = svm.SVC()
# clf.fit(X[:1200], training_set)
# clf.predict(X[:800])
# 
# 
#==============================================================================
















