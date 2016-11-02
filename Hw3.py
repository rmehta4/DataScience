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


"""
Question a

"""

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
       

"""

Question b

train_test_split will randomly distribute data in (training , label ) pair

"""
X_train, X_test, y_train, y_test = train_test_split(mat_X, result, test_size=0.4)


"""

Question c

"""

clf_rbf = svm.SVC(kernel='rbf',C=10.0)
clf_poly = svm.SVC(kernel='poly',C=1.0,gamma=1.0)
clf_linear = svm.SVC(kernel='linear',C=1.0)
clf_sigmoid = svm.SVC(kernel='sigmoid',C=1.0,gamma=0.1,coef0 =0.3)
clf_quadratic = svm.SVC(kernel='poly',C=1.0,degree=2,gamma=1.0)


clf_rbf.fit(X_train, y_train)
y_rbf=clf_rbf.predict(X_test)

clf_poly.fit(X_train, y_train)
y_poly=clf_poly.predict(X_test)

clf_linear.fit(X_train, y_train)
y_linear=clf_linear.predict(X_test)

clf_sigmoid.fit(X_train, y_train)
y_sigmoid=clf_sigmoid.predict(X_test)

clf_quadratic.fit(X_train, y_train)
y_quadratic=clf_quadratic.predict(X_test)

print "rbf score"
print clf_rbf.score(X_test,y_test)
print "poly score"
print clf_poly.score(X_test,y_test)
print "linear score"
print clf_linear.score(X_test,y_test)
print "sigmoid score"
print clf_sigmoid.score(X_test,y_test)
print "quadratic score"
print clf_quadratic.score(X_test,y_test)
print "\n\n\n"

"""

Question d

"""
print "+++++++++++++++++++++++ rbf metrics +++++++++++++++++++++++"
print metrics.classification_report(y_test, y_rbf,digits=4)
print "+++++++++++++++++++++++ poly metrics ++++++++++++++++++++++++"
print metrics.classification_report(y_test, y_poly,digits=4)
print "+++++++++++++++++++++++ linear metrics ++++++++++++++++++++++"
print metrics.classification_report(y_test, y_linear,digits=4)
print "+++++++++++++++++++++++ sigmoid metrics +++++++++++++++++++++"
print metrics.classification_report(y_test, y_sigmoid,digits=4)
print "+++++++++++++++++++++++ quadratic metrics ++++++++++++++++++++++++"
print metrics.classification_report(y_test, y_quadratic,digits=4)













