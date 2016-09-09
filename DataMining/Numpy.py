# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 17:26:06 2016

@author: Team31
       : Rutvij Mehta
       : Sagar Gupta
       : Tanay Pandey
"""

import numpy as np



def get_chunks2(s, n):
  return [s[x:x+n] for x in range(0, len(s), n)]
"""
(a) (1 point) Generate a 4*4 matrix A with input from Gaussian Distribution with
mean 5 and variance 1.

"""  
data = np.random.normal(5,1,16)

A =  np.matrix(get_chunks2(data,4))
print "\nAnswer a)\n"
print A

"""
(b) (1 point) Access rows 2 and 4 only.

"""

print "\nAnswer b)\n"
print A[[1,3],:]

"""
c) (2 points) Calculate sum of the 3rd row, the diagonal and the 4th column in the
matrix.
"""
def calculate_sum(A):
    sum = 0
    for x in range(0,len(A)):
        sum = sum + A[2,x] + A[x,3] + A[x,x]
    return sum

print "\nAnswer c):\n"
print str(calculate_sum(A)) + "\n"

"""
(d) (2 points) Sum of all elements in the matrix (use a for/while loop).
"""
sum_matrix = 0
for x in range(0,len(A)):
    for y in range(0,len(A)):
        sum_matrix = sum_matrix + A[x,y]

print "\nAnswer d):\n"
print sum_matrix
"""

(e) (1 point) Generate a diagonal matrix B with from [2, 3, 4, 5] (using this vector as
the diagonal entries).

"""
B = np.diag([2, 3, 4, 5])
print "\nAnswer e)\n"
print str(B) 

"""
(f) (2 points) From A and B, using one matrix operation to get a new matrix C such
that, the first row of C is equal to the first row of A times 2, the second row of C
is equal to the second row of A times 3 and so on.

"""

C = (A.transpose()*B).transpose()
print "\nAnswer f)\n"
print C

"""
(g) (2 points) From A and B, using one matrix operation to get a new matrix D such
that, the first column of D is equal to the first column of A times 2, the second
column of D is equal to the second column of A times 3 and so on.
"""
D = A * B
print "\nAnswer g)\n"
print D

"""
(h) (2 points) X = [1, 2, 3, 4] T , Y = [9, 6, 4, 1] T . Computing the covariance matrix of
X and Y in one function, and calculating the result by basic operations (without
using that function).

"""
X = np.array([1,2,3,4]).transpose()
Y = np.array([9,6,4,1]).transpose()

print "\nAnswer h)\n"

print X
print Y
print np.cov(X,Y)


def print_cov(X,Y):
    n = len(X)
    x_mean = np.mean(X)
    y_mean = np.mean(Y)
    sum_first =0
    sum_last=0
    sum_diag=0
    for i in range(len(X)):
        sum_first = sum_first + ((X[i]**2)-(x_mean**2))
        sum_last = sum_last +((Y[i]**2)-(y_mean)**2)
        sum_diag = sum_diag + ((X[i]*Y[i])-(x_mean*y_mean))
    return np.matrix([[sum_first/(n-1), sum_diag/(n-1)], [sum_diag/(n-1), sum_last/(n-1)]])
   
cov_data = print_cov(X,Y)   
print cov_data


"""
(i) (2 points) Verifying the equation in X: x  ̄ 2 = (x̄ 2 +σ 2 (x)), where σ(x) is the estimate
of the standard deviation.
"""
print "\nAnswer i)\n"
First = np.array([1,2,3,4])
Second = np.matrix([9,6,4,1])

Third = np.array([1,4,9,16])

mean_First = np.mean(First)
mean_Third = np.mean(Third)
Sd = np.std(First)

if(mean_Third == ((mean_First**2) + (Sd**2))):
    print "both are same:"
    

#x = np.array([[0, 2], [1, 1], [2, 0]]).transpose()
#print np.cov(x)