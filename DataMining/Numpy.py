# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 17:26:06 2016

@author: rutvij
"""

import numpy as np

s = np.random.normal(5,1, 16)
def get_chunks2(s, n):
  return [s[x:x+n] for x in range(0, len(s), n)]
  
A =  np.matrix(get_chunks2(s,4))
#print mat

#print mat[[1,3],:]

#print mat[:,3]
B = np.diag([2, 3, 4, 5])

C = A * B  ## Que g

D = (A.transpose()*B).transpose() ## Que f
#print C
#print A
#print D

First = np.array([1,2,3,4])
Second = np.matrix([9,6,4,1])

Third = np.array([1,4,9,16])

mean_First = np.mean(First)
mean_Second = np.mean(Third)

#print  mean_First
print mean_Second

Sd = np.std(First)
#print Sd
print (mean_First**2) + (Sd**2)

#x = np.array([[0, 2], [1, 1], [2, 0]]).transpose()
#print np.cov(x)