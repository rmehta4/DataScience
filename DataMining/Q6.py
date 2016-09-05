# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 18:28:00 2016

@author: rutvij
"""

import numpy as np
import pandas as pd
import matplotlib.mlab as mlab
from scipy import spatial
from scipy import stats

import matplotlib.pyplot as plt
import pylab

#from numpy.linalg import inv
#np.load()
data = pd.read_csv('hw1q6.csv', sep=',',header=-1)
print data

###For x1

x1_mean = np.mean(data[0])
x1_median = np.median(data[0])
x1_sd=np.std(data[0])
x1_range=(np.max(data[0])-np.min(data[0]))

print x1_mean
print x1_median
print x1_sd
print x1_range
print "\n"

Q1_x,Q2_x,Q3_x,Q4_x,Q5_x = np.percentile(data[0],[0,25,50,75,100])

print Q1_x
print Q2_x
print Q3_x
print Q4_x
print Q5_x
print "\n"
####For x2
x2_mean = np.mean(data[1])
x2_median = np.median(data[1])
x2_sd=np.std(data[1])
x2_range=(np.max(data[1])-np.min(data[1]))

print x2_mean
print x2_median
print x2_sd
print x2_range

Q1_y,Q2_y,Q3_y,Q4_y,Q5_y = np.percentile(data[1],[0,25,50,75,100])

print Q1_y
print Q2_y
print Q3_y
print Q4_y
print Q5_y



#Qc
plt.subplot(2,1,1)
n1,bins1,patches1 = plt.hist(data[0],10,normed=True)
#ax4.hist(data[1],bins=10,normed=True)
y1 = mlab.normpdf(bins1, x1_mean, x1_sd)
plt.plot(bins1, y1, color="red")

plt.subplot(2,1,2)
n2,bins2,patches2 = plt.hist(data[1],10,normed=True)
y2 = mlab.normpdf(bins2, x2_mean, x2_sd)
plt.plot(bins2, y2, color="red")

#Qb
norm_data0 = stats.norm.fit_loc_scale(data)
def nor(data):
    norm = np.linalg.norm(data)
    return data / norm

fig3 = plt.figure()
ax1 = fig3.add_subplot(3,1,1)
stats.probplot(data[0],plot=ax1)
print"\n"
ax2 = fig3.add_subplot(3,1,3)
stats.probplot(data[1],plot=ax2)
#
#
