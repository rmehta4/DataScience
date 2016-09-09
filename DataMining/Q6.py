# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 18:28:00 2016

@author: Team31
       : Rutvij Mehta
       : Sagar Gupta
       : Tanay Pandey
"""

""" Project Imports """
import numpy as np
import pandas as pd
import matplotlib.mlab as mlab
from scipy import stats
import matplotlib.pyplot as plt


""" Read file Operation """
data = pd.read_csv('hw1q6.csv', sep=',',header=-1)

"""
(a) Load the data, Compute and report the mean, median, standard devia-
tion and the range (i.e. the minimum and maximum) for each variable x1, x2.

"""


"""
For x1

"""

x1_mean = np.mean(data[0])
x1_median = np.median(data[0])
x1_sd=np.std(data[0])
x1_range=(np.max(data[0])-np.min(data[0]))
print "Answer : (a)\n\n"
print "\t\tx1"
print "\tmean " +"\t\t" +str(x1_mean)
print "\tmedian "+"\t\t" + str(x1_median)
print "Standard Deviation "+"\t" + str(x1_sd)
print "\tRange "+"\t\t" + str(x1_range)
print "\n"

"""
For x2

"""

x2_mean = np.mean(data[1])
x2_median = np.median(data[1])
x2_sd=np.std(data[1])
x2_range=(np.max(data[1])-np.min(data[1]))

print "\t\tx2"
print "\tmean " +"\t\t" +str(x2_mean)
print "\tmedian "+"\t\t" + str(x2_median)
print "Standard Deviation "+"\t" + str(x2_sd)
print "\tRange "+"\t\t" + str(x2_range)
print "\n"

"""

(b) Compute the quantiles for each variable. The quantiles of data set are
the 0,25,50,75,and 100 percentiles.

"""

print "Answer : (b)\n\n"
Q1_x1,Q2_x1,Q3_x1,Q4_x1,Q5_x1 = np.percentile(data[0],[0,25,50,75,100])

print "\t\tx1"
print "\tQ1" +"\t\t" + str(Q1_x1)
print "\tQ2" +"\t\t" + str(Q2_x1)
print "\tQ3" +"\t\t" + str(Q3_x1)
print "\tQ4" +"\t\t" + str(Q4_x1)
print "\tQ5" +"\t\t" + str(Q5_x1)
print "\n"


Q1_x2,Q2_x2,Q3_x2,Q4_x2,Q5_x2 = np.percentile(data[1],[0,25,50,75,100])

print "\t\tx2"
print "\tQ1" +"\t\t" + str(Q1_x2)
print "\tQ2" +"\t\t" + str(Q2_x2)
print "\tQ3" +"\t\t" + str(Q3_x2)
print "\tQ4" +"\t\t" + str(Q4_x2)
print "\tQ5" +"\t\t" + str(Q5_x2)
print "\n\n"


"""

(c)Create a histogram for each variable using 10 bins. The scale of the
y-axis should be in terms of density. Also, fitting a density curve to the histogram,
In this case, we can simply use normal distribution.

"""
print "Answer :(c)\n\n"
plt.figure(figsize=(12,10))
plt.subplot(2,1,1)
plt.title('For Variable X1')
n1,bins1,patches1 = plt.hist(data[0],10,normed=True)
y1 = mlab.normpdf(bins1, x1_mean, x1_sd)
plt.plot(bins1, y1, color="red")

plt.subplot(2,1,2)
plt.title('For VariableX2')
n2,bins2,patches2 = plt.hist(data[1],10,normed=True)
y2 = mlab.normpdf(bins2, x2_mean, x2_sd)
plt.plot(bins2, y2, color="red")


"""
(d) Create a quantile-quantile plot (commonly called a QQ plot) for each
variable. Include in your plot a line indicating perfect agreement,i.e. y = x. What
could this QQ plot be used for? If the data came from a normal distribution, what
will happen when we plot the quantiles of our data against the that of a normal
distribution?
"""

print "Answer :(d)\n\n"
fig_QQPlot = plt.figure()
fig_QQPlot.set_size_inches(12, 10,forward=True)
fig_QQPlot.savefig('Q6_d.png', dpi=100)
ax1 = fig_QQPlot.add_subplot(3,1,1)
stats.probplot(data[0],plot=ax1)
print"\n"
ax2 = fig_QQPlot.add_subplot(3,1,3)
stats.probplot(data[1],plot=ax2)
#
#
