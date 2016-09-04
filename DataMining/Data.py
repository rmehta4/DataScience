# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 16:41:06 2016

@author: rutvij
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 23:08:33 2016

@author: rutvij
"""
import numpy as np
import pandas as pd
from scipy import spatial
import matplotlib.pyplot as plt

#from numpy.linalg import inv
#np.load()
locations_data = pd.read_csv('locations.csv', sep=',')

# Read titles from dataprint locations_data.dtypes.index 

latitude = locations_data.lat
longitude = locations_data.long

print latitude.dtypes
print locations_data.dtypes.index
print locations_data.shape
#print latitude
#print longitude

#plt.plot(latitude,longitude)

latitude_mean = np.mean(latitude)
longitude_mean = np.mean(longitude)

#print latitude_mean
#print longitude_mean

def dist(x,y,mean,p):  
    dist1 = []
    
    for i in range(len(latitude)):
        
        dist1.append((((abs((x[i]-mean[0])**p) + (abs(y[i]-mean[1])**p))**(1.0/p)),x[i],y[i]))
       # print dist
    return dist1

#print dist(latitude,latitude_mean)
#print dist(longitude,longitude_mean)
mean = (latitude_mean,longitude_mean)
print "mean is"
print mean
Eucledian = dist(latitude,longitude,mean,2)
print "Eucledian"
#sorted(Eucledian, key=lambda data: data[0])
Eucledian.sort()
print "Sorted"
print Eucledian

plot_lat=[]
plot_long=[]
for i in range(10):
   plot_long.append(Eucledian[i][2])
   plot_lat.append(Eucledian[i][1])
   
plot_lat_all=[]
plot_long_all=[]
for i in range(len(Eucledian)):
    plot_long_all.append(Eucledian[i][2])
    plot_lat_all.append(Eucledian[i][1])
plt.xlim([25,45])
plt.scatter(plot_lat_all,plot_long_all)
plt.scatter(plot_lat,plot_long,color="yellow")
plt.scatter(latitude_mean,longitude_mean,color="red")
#print spatial.distance.euclidean(latitude,longitude)
#plt.plot(Eucledian)
Minkowski=dist(latitude,longitude,mean,3)
#print Minkowski
#print spatial.distance.minkowski(latitude,longitude,3)


def Chebyshev_dist(x,y,mean):
    max = -1
    result = []
    diff = 0
    diff1 =0
    for i in range(len(latitude)):
        diff = abs(x[i]-mean[0])
        diff1= abs(y[i]-mean[1])
        if(diff > diff1):
            max = diff
        else:
            max = diff1
        result.append(max)
    return result

print "Chebb"
Chebyshev = Chebyshev_dist(latitude,longitude,mean)
#print Chebyshev
#print spatial.distance.chebyshev(latitude,longitude)



def cosine_dist(x,y,mean):
    dist = 0
    num = 0
    denom1 = 0
    denom2 = 0
    result =[]
    for i in range(len(latitude)):
        num = (x[i]*mean[0]) + (y[i] *mean[1])
        denom1 =  (x[i]**2) + (y[i]**2)
        denom2 =  (mean[0]**2) + (mean[1]**2)
        denom3 = ((denom1)**0.5) * ((denom2)**0.5)
        dist = num / denom3
        result.append(1-dist)
    return result

print "Cosine"
cosine_dist=cosine_dist(latitude,longitude,mean)
#print cosine_dist
#print spatial.distance.cosine(latitude,longitude)
#mat1 = np.mat(latitude_mean)*5
mat1 = np.array(latitude_mean)
repeated1 = np.repeat(mat1,1718)
mat2 = np.array(longitude_mean)
repeated2 = np.repeat(mat2,1718)
#print repeated
trans1 = np.transpose(repeated1)
cov_mat = np.cov(latitude,longitude,rowvar=0)
#print cov_mat
inv_mat = np.linalg.inv(cov_mat)
print "inverse"
#print inv_mat

xy_mean = np.mean(latitude),np.mean(longitude)
x_diff = np.subtract(latitude,repeated1)
#np.array([x_i - xy_mean[0] for x_i in latitude])
y_diff = np.subtract(longitude,repeated2)
#np.array([y_i - xy_mean[1] for y_i in longitude])
print x_diff.shape
#print "upper"

diff_xy = np.transpose([x_diff, y_diff])
diff_xy.shape, diff_xy

#print np.transpose(diff_xy[1]).shape
#print inv_mat.shape
#print diff_xy.shape

md = []
for i in range(len(diff_xy)):
    md.append(np.sqrt(np.dot(np.dot(np.transpose(diff_xy[i]),inv_mat),diff_xy[i])))
print "Answer is"
mahalabonis = md



#######City Block
city_block_total = 0

for j in range(len(latitude)):
    city_block_total = city_block_total +abs(latitude[j] - longitude[j])
    
print city_block_total

print spatial.distance.cityblock(latitude,longitude)
#print np.mat([1,2,3,4,5])-np.mat(np.repeat(2,5))
#x1 = 0
#
#x1_trans = np.transpose(x1)
#x2_trans = np.transpose(x2)
#print spatial.distance.mahalanobis(latitude,longitude,inv_mat)