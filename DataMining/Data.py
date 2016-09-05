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

plot_Euc_lat=[]
plot_Euc_long=[]
plot_Euc_lat_all=[]
plot_Euc_long_all=[]

def min_plot(x,y,Obj):
    for i in range(10):
        y.append(Obj[i][2])
        x.append(Obj[i][1])

def all_plot(x,y,Obj):
    for i in range(len(Obj)):
        y.append(Obj[i][2])
        x.append(Obj[i][1])

min_plot(plot_Euc_lat,plot_Euc_long,Eucledian)  
all_plot(plot_Euc_lat_all,plot_Euc_long_all,Eucledian) 

#==============================================================================
# plt.xlim([25,45])
fig = plt.figure()
ax1 = fig.add_subplot(3,2,1)
ax1.scatter(plot_Euc_lat_all,plot_Euc_long_all)
ax1.scatter(plot_Euc_lat,plot_Euc_long,color="yellow")
ax1.scatter(latitude_mean,longitude_mean,color="red")
#==============================================================================


#print spatial.distance.euclidean(latitude,longitude)
#plt.plot(Eucledian)
Minkowski=dist(latitude,longitude,mean,3)
print ("Minkowski")
print Minkowski
Minkowski.sort()
plot_min_lat=[]
plot_min_long=[]
plot_min_lat_all=[]
plot_min_long_all=[]

min_plot(plot_min_lat,plot_min_long,Minkowski)
all_plot(plot_min_lat_all,plot_min_long_all,Minkowski)

#==============================================================================
# #=Plot=============================================================================
# plt.xlim([35,45])
# plt.scatter(plot_min_lat_all,plot_min_long_all)
# plt.scatter(plot_min_lat,plot_min_long,color="yellow")
# plt.scatter(latitude_mean,longitude_mean,color="red")
# #==============================================================================
#==============================================================================



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
        result.append((max,x[i],y[i]))
    return result

print "Chebb"
Chebyshev = Chebyshev_dist(latitude,longitude,mean)
Chebyshev.sort()
#print Chebyshev
#print spatial.distance.chebyshev(latitude,longitude)
plot_Cheb_lat=[]
plot_Cheb_long=[]
plot_Cheb_lat_all=[]
plot_Cheb_long_all=[]

min_plot(plot_Cheb_lat,plot_Cheb_long,Chebyshev)
all_plot(plot_Cheb_lat_all,plot_Cheb_long_all,Chebyshev)

#plt.xlim([35,45])
ax2 = fig.add_subplot(3,2,2)
ax2.scatter(plot_Cheb_lat_all,plot_Cheb_long_all)
ax2.scatter(plot_Cheb_lat,plot_Cheb_long,color="yellow")
ax2.scatter(latitude_mean,longitude_mean,color="red")

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
        result.append((1-dist,x[i],y[i]))
    return result

print "Cosine"
Cosine=cosine_dist(latitude,longitude,mean)
Cosine.sort()

plot_Cosine_lat=[]
plot_Cosine_long=[]
plot_Cosine_lat_all=[]
plot_Cosine_long_all=[]

min_plot(plot_Cosine_lat,plot_Cosine_long,Cosine)
all_plot(plot_Cosine_lat_all,plot_Cosine_long_all,Cosine)

#plt.xlim([35,45])
ax3 = fig.add_subplot(3,2,3)
ax3.scatter(plot_Cosine_lat_all,plot_Cosine_long_all)
ax3.scatter(plot_Cosine_lat,plot_Cosine_long,color="yellow")
ax3.scatter(latitude_mean,longitude_mean,color="red")



latitude_matrix = np.array(latitude_mean)
repeated1 = np.repeat(latitude_matrix,1718)
longitude_matrix = np.array(longitude_mean)
repeated2 = np.repeat(longitude_matrix,1718)
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
print "X_Diff"

#print "upper"

diff_xy = np.transpose([x_diff, y_diff])
diff_xy.shape, diff_xy

#print np.transpose(diff_xy[1]).shape
#print inv_mat.shape
#print diff_xy.shape

md = []
for i in range(len(diff_xy)):
    md.append((np.sqrt(np.dot(np.dot(np.transpose(diff_xy[i]),inv_mat),diff_xy[i])),latitude[i],longitude[i]))
print "Answer is"
mahalabonis = md
mahalabonis.sort()
print mahalabonis

plot_md_lat=[]
plot_md_long=[]
plot_md_lat_all=[]
plot_md_long_all=[]

min_plot(plot_md_lat,plot_md_long,mahalabonis)
all_plot(plot_md_lat_all,plot_md_long_all,mahalabonis)

#plt.xlim([35,45])
ax4 = fig.add_subplot(3,2,4)
ax4.scatter(plot_md_lat_all,plot_md_long_all)
ax4.scatter(plot_md_lat,plot_md_long,color="yellow")
ax4.scatter(latitude_mean,longitude_mean,color="red")


#######City Block


def CityBlock(x,y):
    city_block_total = 0
    city=[]
    for j in range(len(x)):
        city_block_total = city_block_total +abs(x[j] - y[j])
        city.append((city_block_total,x[j],y[j]))
    return city
            
cityBlock=CityBlock(latitude,longitude)
cityBlock.sort()

#==============================================================================
# print cityBlock
# print spatial.distance.cityblock(latitude,longitude)
# 
#==============================================================================

plot_cb_lat=[]
plot_cb_long=[]
plot_cb_lat_all=[]
plot_cb_long_all=[]

min_plot(plot_cb_lat,plot_cb_long,cityBlock)
all_plot(plot_cb_lat_all,plot_cb_long_all,cityBlock)

#plt.xlim([35,45])
ax5 = fig.add_subplot(3,2,(5))
ax5.scatter(plot_cb_lat_all,plot_cb_long_all)
ax5.scatter(plot_cb_lat,plot_cb_long,color="yellow")
ax5.scatter(latitude_mean,longitude_mean,color="red")