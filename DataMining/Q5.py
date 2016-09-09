# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 16:41:06 2016

@author: Team31
       : Rutvij Mehta
       : Sagar Gupta
       : Tanay Pandey
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


""" Read File Operation """

locations_data = pd.read_csv('locations.csv', sep=',')


""" 
(a) Load the file and read ’lat’ and ’long’ columns. 

"""
""" Read titles from Data """

latitude = locations_data.lat
longitude = locations_data.long

#print latitude
#print longitude

"""
(b) Make a 2D plot and label the axes (latitude should be x-axis and longitude
should be y-axis)

"""
plt.figure(figsize=(12,10))
plt.plot(latitude,longitude)
plt.title("Points graph")
plt.xlabel("Latitude")
plt.ylabel("Longitude")


"""

(c) Compute the mean of latitude and longitude values. Consider this point as P.

"""

latitude_mean = np.mean(latitude)
longitude_mean = np.mean(longitude)
P = (latitude_mean,longitude_mean)

"""

(c)-->(a) Compute the distance between P and the 1718 data points us-
ing the following distance measures: Euclidean distance, Mahalanobis dis-
tance,City block metric, Minkowski metric (for p=3), Chebyshev distance and
Cosine distance.


"""


def dist(x,y,mean,p):  
    dist1 = []
    
    for i in range(len(latitude)):
        dist1.append((((abs((x[i]-mean[0])**p) + (abs(y[i]-mean[1])**p))**(1.0/p)),x[i],y[i]))
       # print dist
    return dist1
    
def min_plot(x,y,Obj):
    for i in range(10):
        y.append(Obj[i][2])
        x.append(Obj[i][1])

def all_plot(x,y,Obj):
    for i in range(len(Obj)):
        y.append(Obj[i][2])
        x.append(Obj[i][1])
        
"""

Eucliedian Distance

"""
Eucledian = dist(latitude,longitude,P,2)
Eucledian.sort()


plot_Euc_lat=[]
plot_Euc_long=[]
plot_Euc_lat_all=[]
plot_Euc_long_all=[]

min_plot(plot_Euc_lat,plot_Euc_long,Eucledian)  
all_plot(plot_Euc_lat_all,plot_Euc_long_all,Eucledian) 

#print "\t\t" + str(plot_Euc_lat_all) + str(plot_Euc_long_all)
#==============================================================================
fig = plt.figure()
fig.set_size_inches(10, 10,forward=True)
circle1 = plt.Circle(P, 1.3, color='b', fill=False)
ax1 = fig.add_subplot(1,1,1)
ax1.set_title("Eucledian")
ax1.set_xlabel("Latitude")
ax1.set_ylabel("Longitude")
#ax1.set_xlim([37,37.8])
#ax1.set_ylim([-80,-78])
ax1.add_artist(circle1)
ax1.annotate('X', xy=(latitude_mean,longitude_mean), xytext=P,)
ax1.scatter(plot_Euc_lat_all,plot_Euc_long_all)
ax1.scatter(plot_Euc_lat,plot_Euc_long,color="yellow")
ax1.scatter(latitude_mean,longitude_mean,color="red")
#==============================================================================

"""

Mahalanobis Distance

"""
latitude_matrix = np.array(latitude_mean)
longitude_matrix = np.array(longitude_mean)

cov_mat = np.cov(latitude,longitude,rowvar=0)

inv_mat = np.linalg.inv(cov_mat)

repeated1 = np.repeat(latitude_matrix,1718)
x_diff = np.subtract(latitude,repeated1)

repeated2 = np.repeat(longitude_matrix,1718)
y_diff = np.subtract(longitude,repeated2)

diff_xy = np.transpose([x_diff, y_diff])

md = []
for i in range(len(diff_xy)):
    md.append((np.sqrt(np.dot(np.dot(np.transpose(diff_xy[i]),inv_mat),diff_xy[i])),latitude[i],longitude[i]))

mahalabonis = md
mahalabonis.sort()
print mahalabonis

plot_md_lat=[]
plot_md_long=[]
plot_md_lat_all=[]
plot_md_long_all=[]

min_plot(plot_md_lat,plot_md_long,mahalabonis)
all_plot(plot_md_lat_all,plot_md_long_all,mahalabonis)
print plot_md_lat
print plot_md_long


fig1 = plt.figure()
circle3 = plt.Circle(P, 1.3, color='b', fill=False)
fig1.set_size_inches(10, 10,forward=True)
ax4 = fig1.add_subplot(1,1,1)
ax4.set_title("Mahalanobis")
ax4.set_xlabel("Latitude")
ax4.set_ylabel("Longitude")
#ax4.set_xlim([37,38])
#ax4.set_xlim([37,41])
#ax4.set_ylim([-80,-76])
ax4.add_artist(circle3)
ax4.annotate('X', xy=(latitude_mean,longitude_mean), xytext=P,)
ax4.scatter(plot_md_lat_all,plot_md_long_all)
ax4.scatter(plot_md_lat,plot_md_long,color="yellow")
ax4.scatter(latitude_mean,longitude_mean,color="red")


"""

Minkowski

"""
#print spatial.distance.euclidean(latitude,longitude)
#plt.plot(Eucledian)
Minkowski=dist(latitude,longitude,P,3)
Minkowski.sort()
plot_min_lat=[]
plot_min_long=[]
plot_min_lat_all=[]
plot_min_long_all=[]

min_plot(plot_min_lat,plot_min_long,Minkowski)
all_plot(plot_min_lat_all,plot_min_long_all,Minkowski)

fig6 = plt.figure()
fig6.set_size_inches(10, 10,forward=True)
ax6 = fig6.add_subplot(1,1,1)
circle4 = plt.Circle(P, 1.3, color='b', fill=False)
ax6.add_artist(circle4)
ax6.annotate('X', xy=(latitude_mean,longitude_mean), xytext=P,)
##ax6.set_xlim([37,38])
#ax6.set_xlim([37,37.8])
#ax6.set_ylim([-80,-78])
ax6.set_title("Minkowski")
ax6.set_xlabel("Latitude")
ax6.set_ylabel("Longitude")
ax6.scatter(plot_min_lat_all,plot_min_long_all)
ax6.scatter(plot_min_lat,plot_min_long,color="yellow")
ax6.scatter(latitude_mean,longitude_mean,color="red")


"""

ChebyShev

"""

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

Chebyshev = Chebyshev_dist(latitude,longitude,P)
Chebyshev.sort()
plot_Cheb_lat=[]
plot_Cheb_long=[]
plot_Cheb_lat_all=[]
plot_Cheb_long_all=[]

min_plot(plot_Cheb_lat,plot_Cheb_long,Chebyshev)
all_plot(plot_Cheb_lat_all,plot_Cheb_long_all,Chebyshev)

fig2 = plt.figure()
fig2.set_size_inches(10, 10,forward=True)
ax2 = fig2.add_subplot(1,1,1)
#ax2.set_xlim([37,38])
#ax2.set_xlim([37,37.8])
#ax2.set_ylim([-80,-78])
circle5 = plt.Circle(P, 1.3, color='b', fill=False)
ax2.add_artist(circle5)
ax2.annotate('X', xy=(latitude_mean,longitude_mean), xytext=P,)
ax2.set_title("Chebyshev")
ax2.set_xlabel("Latitude")
ax2.set_ylabel("Longitude")
ax2.scatter(plot_Cheb_lat_all,plot_Cheb_long_all)
ax2.scatter(plot_Cheb_lat,plot_Cheb_long,color="yellow")
ax2.scatter(latitude_mean,longitude_mean,color="red")

"""

Cosine

"""


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


Cosine=cosine_dist(latitude,longitude,P)
Cosine.sort()
plot_Cosine_lat=[]
plot_Cosine_long=[]
plot_Cosine_lat_all=[]
plot_Cosine_long_all=[]

min_plot(plot_Cosine_lat,plot_Cosine_long,Cosine)
all_plot(plot_Cosine_lat_all,plot_Cosine_long_all,Cosine)
#plt.xlim([37,38])
fig3 = plt.figure()
circle2 = plt.Circle(P, 1.3, color='b', fill=False)
fig3.set_size_inches(10, 10,forward=True)
ax3 = fig3.add_subplot(1,1,1)
ax3.add_artist(circle2)
ax3.annotate('X', xy=(latitude_mean,longitude_mean), xytext=P,)
ax3.set_title("Cosine")
ax3.set_xlabel("Latitude")
ax3.set_ylabel("Longitude")
#ax3.set_xlim([37,37.8])
#ax3.set_ylim([-80,-78])
ax3.scatter(plot_Cosine_lat_all,plot_Cosine_long_all)
ax3.scatter(plot_Cosine_lat,plot_Cosine_long,color="yellow")
ax3.scatter(latitude_mean,longitude_mean,color="red")

"""

City Block

"""


def CityBlock(x,y):
    city_block_total = 0
    city=[]
    for j in range(len(x)):
        city_block_total = city_block_total +abs(x[j] - y[j])
        city.append((city_block_total,x[j],y[j]))
    return city
            
cityBlock=CityBlock(latitude,longitude)
cityBlock.sort()

plot_cb_lat=[]
plot_cb_long=[]
plot_cb_lat_all=[]
plot_cb_long_all=[]

min_plot(plot_cb_lat,plot_cb_long,cityBlock)
all_plot(plot_cb_lat_all,plot_cb_long_all,cityBlock)


fig5 = plt.figure()
fig5.set_size_inches(10, 10,forward=True)
circle6 = plt.Circle(P, 1.3, color='b', fill=False)
ax5 = fig5.add_subplot(1,1,1)
#ax5.set_xlim([37,38])
#ax5.set_xlim([37,45])
#ax5.set_ylim([-80,-75])
ax5.add_artist(circle6)
ax5.annotate('X', xy=(latitude_mean,longitude_mean), xytext=P,)
ax5.set_title("CityBlock")
ax5.set_xlabel("Latitude")
ax5.set_ylabel("Longitude")
ax5.scatter(plot_cb_lat_all,plot_cb_long_all)
ax5.scatter(plot_cb_lat,plot_cb_long,color="yellow")
ax5.scatter(latitude_mean,longitude_mean,color="red")
