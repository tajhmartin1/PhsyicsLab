from cmath import sqrt
from typing import final
import numpy as np    
import matplotlib.pyplot as plt 
import csv
from sympy import symbols, cos, diff
import math
import seaborn as sns

import pandas


data1_2 = pandas.read_csv('1.2mm.csv')
data1_2 = data1_2.drop(data1_2.columns[[0]], axis=1)

data1_2 = data1_2.values 

data2_4 = pandas.read_csv('2.4mm.csv')
data2_4 = data2_4.drop(data2_4.columns[[0]], axis=1)

data2_4 = data2_4.values 

data3_7 = pandas.read_csv('3.7mm.csv')
data3_7 = data3_7.drop(data3_7.columns[[0]], axis=1)

data3_7 = data3_7.values 

data4_8= pandas.read_csv('4.8mm.csv')
data4_8 = data4_8.drop(data4_8.columns[[0]], axis=1)
data4_8 = data4_8.values 

data6_0 = pandas.read_csv('6.0mm.csv')
data6_0 = data6_0.drop(data6_0.columns[[0]], axis=1)
data6_0 = data6_0.values 

data1_2 = np.transpose(data1_2)
ax1_2 = data1_2[2]
ax1_2uncertain = data1_2[3]

data2_4= np.transpose(data2_4)
ax2_4= data2_4[2]
ax2_4uncertain = data2_4[3]

data3_7 = np.transpose(data3_7)
ax3_7 = data3_7[2]
ax3_7uncertain = data3_7[3]

data4_8= np.transpose(data4_8)
ax4_8= data4_8[2]
ax4_8uncertain = data4_8[3]
data6_0= np.transpose(data6_0)
ax6_0= data6_0[2]
ax6_0uncertain = data6_0[3]
ax_tot = [ax1_2, ax2_4, ax3_7, ax4_8, ax6_0]
ax_tot_uncertain = [ax1_2uncertain, ax2_4uncertain, ax3_7uncertain, ax4_8uncertain, ax6_0uncertain]
print("uncertain:", ax6_0uncertain)
def leastsquaresUW(xdata,ydata,sigma):
    N = len(xdata)
    D = N*np.sum(xdata**2)-(np.sum(xdata))**2
    a = (N*np.sum(xdata*ydata)-np.sum(xdata)*np.sum(ydata))/D
    b = (np.sum(xdata**2)*np.sum(ydata)-np.sum(xdata)*np.sum(xdata*ydata))/D 
    sigma_a = sigma*(N/D)**.5
    sigma_b = sigma*(np.sum(xdata**2)/D)**.5
    return a, b, sigma_a, sigma_b
# for i in ax_tot:
#     for j in ax_tot_uncertain:
#         print(leastsquaresUW(i,j,1.0))
h = [1.2,2.4,3.7,4.8,6.0]
avg = []
for i in ax_tot:  
    avg.append(sum(i) / len(i))

avg_uncertain = []

for i in ax_tot_uncertain:  
    avg_uncertain.append(sum(i) / len(i))

plt.figure(figsize=(10,6))
# Using different function, and defining marker and error bar colors, size etc..
plt.errorbar(avg, h, avg_uncertain, marker='x', ecolor='black',mec='red', linestyle='None',ms=4, mew=4)
plt.xlabel('height(mm)')
plt.ylabel('unweighted mean ax')
plt.title('height vs unweighted mean ax')
plt.show()
for a in avg:
    for i in h:
        fig = sns.regplot(a,i)
    fig.set(title = "Height vs A", xlabel = 'height (mm)', ylabel = 'acceleration (m/s^2)' )
    fig.errorbar(a,i,yerr=avg_uncertain , fmt = 'none', capsize = 3) 
plt.savefig('h_and_a.png')