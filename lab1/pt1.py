from cmath import sqrt
from typing import final
import numpy as np    # Note you can also import functions as abbreviations
import matplotlib.pyplot as plt 
import csv
from sympy import symbols, cos, diff
import math
import pandas

# data = np.array(np.genfromtxt('SampleDataset.csv', delimiter='')) 
data = pandas.read_csv('pt1.csv') # genfromtxt extracts data as a DataFrame, you will want to convert it into arrays for convenience.

data = data.drop(data.columns[[0]], axis=1)
data1 = data.values  # Extracts only the values of interest


data = np.transpose(data1)  # If you just want the array which contains all x data points, and y data points


vi = np.array(data[0])
vf = np.array(data[1])

inituncertain = np.array(data[2])

finaluncertain = np.array(data[3])

restitution = []
count = 0
for ele in vi:
    float(ele)
    for x in vf:
        float(x)
        res = abs(x/ele)
    restitution.append(res)


plt.bar(range(10), restitution, color ='maroon',
        width = 0.4)
    
totuncertain = []
for ele in vf:
    float(ele)
    
    for x in vf:
        float(x)
        for z in restitution:
            float(z)
            for y in finaluncertain:
                float(y)
                for v in inituncertain:
                    float(v)
                uncertain = math.sqrt((z/ele)**2)*(v**2)+ ((z/x)**2)*(y**2)
    totuncertain.append(uncertain)
n = len(vi)

unweighted_mean = sum(restitution)/n
unweighted_mean_error = math.sqrt((sum((totuncertain[i] - unweighted_mean)**2 for i in range(n)))/(n-1))
standard_error_uw= unweighted_mean_error/ math.sqrt(n)
print(totuncertain)
print(unweighted_mean)
print(unweighted_mean_error)
print(standard_error_uw)


# Using different function, and defining marker and error bar colors, size etc..

# Labels axis

#plt.show()
## The customization options are only limited by your imagination

e_sub_w = []
for i in restitution:
    e_sub_w.append((i/(unweighted_mean**2))/(1/unweighted_mean**2)) 


sigma_sub_e = 1/pow(unweighted_mean**2,0.5)



numerator = 0 #sum((e_sub_j[i] / (e_sub_j_error[i])**2 )for i in range(n))
denominator = 0 #sum(1/ (e_sub_j_error[i])**2 for i in range(n))
for i in range(n):
    numerator += restitution[i] / (totuncertain[i])**2 
    denominator += 1/ (totuncertain[i])**2
weighted_mean_e = numerator / denominator 
    
standard_error_on_weighted_mean_e = sum(1/ totuncertain[i]**2 for i in range(n))**(-1/2)
print(standard_error_on_weighted_mean_e)


# Sets plot size
plt.figure(figsize=(10,6))
# Using different function, and defining marker and error bar colors, size etc..
plt.errorbar(restitution, vi, inituncertain, marker='x', ecolor='black',mec='red', linestyle='None',ms=4, mew=4)
# Labels axis
plt.xlabel('coefficient of restitution')
plt.ylabel('Velocity inital(m$s^{-1}$)')
plt.title('ej vs Vi')

plt.figure(figsize=(10,6))
plt.scatter(weighted_mean_e ,standard_error_on_weighted_mean_e)
plt.show()