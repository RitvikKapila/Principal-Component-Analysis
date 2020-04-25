#!/usr/bin/env python
# coding: utf-8

"""
Author: Ritvik Kapila
"""


# Import Libraries

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig

from mpl_toolkits.mplot3d import Axes3D

x_train = pd.read_csv('Iris.csv')
y_train = x_train['Species']
x_train = x_train.drop(['Id'], axis=1)
x_train = x_train.drop(['Species'], axis=1)
# print(x_train, y_train)

# Min-Max Scaling the data

x_train = (x_train - x_train.min())/(x_train.max() - x_train.min())
x_train

# Principle Component Analysis

# calculate covariance matrix of centered matrix
V = cov(x_train.T)
# print("Covariance Matrix", V)

# eigendecomposition of covariance matrix
values, vectors = eig(V)
# print("Eigen Vectors", vectors)
# print("Eigen Values", values)

vectors = vectors[:-1,:]
values = values[:-1]
# print("Eigen Vectors", vectors)
# print("Eigen Values", values)

# # project data
P = vectors.dot(x_train.T)
# print(P.T)
x_train = P.T

#Visualization
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)

for i in range(150):
    if i<50:
        ax.scatter(x_train[i][0], x_train[i][1], x_train[i][2], c='r', cmap=plt.cm.Set1, edgecolor='k', s=40)
    elif i<100:
        ax.scatter(x_train[i][0], x_train[i][1], x_train[i][2], c='g', cmap=plt.cm.Set1, edgecolor='k', s=40)
    else:
        ax.scatter(x_train[i][0], x_train[i][1], x_train[i][2], c='b', cmap=plt.cm.Set1, edgecolor='k', s=40)

ax.set_title("First three PCA directions")
ax.set_xlabel("principal component 1")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("principal component 2")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("principal component 3")
ax.w_zaxis.set_ticklabels([])
plt.show()