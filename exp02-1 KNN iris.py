#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import packages
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


# In[2]:


# read dataset of Iris
iris = load_iris()
X = iris.data
y = iris.target


# In[3]:


X,y


# In[4]:


iris


# In[8]:


iris.target


# In[9]:



X1,y1 = load_iris(return_X_y=True)





X, X1


# In[10]:


k_range = range(1,40)
k_error = []


# In[11]:


k_range = range(1,40)
k_error = []# iterate set k=1 to k=40 check the errors
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    # parameter cv set the percentage of dataset, there is 5:1 of training VS. Testing
    scores = cross_val_score(knn, X, y, cv=6, scoring='accuracy')
    k_error.append(1-scores.mean())


# In[12]:


# plot the curve, x axis is K, y axis is error
plt.plot(k_range, k_error)
plt.xlabel('Value of K for KNN')
plt.ylabel('Error')
plt.show()


# In[31]:


import matplotlib.pyplot as plt
import numpy as np
from numpy import *
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

k_neighbors = 11
# load dataset of iris
iris = datasets.load_iris()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)


# In[32]:


clf_distance = neighbors.KNeighborsClassifier(n_neighbors = k_neighbors, weights ='distance')
clf_distance.fit(X_train, y_train)


# In[33]:


clf_distance.score(X_test, y_test)


# In[34]:


clf_distance.predict(X_test)


# In[ ]:




