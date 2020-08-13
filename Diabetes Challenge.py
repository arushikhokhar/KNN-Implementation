#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
dfx = pd.read_csv('Diabetes_XTrain.csv')
dfy = pd.read_csv('Diabetes_YTrain.csv')

X = dfx.values
Y = dfy.values

Y =Y.reshape((-1,))
Y.shape

def dist(x1,x2):
    return np.sqrt(sum((x1-x2)**2))

def knn(X,Y,QueryPoint,k = 5):
    vals = []
    for i in range(X.shape[0]):
        d = dist(QueryPoint,X[i])
        vals.append((d,Y[i]))
    vals= np.array(vals)
    
    vals = sorted(vals)
    vals = vals[:k]
    
    vals= np.array(vals)
    new_vals = np.unique(vals[:,1],return_counts = True)
    
    index = new_vals[1].argmax()
    pred = new_vals[0][index]
    
    return pred

x_test = pd.read_csv('Diabetes_Xtest.csv')
x_test = x_test.values

mylist = []
for i in range(x_test.shape[0]):
    p = knn(X,Y,x_test[i],17)
    mylist.append(p)

ou = pd.DataFrame({'Outcome':np.array(mylist)})
ou.to_csv('Solution.csv',index = False)


# In[ ]:




