#!/usr/bin/env python
# coding: utf-8

# # Recognising Handwritten Digits from MNIST Dataset Using KNN

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ### Preparing Data

# In[4]:


df=pd.read_csv("train.csv")
print(df.shape)


# In[5]:


print(df.columns)


# In[6]:


df.head()


# In[10]:


data=df.values              #creating a numpy array out of this data
print(data.shape)
print(type(data))


# In[11]:


X=data[:,1:]
Y=data[:,0]


# In[12]:


print(X)


# In[13]:


print(Y)


# In[14]:


print(X.shape)
print(Y.shape)


# In[15]:


split=int(0.8*X.shape[0])
print(split)


# In[16]:


X_train=X[:split,:]
Y_train=Y[:split]

X_test=X[split:,:]
Y_test=Y[split:]


# In[17]:


print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)


# ### Visualising Training Data

# In[22]:


def drawImg(sample):
    img=sample.reshape((28,28))
    plt.imshow(img,cmap="gray")
    plt.show
drawImg(X_train[3])
print(Y_train[3])
    


# ### Applying KNN

# In[29]:


def dist(x1,x2):
    return np.sqrt(sum((x1-x2)**2))
def knn(X,Y,query_point,k=5):
    vals=[]
    m=X_train.shape[0]
    for i in range(m):
        d=dist(query_point,X[i])                    #query_point and X[i] are vectors. dist function will compute the distance of 
                                                    #query_point from each X vector i,e. from each of the remaining rows. 
                                                    #here, every vector has 784 dimensions.
                
        vals.append((d,Y_train[i]))                 # vals is a list of tuples. Each tuple contains the distance between points
                                                    #and 
                                                    # the label corresponding to that point.
    vals=sorted(vals)                               #arrainging vales in ascending order
    vals=vals[:k]                                   #nearest k points
    new_vals=np.unique(vals[:1],return_counts=True) #returns a tuple where first is an array of all the labels, the second is
                                                    #an array of the count of that label. How many times that label is occuring 
                                                    #in the nearest numbers. (array([0,1]),array([2,3]))
                                                    # array([0,1]) are the lables and array([2,3]) are the frequencies of 
                                                    #occurence of these labels
    index=new_vals[1].argmax()                      #finding out the maximum frequency from the second array
    pred=new_vals[0][index]                         #finding out the label from the first array to which the frequency corresponds
    return pred        


# ### Making Predictions

# In[33]:


pred=knn(X_train,Y_train,X_test[43],k=5)


# In[34]:


drawImg(X_test[43])
print(Y_test[43])


# In[35]:


print(int(pred))


# 

# In[ ]:




