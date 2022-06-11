#!/usr/bin/env python
# coding: utf-8

# In[71]:


# Import the required modules
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import xgboost as xgb


# In[72]:


# define the iris data set
iris = load_iris()


# In[73]:


# View data as a dataframe (Purely for visual purposes, not required for GBM)
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df


# In[74]:


# Define the number of samples and the number of features
numSamples, numFeatures = iris.data.shape


# In[75]:


# View the number of samples and the number of features
# View the target categories : 0 = setosa, 1 = versicolor, 2 = virginica
print(numSamples)
print(numFeatures)
print(list(iris.target_names))


# In[76]:


# Define the data to train on and the data to test on
# x is our explanatory variables while y is our predictor variable
# We train on 80% of the data and test on 20%
# Shuffle the data 0 times before splitting it
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size = 0.2, random_state = 0)


# In[77]:


# Define train and test by specifying the data and label to use
train = xgb.DMatrix(x_train, label = y_train)
test = xgb.DMatrix(x_test, label = y_test)


# In[78]:


# Define the parameters and epochs for the model
param = {'max_depth' : 4,               # maximum depth of a tree. Default: 6
         'eta' : 0.3,                   # control the learning rate
         'objective' : 'multi:softmax', # specify the learning task : multiclass classification
         'num_class' : 3}               # specify the number of values for the predictor variable
epochs = 10


# In[79]:


# Create the model
model = xgb.train(param, train, epochs)


# In[80]:


# Get the predictions
predictions = model.predict(test)


# In[81]:


# View the predictions and the actual values
for i in zip(predictions, y_test):
    print(f"Prediction, Actual = {i}")


# In[82]:


# View the accuracy of the model
accuracy_score(y_test, predictions)

