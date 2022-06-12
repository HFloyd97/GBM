#!/usr/bin/env python
# coding: utf-8

# # Import required modules

# In[289]:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score

from xgboost import plot_tree
import matplotlib.pyplot as plt


# # Import the data set

# In[290]:


# Read in the csv that we will use
data = pd.read_csv('/Users/harryfloyd/Desktop/Breast_cancer_data.csv')
data


# # Split the data into train and test data

# In[291]:


train = data.drop(['diagnosis'], axis = 1)
test = data['diagnosis']


# In[292]:


x_train, x_test, y_train, y_test = train_test_split(train, test, test_size = 0.2, random_state = 7)


# In[293]:


train = xgb.DMatrix(x_train, label = y_train, enable_categorical = True)


# In[294]:


test = xgb.DMatrix(x_test, label = y_test, enable_categorical = True)


# In[295]:


param = {
    'max_depth' : 1,
    'eta' : 0.3,
    'objective' : 'multi:softmax',
    'num_class' : 2
}

epochs = 10


# In[296]:


model = xgb.train(param, train, epochs)


# In[297]:


predictions = model.predict(test)


# In[298]:


# View estimates vs actuals
# for i in zip(y_test,predictions):
#    print(i)


# In[299]:


print(predictions)


# In[300]:


accuracy_score(y_test, predictions)


# In[301]:


# plot single tree
plot_tree(model)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




