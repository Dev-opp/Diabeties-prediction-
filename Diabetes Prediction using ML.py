#!/usr/bin/env python
# coding: utf-8

# Importing the Liabraries

# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# Data Collection and Analysis

# In[2]:


diabetes_dataset = pd.read_csv("diabetes.csv")


# In[3]:


diabetes_dataset.head()


# In[4]:


diabetes_dataset.tail()


# In[5]:


diabetes_dataset.shape


# In[6]:


diabetes_dataset.describe()


# In[10]:


diabetes_dataset['Outcome'].value_counts()


# 0 --> Non Diabetic
# 1 --> Diabetic

# In[14]:


diabetes_dataset.groupby('Outcome').mean()


# separating all the Data and Labels 

# In[25]:


X = diabetes_dataset.drop(columns = 'Outcome', axis = 1)
Y = diabetes_dataset['Outcome']


# In[23]:


X


# In[27]:


Y


# Data Standarization

# In[33]:


scaler = StandardScaler()


# In[34]:


scaler.fit(X)


# In[35]:


standarized_data = scaler.transform(X)


# In[36]:


print(standarized_data)


# In[38]:


X = standarized_data
Y = diabetes_dataset['Outcome']


# In[40]:


print(X)
print(Y)


# Train Test Split 

# In[44]:


X_train, X_test , Y_train , Y_test = train_test_split(X,Y, test_size = 0.2 , stratify =Y , random_state =2)


# In[45]:


print(X.shape , X_train.shape , X_test.shape)


# Training the model

# In[46]:


classifier = svm.SVC(kernel = 'linear')


# In[47]:


#training the support vector machine classifier
classifier.fit(X_train, Y_train)


# Model Evaluation

# Accuracy Score

# In[49]:


X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction,Y_train)


# In[52]:


print('Accuracy_score_of_the_prediction:' ,training_data_accuracy)


# In[53]:


X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Y_test)


# In[54]:


print('Accuracy_score_of_the_prediction :',test_data_accuracy)


# In[60]:


input_data =(4,110,92,0,0,37.6,0.191,30)

#changing the input_data into numppy_array
input_data_as_numpy_array = np.asarray(input_data)


# In[62]:


#reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)


# In[64]:


# Standarize the input_data 
std_data = scaler.transform(input_data_reshaped)


# In[65]:


print(std_data)
prediction = classifier.predict(std_data)
print(prediction)


# In[66]:


if (prediction[0] == 0):
    print('The person is Non-Diabetic')
else :
    print('The person is Diabetic')


# In[ ]:




