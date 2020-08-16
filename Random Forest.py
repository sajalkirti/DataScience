#!/usr/bin/env python
# coding: utf-8

# In[2]:


#import sklearn data set library
from sklearn import datasets

#Load DataSet
iris=datasets.load_iris()


# In[3]:


print(iris.target_names)


# In[4]:


print(iris.feature_names)


# In[6]:


print(iris.data[0:5])


# In[9]:


print(iris.target)


# In[10]:


import pandas as pd


# In[18]:


dt=pd.DataFrame({'sepal length':iris.data[:,0],
                'sepal Width':iris.data[:,1],
                'petal length':iris.data[:,2],
                'petal width':iris.data[:,3],
                'species':iris.target})


# In[20]:


dt.head()


# In[22]:


#Import Train Test Split 
from sklearn.model_selection import train_test_split
X=dt[['sepal length','sepal Width','petal length','petal width']]


# In[23]:


y=dt[['species']]


# In[25]:


X


# In[29]:


#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier


# In[32]:


#Create Gaussian Classifier
cls=RandomForestClassifier(n_estimators=100)


# In[33]:


cls


# In[35]:


X.size


# In[34]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.3)


# In[38]:


X_train.size


# In[41]:


#Train the model using the Training set y_predict = cls.predict(X_train)
cls.fit(X_train,y_train)


# In[43]:


#predict the data
y_pred=cls.predict(X_test)


# In[44]:


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics


# In[46]:


# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[49]:


y_pred.size


# In[51]:


#r_sqr of the prediction
cls.score(X_test,y_pred)


# In[ ]:





# In[ ]:




