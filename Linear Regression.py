#!/usr/bin/env python
# coding: utf-8

# In[71]:


import numpy as np
from sklearn.linear_model import LinearRegression


# In[72]:


x=np.array([5, 15, 25, 35, 45, 55]).reshape((-1,1))


# In[73]:


print(x)


# In[74]:


y=np.array([5, 20, 14, 32, 22, 38])


# In[75]:


print(y)


# In[76]:


model=LinearRegression()


# In[77]:


model


# In[78]:


model=LinearRegression(n_jobs=-1)


# In[79]:


model


# In[80]:


model=LinearRegression(n_jobs=-1).fit(x,y)


# In[81]:


model


# In[82]:


r_sq=model.score(x,y)


# In[83]:


print(r_sq)


# In[84]:


model.intercept_


# In[85]:


model.coef_


# In[86]:


x


# In[87]:


y_pred_cal=model.intercept_+model.coef_*(x)


# In[88]:


y_pred_cal


# In[89]:


y_pred=model.predict(x)


# In[90]:


y_pred


# In[33]:


#MultiLinear Regression


# In[35]:


import numpy as np 
from sklearn.linear_model import LinearRegression


# In[36]:


x = [[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]]
y = [4, 5, 20, 14, 32, 22, 38, 43]


# In[37]:


x,y=np.array(x),np.array(y)


# In[38]:


print(x)


# In[39]:


print(y)


# In[40]:


model=LinearRegression(n_jobs=-1).fit(x,y)


# In[41]:


model


# In[43]:


#get Result
#R2
r_sq=model.score(x,y)
print(r_sq)


# In[44]:


#print intercept
model.intercept_


# In[45]:


#Print Coefficient
model.coef_


# In[46]:


y_pred=model.predict(x)


# In[47]:


#printing predict
print('predicted Response',y_pred,sep='\n')


# In[57]:


print(model.coef_)
print(x)


# In[65]:


y_preCal=model.intercept_+model.coef_.dot(x.transpose())


# In[66]:


y_preCal


# In[ ]:





# In[ ]:




