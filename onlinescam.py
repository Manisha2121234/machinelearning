#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re


# In[2]:


df=pd.read_csv("onlinefraud.csv.zip")


# In[3]:


df.shape


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df.describe()


# In[7]:


df.info()


# In[8]:


df.isnull().sum()


# In[9]:


from sklearn.preprocessing import LabelEncoder


# In[10]:


le=LabelEncoder()
le


# In[11]:


df['type'] = le.fit_transform(df['type'])


# In[12]:


df


# In[13]:


df['nameOrig'] =df['nameOrig'].apply(lambda x: int(''.join(re.findall(r'\d+',str(x)))))


# In[14]:


df


# In[15]:


df['nameDest'] =df['nameDest'].apply(lambda x: int(''.join(re.findall(r'\d+',str(x)))))


# In[16]:


df


# In[17]:


df.info()


# In[18]:


from sklearn.preprocessing import StandardScaler


# In[19]:


scaler = StandardScaler()


# In[20]:


x=df.drop(columns=['isFraud'])
x.head()


# In[21]:


y=df[['isFraud']]
y.head()


# In[22]:


from sklearn.model_selection import train_test_split


# In[23]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[24]:


print("x_train shape",x_train.shape)
print("x_test shape",x_test.shape)
print("y_train shape",y_train.shape)
print("y_test shape",y_test.shape)


# In[25]:


x_train_scaler = scaler.fit_transform(x_train)
x_train_scaler


# In[26]:


x_test_scaler = scaler.fit_transform(x_test)
x_test_scaler


# In[27]:


from sklearn.linear_model import LogisticRegression


# In[28]:


LR=LogisticRegression()
LR


# In[29]:


LR.fit(x_train_scaler,y_train)


# In[30]:


y_pred = LR.predict(x_test_scaler)


# In[31]:


print(y_pred)


# In[32]:


from sklearn.metrics import classification_report,accuracy_score


# In[33]:


score=accuracy_score(y_pred,y_test)


# In[34]:


score


# In[35]:


score1=classification_report(y_pred,y_test)


# In[36]:


score1


# In[37]:


from sklearn.tree import DecisionTreeClassifier


# In[38]:


DT=DecisionTreeClassifier()


# In[40]:


DT.fit(x_train_scaler,y_train)


# In[41]:


y_pred_DT=DT.predict(x_test_scaler)


# In[42]:


y_pred_DT


# In[43]:


DT_score=accuracy_score(y_pred_DT,y_test)


# In[44]:


DT_score


# In[45]:


DT_score=classification_report(y_pred_DT,y_test)


# In[46]:


DT_score


# In[47]:


from sklearn.ensemble import RandomForestClassifier


# In[48]:


RF=RandomForestClassifier()


# In[ ]:





# In[ ]:




