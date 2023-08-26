#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # Add this import
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[19]:


df=pd.read_csv("student_performance_new.csv")


# In[20]:


df.describe()


# In[21]:


df.info()


# In[22]:


df.shape


# In[23]:


df.dropna(subset=["Compensatory"], inplace=True)


# In[24]:


df.isna().sum()


# In[25]:


x = df.drop(["STUDENT NAME", "USN", "Result"], axis=1)
y = df["Result"]


# In[26]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)


# In[27]:


clf = DecisionTreeClassifier(random_state=42)

clf.fit(x_train, y_train)

plt = plot_tree(clf, class_names=["pass", "fail"], filled=True)


# In[28]:


y_pred = clf.predict(x_test)

a = accuracy_score(y_test, y_pred)
print("Accuracy:", a)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

x_test


# In[29]:


y_pred = clf.predict(x_test)

a = accuracy_score(y_test, y_pred)
print("Accuracy:", a)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

x_test


# In[34]:


result = clf.predict([[27, 33, 22, 27.3, 12.666667, 1, 7, 10, 10, 20, 1, 32.66667, 13, 14, 1]])
print("Prediction for input:", result)
df


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




