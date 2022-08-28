#!/usr/bin/env python
# coding: utf-8

# import libararies

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# Reading File
df = pd.read_csv('shopping_data.csv')
df.head()


# In[3]:


df.shape


# In[4]:


#data  = df[['Annual Income (k$)', 'Spending Score (1-100)']]
#data


# In[5]:


# We need to find relation between Income and Score


# In[9]:


X = df.iloc[:,3:5].values


# In[39]:


X


# In[10]:


from sklearn.cluster import KMeans


# In[16]:


i = 1
wcss = []


# In[17]:


cluster  =  KMeans(n_clusters=i,init='k-means++',random_state=42)
cluster.fit(X)
wcss.append(cluster.inertia_)


# In[22]:


wcss = []
for each in range(1,11):
    #print(each)
    cluster  =  KMeans(n_clusters=each,init='k-means++',random_state=42)
    cluster.fit(X)
    wcss.append(cluster.inertia_)


# In[26]:


xaxis = range(1,11)
yaxis = wcss
plt.plot(xaxis, yaxis)


# In[48]:


# Identify the Number of Clusters
# Find WCSS for range of clusters
# Assume range is 10 clusters


# In[27]:


from sklearn.cluster import KMeans


# In[28]:


wcss=[]


# In[29]:


for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=12345)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)


# In[30]:


wcss


# In[31]:


plt.plot(range(1,11), wcss)
plt.title('The Elbow Plot')
plt.xlabel('Number of clusrter')
plt.ylabel('WCSS')


# In[15]:


# Number of cluster identified is 5


# In[32]:


kmeans = KMeans(n_clusters=5, init='k-means++', random_state=12345)


# In[33]:


y_pred = kmeans.fit_predict(X)


# In[34]:


df


# In[35]:


y_pred


# In[36]:


df['label'] = y_pred


# In[37]:


df.head(5)


# In[36]:


# Split the data
# Train The data
# Predicting
# Accuracy


# In[38]:


# Visualization
plt.scatter(X[y_pred == 0, 0], X[y_pred == 0, 1], s =100, c = 'red', label = 'cluster1')
plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], s =100, c = 'yellow', label = 'cluster2')
plt.scatter(X[y_pred == 2, 0], X[y_pred == 2, 1], s =100, c = 'blue', label = 'cluster3')
plt.scatter(X[y_pred == 3, 0], X[y_pred == 3, 1], s =100, c = 'orange', label = 'cluster4')
plt.scatter(X[y_pred == 4, 0], X[y_pred == 4, 1], s =100, c = 'green', label = 'cluster5')
plt.title('K Means')
plt.xlabel('Income')
plt.ylabel('Spending')
plt.legend()
plt.show()


# In[38]:


### Insights:::


# ## Evaluated by a Bank
#     Red - Low Income Low Spending
#     Blue - High Income Low Spending
#     Yellow - ----
#     If you a product , which is Credit Card --- ? 

# In[ ]:





# In[ ]:





# ## Evaluated by a Shopping Mall

# In[ ]:




