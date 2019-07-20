#!/usr/bin/env python
# coding: utf-8

# In[6]:


import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt 
import seaborn as sns
import scipy
import sklearn

print('Python: {}'.format(sys.version))
print('Numpy: {}'.format(np.__version__))
print('Pandas: {}'.format(pd.__version__))
print('Matplotlb: {}'.format(matplotlib.__version__))
print('Seaborn: {}'.format(sns.__version__))
print('Scipy: {}'.format(scipy.__version__))
print('Sklearn: {}'.format(sklearn.__version__))


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


data = pd.read_csv('creditcard.csv')


# In[9]:


print(data.columns)


# In[14]:


print(data.head())


# In[13]:


print(data.shape)


# In[15]:


print(data.describe())


# In[17]:


data.hist(figsize=(20,20))
plt.show()


# In[18]:


Fraud = data[data['Class']==1]
Valid = data[data['Class']==0]

outlier_fraction=len(Fraud)/float(len(Valid))


# In[19]:


print(outlier_fraction)


# In[20]:


print("Valid {}".format(len(Valid)))
print("Fraud {}".format(len(Fraud)))


# In[21]:


corrmat=data.corr()
fig=plt.figure(figsize=(12,9))

sns.heatmap(corrmat,vmax=0.8,square=True)
plt.show()


# In[25]:


target='Class'
X=data.drop('Class',axis=1)
y=data[target]

print(X.shape)
print(y.shape)


# In[26]:


from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


# In[27]:


state=1;

classifiers = {
    'Isolation Forest' : IsolationForest(max_samples=len(X),
                                        contamination=outlier_fraction,
                                        random_state=state),
    'Local Outlier Factor' : LocalOutlierFactor(
    n_neighbors=20,
    contamination=outlier_fraction)
}


# In[33]:


plt.figure(figsize=(9, 7))
n_outliers = len(Fraud)


for i, (clf_name, clf) in enumerate(classifiers.items()):
    
    # fit the data and tag outliers
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X)
    
    # Reshape the prediction values to 0 for valid, 1 for fraud. 
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    
    n_errors = (y_pred != y).sum()
    
    # Run classification metrics
    print('{}: {}'.format(clf_name, n_errors))
    print(accuracy_score(y, y_pred))
    print(classification_report(y, y_pred))


# In[ ]:




