
# coding: utf-8

# In[1]:


# Import necessary modules
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np


# In[2]:


#Loading the iris dataset
iris = datasets.load_iris()
X, y = iris['data'], iris['target']
#Let’s introduce some NaNs to the dataset.
X.ravel()[np.random.choice(X.size, 10, replace=False)] = np.nan


# In[3]:


imp = Imputer(missing_values='NaN', strategy='median', axis=0)


# In[4]:


params = {'kernel': 'rbf'}
classifier = SVC(**params)


# In[5]:


# Setup the pipeline steps
steps = [('imputation', imp),
        ('SVC', classifier)]


# In[6]:


# Create the pipeline
pipeline = Pipeline(steps)


# In[7]:


# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)


# In[8]:


# Fit the pipeline to the training set
pipeline.fit(X_train, y_train)


# In[9]:


# Predict the labels of the test set
y_test_pred = pipeline.predict(X_test)


# In[10]:


# Compute metrics
print "#"*30
print "\nClassification report on test dataset\n"
print classification_report(y_test, y_test_pred)
print "#"*30 + "\n"

