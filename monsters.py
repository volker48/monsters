
# coding: utf-8

# In[8]:

import pandas as pd
import seaborn as sns
get_ipython().magic('matplotlib inline')


# # Data analysis
# The first step in a machine learning project is to load the data and get a sense for what we are looking at.

# In[2]:

train = pd.read_csv('data/train.zip')
test = pd.read_csv('data/test.csv.zip')


# ## Training data
# After loading in our training and test data the next step should be printing out the tabular data.

# In[3]:

train


# ## Test data
# Now lets look at the test data.

# In[5]:

test


# In[45]:

features = train.columns[1:-1]
numeric_features = features[:-2]


# ## Features
# Looking at the data we have a few different features. 
# 
# ### Numerical Features
# `bone_length`, `rotting_flesh`, `hair_length`, and `has_soul` are all continuous numerical variables.
# 
# ### Categorical Features
# The feature `color` is catagorical meaning it takes on a few discrete types. The numerical features we can use directly, but the categorical feature will need some processing so we can use it with most machine learning models.
# 
# ### Response
# The column `type` is the response, meaning the thing we are going to predict. If we look at the test data we can see taht this column is missing. Thats `type` is what we are trying to predict. We want our model to train on `bone_length`, `rotting_flesh`, `hair_length`, and `has_soul` and `color` and then make predictions as to which type of Monster the current sample is describing.
# 
# ### Graphing
# We can graph these continuous features against the categorical features to try and find patterns in the data to help us with our predictions. Lets look at the distributions of the numeric variables.

# In[140]:

sns.pairplot(train[['bone_length', 'rotting_flesh', 'hair_length', 'has_soul']])


# In[72]:

sns.pairplot(train[['bone_length', 'rotting_flesh', 'hair_length', 'has_soul', 'color']], hue='color')


# In[73]:

sns.pairplot(train[['bone_length', 'rotting_flesh', 'hair_length', 'has_soul', 'type']], hue='type')


# In[84]:

sns.distplot(train.bone_length)


# In[42]:

sns.distplot(train.rotting_flesh)


# In[43]:

sns.distplot(train.hair_length)


# In[44]:

sns.distplot(train.has_soul)


# In[19]:

train[features].corr()


# In[35]:

sns.heatmap(train[features].corr())


# ## Correlation
# It looks like hair_length and has_soul are the most strongly correlated numeric features followed by bone_length and has_soul. Lets plot some of the data.

# In[99]:

sns.lmplot(x='bone_length', y='has_soul', hue='color', col='type', data=train)


# In[14]:

sns.lmplot(x='bone_length', y='hair_length', hue='type', data=train)


# In[12]:

sns.lmplot(x='bone_length', y='has_soul', hue='type', data=train)


# In[15]:

sns.lmplot(x='has_soul', y='bone_length', hue='type', data=train)


# In[75]:

sns.countplot('type', data=train)


# In[77]:

sns.countplot('color', data=train)


# In[91]:

sns.distplot(train[train.type == 'Ghoul']['hair_length'])


# In[92]:

sns.distplot(train[train.type == 'Ghost']['hair_length'])


# In[93]:

train[train.type == 'Ghost']['hair_length'].describe()


# In[94]:

train[train.type == 'Ghoul']['hair_length'].describe()


# In[116]:

sns.factorplot(x='bone_length', y='color', col='type', data=train, kind='violin')


# In[117]:

sns.factorplot(x='hair_length', y='color', col='type', data=train, kind='violin')


# In[115]:

sns.factorplot(x='rotting_flesh', y='color', col='type', data=train, kind='violin')


# In[114]:

sns.factorplot(x='has_soul', y='color', col='type', data=train, kind='violin')


# In[ ]:



