
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

pd.set_option('display.max_columns', 100)
df = pd.read_csv('https://query.data.world/s/uikepcpffyo2nhig52xxeevdialfl7')
df.head()


# In[3]:


df.shape


# In[4]:


df = df[['Title','Genre','Director','Actors','Plot']]
df.head()


# In[5]:


df.shape


# In[6]:


# discarding the commas
df['Actors'] = df['Actors'].map(lambda x: x.split(',')[:3])

# putting the genres in a list of words
df['Genre'] = df['Genre'].map(lambda x: x.lower().split(','))

df['Director'] = df['Director'].map(lambda x: x.split(' '))

# merging together first and last name for each actor and director
for index, row in df.iterrows():
    row['Actors'] = [x.lower().replace(' ','') for x in row['Actors']]
    row['Director'] = ''.join(row['Director']).lower()


# In[7]:


df.set_index('Title', inplace = True)
df.head()


# In[8]:


df['bag_of_words'] = ''
columns = df.columns
for index, row in df.iterrows():
    words = ''
    for col in columns:
        if col != 'Director':
            words = words + ' '.join(row[col])+ ' '
        else:
            words = words + row[col]+ ' '
    row['bag_of_words'] = words
    
df.drop(columns = [col for col in df.columns if col!= 'bag_of_words'], inplace = True)


# In[9]:


df.head()


# In[10]:


# instantiating and generating the count matrix
count = CountVectorizer()
count_matrix = count.fit_transform(df['bag_of_words'])


indices = pd.Series(df.index)
indices[:5]


# In[11]:


cosine_sim = cosine_similarity(count_matrix, count_matrix)
cosine_sim


# In[12]:


def recommendations(title, cosine_sim = cosine_sim):
    
    recommended_movies = []
    
    idx = indices[indices == title].index[0]

    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)

    top_10_indexes = list(score_series.iloc[1:11].index)
        
    for i in top_10_indexes:
        recommended_movies.append(list(df.index)[i])
        
    return recommended_movies


# In[16]:


recommendations('The Godfather: Part II')


# ###### 
