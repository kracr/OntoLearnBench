#!/usr/bin/env python
# coding: utf-8

# In[1]:


from urllib.request import urlopen
from bs4 import BeautifulSoup
import webbrowser
import pandas as pd
import wikipedia


# Below method is used to read r

# In[5]:


def random_selector(n):
    summary_list = []
    title_list = []
    while len(summary_list)<n:
        try:
            url = 'https://en.wikipedia.org/wiki/Special:Random'
            page = urlopen(url)
            html = page.read().decode('utf-8')
            soup = BeautifulSoup(html,'html.parser')
#             print('successful')
            title = soup.title.string 
            title_list.append(title)
            print(title)
            summary = wikipedia.summary(title,sentences=10).split('\n')[0]
#             print(summary)
            summary_list.append(summary)
        except: 
            print('error')
    df = pd.DataFrame(list(zip(title_list, summary_list)),columns =['title', 'summary'])
    df.to_csv('../data/wikipedia/wiki_summary.csv',index = False)
    return df


# In[4]:


#n is number of articles
df = random_selector(n=1000)


# In[6]:


print(len(df))
display(df.head())


# In[ ]:


references : https://www.youtube.com/watch?v=0vyLJ-QjNuY&t=905s

