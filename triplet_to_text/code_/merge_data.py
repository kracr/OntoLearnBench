#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
from sklearn.model_selection import train_test_split


# Below function takes a list of csv files and merge them to give combined csv files

# In[19]:


def clean_df(df):
    # print(df.columns)
    cols = ['prefix','Unnamed: 0','Unnamed: 0.1']
    for col in cols:
        if col in df.columns:
            df = df.drop(col,axis=1)
    return df


# In[20]:


def read_data_files(files):
    df_list = []
    for file in files:
        df_list.append(clean_df(pd.read_csv(file).dropna()))
#     for df in df_list:
#         display(df.head())
    
    combined_df = pd.concat(df_list)
    combined_df.to_csv('../data/all_merged/combined.csv',index = False)
    combined_df = pd.read_csv('../data/all_merged/combined.csv')
    train, test = train_test_split(combined_df, test_size=0.1)
    train.to_csv('../data/all_merged/train.csv',index = False)
    test.to_csv('../data/all_merged/test.csv',index = False)
    
    return combined_df


# In[21]:


files = ['../data/WebNLG/webNLG2020_train.csv','../data/WebNLG/webNLG2020_dev.csv','../data/imdb_bio/final_bio_with_merged_triplets.csv','../data/wikipedia/wiki_summary_with_merged_triplets.csv']
combined_df = read_data_files(files)
print(len(combined_df))
display(combined_df.head())


# In[ ]:




