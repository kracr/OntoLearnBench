#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import spacy
from openie import StanfordOpenIE


# In[2]:


# !python3 -m spacy download en


# In[3]:


spacy_eng = spacy.load("en")


# Below function is used to write relation or predicate string in camel case to match the formatting of WebNLG dataset.
# input: is located in
# output: isLocatedIN

# In[4]:


def relation_camel_casing(words):
    s = "".join(word[0].upper() + word[1:].lower() for word in words)
    return s[0].lower() + s[1:]


# Below function is used to merge the objects of the triplets having same relationship or predicate terms
# input: list of triplets dictionary
# 

# In[5]:


def triplet_merge(triplet_list):
    relations = set()
    for triplet in triplet_list:
        relations.add(triplet['relation'])
#     print('total relations are ',relations)
    unique_triplet_list = []
    for relation in relations:
        subject = ''
        object_list = []
        for triplet in triplet_list:
            if relation  == triplet['relation']:
                subject = triplet['subject']
                object_list.append('_'.join(triplet['object'].split()))
        temp_triplet = {}
        temp_triplet['subject'] = subject
        temp_triplet['relation'] = relation
        temp_triplet['object'] = ','.join(object_list)
#         print(temp_triplet)
        unique_triplet_list.append(temp_triplet)
    return unique_triplet_list


# Below funtion takes a sentence as input and gives the triplets present in it. 
# input: "Fred Astaire was born in Omaha, Nebraska, to Johanna (Geilus) and Fritz Austerlitz, a brewer. "
# output: 'Fred_Astaire | wasBornTo | Geilus,Johanna && Fred_Astaire | wasBornIn | Omaha,Nebraska && Fred_Astaire | was | born_in_Omaha_to_Johanna,born.'

# In[6]:


def setence_to_triplet(sen):
    with StanfordOpenIE() as client:
#         print('Text: %s.' % sen)
        partial_triplet_list = []
        triplet_list = client.annotate(sen)
        triplet_list = triplet_merge(triplet_list)
        for triple in triplet_list:
            sub = '_'.join(triple['subject'].split())
            rel = relation_camel_casing(triple['relation'].split())
            obj = triple['object']
            partial_triplet = sub + ' | ' + rel + ' | ' + obj
            partial_triplet_list.append(partial_triplet)
        if len(partial_triplet_list) <= 5:
            final_triplet = ' && '.join(partial_triplet_list) + '.'
#             print(final_triplet)
            #if length of tokenized triplet is more than 140 then drop it
            tokens_in_triplet = [tok.text for tok in spacy_eng.tokenizer(final_triplet)]
            l_token_in_triplet = len(tokens_in_triplet)
#             print(l_token_in_triplet)
            if l_token_in_triplet <=140:
                return final_triplet
            else:
                print(len(final_triplet),final_triplet)
                return
        else: 
            return
# sen = 'Fred Astaire was born in Omaha, Nebraska, to Johanna (Geilus) and Fritz Austerlitz, a brewer.'
# setence_to_triplet(sen)


# Below function takes input a paragraph or text as input and split the sentences, then each sentence is passed to sentence_to_triplet method. 
# input: text
# output: add extracted triplets and their correspoding sentences in lists 
# 

# In[8]:


def text_to_triplet_sentence(text):
    global sentence_list
    global triplet_list
    sents = text.split('.')
    for sent in sents:
        triplet = setence_to_triplet(sent)
        if triplet:
            triplet_list.append(triplet)
            sentence_list.append(sent)


# Below function takes a dataframe as input which is having coreference resolved text second argument is the path where to save the output dataframe.
# input: dataframe, output_file
# output: a dataframe having input_text and target_text columns which are in format similar to WebNLG format.

# In[9]:


def dataframe_formatting(df,output_file):
    global sentence_list
    global triplet_list
    df['resolved_text'].apply(text_to_triplet_sentence)
#     print(sentence_list,triplet_list)
    final_df = pd.DataFrame(list(zip(triplet_list,sentence_list)),columns=['input_text','target_text'])
    final_df.to_csv(output_file,index = False)
    display(final_df.head())
    print(len(final_df))


# In[10]:


sentence_list = []
triplet_list = []
df_bio = pd.read_csv('../data/imdb_bio/resolved_bio.csv')
display(df_bio.head())
output_file = '../data/imdb_bio/final_bio_with_merged_triplets.csv'
dataframe_formatting(df_bio,output_file)


# In[11]:


sentence_list = []
triplet_list = []
df_wiki = pd.read_csv('../data/wikipedia/wiki_summary_resolved.csv')
display(df_wiki.head())
output_file = '../data/wikipedia/wiki_summary_with_merged_triplets.csv'
dataframe_formatting(df_wiki,output_file)


# #references 
# #1.https://www.tutorialspoint.com/camelcase-in-python
