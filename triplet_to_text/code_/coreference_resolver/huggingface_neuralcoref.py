#!/usr/bin/env python
# coding: utf-8

# In[1]:


import spacy
import neuralcoref


# ## Instantiate `NeuralCoref`

# 1. Load a `spaCy` English model - `sm`, `md` or `lg`.
# 2. Add `NeuralCoref` to `spaCy`'s pipe.
# 3. Use `Neuralcoref` as you would usually use a `spaCy` Document - so simple!

# In[2]:


nlp = spacy.load('en_core_web_sm')
neuralcoref.add_to_pipe(nlp)


# In[3]:


text = "Eva and Martha didn't want their friend Jenny to feel lonely so they invited her to the party in Las Vegas."
doc = nlp(text)


# ## Coreference resolution with `Huggingface`

# `Huggingface` has a [demo](https://huggingface.co/coref/) and a very good [README](https://github.com/huggingface/neuralcoref#using-neuralcoref) where all additional CR functionalities are explained.   
# 
# Here's a fragment of Huggingface's documentation - the most commonly used CR methods:
# 
# |  Attribute                |  Type              |  Description
# |:--------------------------|:-------------------|:----------------------------------------------------
# |`doc._.coref_clusters`     |list of `Cluster`   |All the clusters of corefering mentions in the doc
# |`doc._.coref_resolved`     |unicode             |Each corefering mention is replaced by the main mention in the associated cluster.
# |`doc._.coref_scores`       |Dict of Dict        |Scores of the coreference resolution between mentions.
# |`span._.coref_cluster`     |`Cluster`           |Cluster of mentions that corefer with the span
# |`span._.coref_scores`      |Dict                |Scores of the coreference resolution of & span with other mentions (if applicable).
# |`token._.coref_clusters`   |list of `Cluster`   |All the clusters of corefering mentions that contains the token
# 

# In[4]:


# it's our original text
doc  


# In[5]:


# it has two clusters 
clusters = doc._.coref_clusters
clusters


# In[6]:


# and that's how it looks after coreference resolution
doc._.coref_resolved


# In[7]:


# now we'll show how those NeuralCoref methods can be used - we need a Span and a Token
eva_and_martha_span = clusters[0][0]  # the first Span form the first Cluster is 'Eva and Martha'
eva_token = eva_and_martha_span[0]    # the first Token from this Span is 'Eva'


# In[8]:


# we see the score values between our span and all other candidate mentions
eva_and_martha_span._.coref_scores    # the same can be achieved with doc._.coref_scores[eva_and_martha_span]


# In[9]:


# the Cluster to which the Span belongs
eva_and_martha_span._.coref_cluster


# In[10]:


# all Clusters (in the case of nested clusters there can be more than one cluster) to which the Token belongs
eva_token._.coref_clusters


# ## Use it just like a regular `spaCy` document

# The beauty of `Huggingface`'s `NeuralCoref` is that it just adds those coreference resolution functionalities to the `spaCy`'s Document, making it easy to use and enabling access to all its additional functions.

# In[11]:


from spacy import displacy
import pandas as pd


# In[12]:


displacy.render(doc, style="ent")


# In[13]:


df = pd.DataFrame([[token.text, token.pos_, token.tag_]  for token in doc], columns=['token', 'POS', 'TAG'])
df.head()

