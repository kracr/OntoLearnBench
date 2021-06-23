#!/usr/bin/env python
# coding: utf-8

# In[1]:


from allennlp.predictors.predictor import Predictor


# ## Instantiate AllenNLP `Predictor`

# 1. Load the same model that is used in the [demo](https://demo.allennlp.org/coreference-resolution) (*don't get alarmed by the warning - we don't need to fine-tune the model to use it*).
# 2. Get the prediction :)

# In[2]:


model_url = 'https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz'
predictor = Predictor.from_path(model_url)


# In[3]:


text = "Eva and Martha didn't want their friend Jenny to feel lonely so they invited her to the party in Las Vegas."
prediction = predictor.predict(document=text)


# ## Coreference resolution with `Allen Institute`

# What we get as a result (`prediction`) is a dictionary as Allen outputs multiple different information at once.   
# The ones that we found to be using the most are:
# 
# |  Key                |  Type              |  Description
# |:--------------------------|:-------------------|:----------------------------------------------------
# | `top_spans`     | `List[List[int]]` | List of `spaCy` token indices pairs representing spans
# | `document` | `List[str]` | Document's tokens (from `spaCy`; but represented as string not Token)
# | `clusters` | `List[List[List[int]]]` | Clusters of spans (represented by token indices pairs)

# In[4]:


# it's our original text (with extra whitespaces as we trivialy just joined tokens with ' ')
' '.join(prediction['document'])


# In[5]:


# and the found clusters - however, they are not easily understood...
prediction['clusters']


# In[6]:


# but that's how it looks after coreference resolution (notice the possessive!)
predictor.coref_resolved(text)


# As Allen's coreference resolution `Predictor` has quite a limited number of functionalities, in order to turn its output to a more readable one, we need to manually write some functions:

# In[7]:


def get_span_words(span, document):
    return ' '.join(document[span[0]:span[1]+1])

def print_clusters(prediction):
    document, clusters = prediction['document'], prediction['clusters']
    for cluster in clusters:
        print(get_span_words(cluster[0], document) + ': ', end='')
        print(f"[{'; '.join([get_span_words(span, document) for span in cluster])}]")


# In[8]:


print_clusters(prediction)

