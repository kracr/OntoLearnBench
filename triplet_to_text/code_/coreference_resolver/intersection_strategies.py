#!/usr/bin/env python
# coding: utf-8

# In[3]:


#pip install neuralcoref --no-binary neuralcoref
#pip install allennlp
#pip install --pre allennlp-models


# In[1]:


from utils import load_models, print_clusters
from utils import IntersectionStrategy, StrictIntersectionStrategy, PartialIntersectionStrategy, FuzzyIntersectionStrategy


# In[2]:


predictor, nlp = load_models()


# ## Models intersection strategies (ensemble)
# 
# We use two models: **AllenNLP** and **Huggingface**. In multiple tests AllenNLP turned out much better - it has better precision and recall (on [Google GAP dataset](https://github.com/google-research-datasets/gap-coreference)), and finds much more clusters at all. That's why we decide to take <ins>AllenNLP answers as a ground truth</ins>.   
# However AllenNLP also makes mistakes - it has about 8% of false positives which we would like to minimize. That's why we propose **several intersections of AllenNLP and Huggingface outputs (an ensemble)** to modify the results and gain more confidence about the final clusters.  
# 
# We propose four intersection strategies:
# - **strict** - we take only those clusters that are identical both in AllenNLP and Huggingface (intersection of clusters)
# - **partial** - we take all of the spans that are identical both in AllenNLP and Huggingface (intersection of spans/mentions)
# - **fuzzy** - we take all of the spans that are the same but also we find spans that overlap (are relating to the same entity but are composed of different number of tokens) and choose the shorter one

# ## Example
# **Text**  
# *In 1311 it was settled on Peter and Lucy for life with remainder to William Everard and his wife Beatrice. Peter had died by 1329 but Lucy lived until 1337 and she was succeeded by William Everard who died in 1343. William's son, Sir Edmund Everard inherited and maintained ownership jointly with his wife Felice until he died in 1370.*  
# 
# **AllenNLP clusters**   
# William Everard --> William Everard; his; William Everard who died in 1343; William's   
# Peter --> Peter; Peter   
# Lucy --> Lucy; Lucy; she   
# William's son, Sir Edmund Everard --> William's son, Sir Edmund Everard; his; he
# 
# **Huggingface clusters**  
# William Everard --> William Everard; his; William Everard; Sir Edmund Everard; his  
# Peter --> Peter; Peter; he  
# Lucy --> Lucy; Lucy; she  
# his wife Beatrice --> his wife Beatrice; his wife Felice
# 
# ### Strategies
# 
# 1. **Strict**  
#     Lucy --> Lucy; Lucy; she
# 
# 2. **Partial**  
#     William Everard --> William Everard; his  
#     Peter --> Peter; Peter  
#     Lucy --> Lucy; Lucy; she
# 
# 3. **Fuzzy**  
#     William Everard --> William Everard; his; William Everard  
#     Peter --> Peter; Peter  
#     Lucy --> Lucy; Lucy; she  
#     Sir Edmund Everard --> Sir Edmund Everard; his  
# 
# 

# In[3]:


text = "Austin Jermaine Wiley (born January 8, 1999) is an American basketball player. He currently plays for the Auburn Tigers in the Southeastern Conference. Wiley attended Spain Park High School in Hoover, Alabama, where he averaged 27.1 points, 12.7 rebounds and 2.9 blocked shots as a junior in 2015-16, before moving to Florida, where he went to Calusa Preparatory School in Miami, Florida, while playing basketball at The Conrad Academy in Orlando."

clusters = predictor.predict(text)['clusters']
doc = nlp(text)


# In[4]:


print("~~~ AllenNLP clusters ~~~")
print_clusters(doc, clusters)
print("\n~~~ Huggingface clusters ~~~")
for cluster in doc._.coref_clusters:
    print(cluster)


# In[5]:


strict = StrictIntersectionStrategy(predictor, nlp)
partial = PartialIntersectionStrategy(predictor, nlp)
fuzzy = FuzzyIntersectionStrategy(predictor, nlp)


# In[11]:


for intersection_strategy in [strict, partial, fuzzy]:
    print(f'\n~~~ {intersection_strategy.__class__.__name__} clusters ~~~')
    print_clusters(doc, intersection_strategy.clusters(text))
    print(IntersectionStrategy.resolve_coreferences(text))


# In[ ]:




