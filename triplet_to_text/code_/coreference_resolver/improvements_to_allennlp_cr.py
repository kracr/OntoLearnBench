#!/usr/bin/env python
# coding: utf-8

# In[1]:


from typing import List
from spacy.tokens import Doc, Span
from utils import load_models, print_clusters, print_comparison
import pandas as pd


# In[31]:


predictor, nlp = load_models()


# ## The problem
# AllenNLP coreference resolution models seems to find better clusters - numerous clusters that are usually more accurate than the ones found by Huggingface NeuralCoref model. However, the biggest problem lies in the next step - the step of replacing found mentions with the most meaningfull spans from each clusters (that we call the "heads"). We've found a couple of easy-to-fix problems which seem to lead to errors most often. Our ideas can be summed up as:
# - not resolving coreferences in the clusters that doesn't contain any noun phrases (usually it comes down to the clusters composed only of pronouns),
# - chosing the head of the cluster which is a noun phrase (isn't a pronoun),
# - resolving only the inner span in the case of nested coreferent mentions.
# 
# We show all of our improvements by example - for more details please refer to our [third and last blog post](). If you're interested in problems themselves our [second blog post](https://neurosys.com/article/most-popular-frameworks-for-coreference-resolution/) contains all the definitions and theory.

# ## Original AllenNLP impelemntation of the `replace_corefs` method
# 
# We extract the main "logic" into the separate function that will be used in our every method as we leave the core of AllenNLP's logic untouched. So as for now, we will compare our solutions to the `original_replace_corefs` method implemented in AllenNLP `coref.py` (we've just copied it here explicitly in order to compare with the improved method we propose).

# In[32]:


def core_logic_part(document: Doc, coref: List[int], resolved: List[str], mention_span: Span):
    final_token = document[coref[1]]
    if final_token.tag_ in ["PRP$", "POS"]:
        resolved[coref[0]] = mention_span.text + "'s" + final_token.whitespace_
    else:
        resolved[coref[0]] = mention_span.text + final_token.whitespace_
    for i in range(coref[0] + 1, coref[1] + 1):
        resolved[i] = ""
    return resolved


def original_replace_corefs(document: Doc, clusters: List[List[List[int]]]) -> str:
    resolved = list(tok.text_with_ws for tok in document)

    for cluster in clusters:
        mention_start, mention_end = cluster[0][0], cluster[0][1] + 1
        mention_span = document[mention_start:mention_end]

        for coref in cluster[1:]:
            core_logic_part(document, coref, resolved, mention_span)

    return "".join(resolved)


# ## Improvements
# ### Redundant clusters - lack of a meaningfull mention that could become the head
# We completely ignore (we don't resove them at all) the clusters that doesn't contain any noun phrase.

# In[33]:


def get_span_noun_indices(doc: Doc, cluster: List[List[int]]) -> List[int]:
    spans = [doc[span[0]:span[1]+1] for span in cluster]
    spans_pos = [[token.pos_ for token in span] for span in spans]
    span_noun_indices = [i for i, span_pos in enumerate(spans_pos)
        if any(pos in span_pos for pos in ['NOUN', 'PROPN'])]
    return span_noun_indices

def improved_replace_corefs(document, clusters):
    resolved = list(tok.text_with_ws for tok in document)

    for cluster in clusters:
        noun_indices = get_span_noun_indices(document, cluster)

        if noun_indices:  # if there is at least one noun phrase...
            mention_start, mention_end = cluster[0][0], cluster[0][1] + 1
            mention_span = document[mention_start:mention_end]

            for coref in cluster[1:]:
                core_logic_part(document, coref, resolved, mention_span)

    return "".join(resolved)


# **Example**  
# We want to take our code and create a game. Let's remind ourselves how to do that.  
# 
# **Original coreference resolution clusters** (by AllenNLP)   
# We --> We; our; 's; ourselves  
# create --> create; that

# In[5]:


text = "We want to take our code and create a game. Let's remind ourselves how to do that."
clusters = predictor.predict(text)['clusters']
doc = nlp(text)


# In[6]:


print_comparison(original_replace_corefs(doc, clusters), improved_replace_corefs(doc, clusters))


# ### Cataphora problem - choosing the wrong cluster *head*
# We redefine the span that becomes a cluster head. Instead of choosing the first mention in the cluster, we pick the one that is the first **noun phrase in the cluster** - we define it as the first span that contains a noun.

# In[34]:


def get_cluster_head(doc: Doc, cluster: List[List[int]], noun_indices: List[int]):
    head_idx = noun_indices[0]
    head_start, head_end = cluster[head_idx]
    head_span = doc[head_start:head_end+1]
    return head_span, [head_start, head_end]


def improved_replace_corefs(document, clusters):
    resolved = list(tok.text_with_ws for tok in document)

    for cluster in clusters:
        noun_indices = get_span_noun_indices(document, cluster)

        if noun_indices:
            mention_span, mention = get_cluster_head(document, cluster, noun_indices)

            for coref in cluster:
                if coref != mention:  # we don't replace the head itself
                    core_logic_part(document, coref, resolved, mention_span)

    return "".join(resolved)


# **Example**   
# "He is a great actor!", he said about John Travolta.
# 
# **Original coreference resolution clusters** (by AllenNLP)  
# He --> He; John Travolta

# In[7]:


text = '"He is a great actor!", he said about John Travolta.'
clusters = predictor.predict(text)['clusters']
doc = nlp(text)


# In[9]:


print_comparison(original_replace_corefs(doc, clusters), improved_replace_corefs(doc, clusters))


# ### Nested coreferent mentions
# I the case of nested mentions we choose to resolve the inner span (e.g. for the mention "his dog" the token *his* can be the inner span). That just means we don't want to resolve outer spans.

# In[35]:


def is_containing_other_spans(span: List[int], all_spans: List[List[int]]):
    return any([s[0] >= span[0] and s[1] <= span[1] and s != span for s in all_spans])


def improved_replace_corefs(document, clusters):
    resolved = list(tok.text_with_ws for tok in document)
    all_spans = [span for cluster in clusters for span in cluster]  # flattened list of all spans

    for cluster in clusters:
        noun_indices = get_span_noun_indices(document, cluster)

        if noun_indices:
            mention_span, mention = get_cluster_head(document, cluster, noun_indices)

            for coref in cluster:
                if coref != mention and not is_containing_other_spans(coref, all_spans):
                    core_logic_part(document, coref, resolved, mention_span)

    return "".join(resolved)


# **Example**  
# Anna likes Tom. Tom is Anna's brother. Her brother is tall.
# 
# **Original coreference resolution clusters** (by AllenNLP)  
# Tom --> Tom; Tom; Her brother  
# Anna --> Anna; Anna 's; Her

# In[9]:


text = "Fred Astaire was born in Omaha, Nebraska, to Johanna (Geilus) and Fritz Austerlitz, a brewer. Fred entered show business at age 5. He was successful both in vaudeville and on Broadway in partnership with his sister, Adele Astaire. After Adele retired to marry in 1932, Astaire headed to Hollywood. Signed to RKO, he was loaned to MGM to appear in La danza di Venere (1933) before starting work on RKO's Carioca (1933). In the latter film, he began his highly successful partnership with Ginger Rogers, with whom he danced in 9 RKO pictures. During these years, he was also active in recording and radio. On film, Astaire later appeared opposite a number of partners through various studios. After a temporary retirement in 1945-7, during which he opened Fred Astaire Dance Studios, Astaire returned to film to star in more musicals through 1957. He subsequently performed a number of straight dramatic roles in film and TV."
clusters = predictor.predict(text)['clusters']
doc = nlp(text)


# In[ ]:


# Fred Astaire was born in Omaha, Nebraska, to Johanna (Geilus) and Fritz Austerlitz, a brewer. Fred entered show business at age 5. He was successful both in vaudeville and on Broadway in partnership with his sister, Adele Astaire. After Adele retired to marry in 1932, Astaire headed to Hollywood. Signed to RKO, he was loaned to MGM to appear in La danza di Venere (1933) before starting work on RKO's Carioca (1933). In the latter film, he began his highly successful partnership with Ginger Rogers, with whom he danced in 9 RKO pictures. During these years, he was also active in recording and radio. On film, Astaire later appeared opposite a number of partners through various studios. After a temporary retirement in 1945-7, during which he opened Fred Astaire Dance Studios, Astaire returned to film to star in more musicals through 1957. He subsequently performed a number of straight dramatic roles in film and TV.
# alan shepard was born in new hampshire and died in california. he graduated from nwc, m.a. in 1957, was a test pilot and crew member of the apollo 14, which was operated by nasa.


# In[34]:


print('~~~ original text ~~~\n', text ,'\n')
# print_comparison(original_replace_corefs(doc, clusters), improved_replace_corefs(doc, clusters))
print('\n',improved_replace_corefs(doc, clusters))


# In[36]:


def resolve_ref(text):
    global count
    count += 1
    if type(text)==str and len(text) < 1200: 
        print('accepting ',count)
        clusters = predictor.predict(text)['clusters']
        doc = nlp(text) 
        return improved_replace_corefs(doc, clusters)
    else:
#         print('discarding ',count)
        return 


# In[37]:


#read data in csv format
#make a list of coref-resolved text and append it to dataframe
count = 0
def read(df,column,output_file):
    th = 1000
    ll = 0
    ul = ll + th
    resolved_text_list = []
    print(len(df))
#     display(df.head())
    while len(resolved_text_list)<th:
        trimmed_df = df[ll:ul]
        ll = ul
        ul = ul + th
        for text in trimmed_df[column]:
            resolved_text = resolve_ref(text)
            if resolved_text:
                resolved_text_list.append(resolved_text)
            if len(resolved_text_list) >= th:
                break
    resolved_df = pd.DataFrame(list(zip(resolved_text_list)),columns = ['resolved_text'])
    display(resolved_df.head())
    print(len(resolved_df))
    resolved_df.to_csv(output_file,index=False)


# In[ ]:


df = pd.read_csv('../../data/wiki_summary.csv')
column = 'summary'
output_file = '../../data/wiki_summary_resolved.csv'
read(df,column,output_file)


# In[ ]:


df = pd.read_csv('../../data/wiki_summary.csv')
column = 'summary'
output_file = '../../data/wiki_summary_resolved.csv'
read(df,column,output_file)


# Our last version of the `improved_replace_corefs` contains all of ours refinements. And that's it! You can now use it in your project with or without the intersection strategies (see the other Jupyter Notebook file). Good luck!

# In[ ]:


#references 
#1.https://www.tutorialspoint.com/camelcase-in-python

