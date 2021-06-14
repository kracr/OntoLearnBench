#!/usr/bin/env python
# coding: utf-8

# In[11]:


import nltk
from io import BytesIO
import codecs
import math
import nltk
from nltk.grammar import is_nonterminal
from nltk.tokenize import word_tokenize
import re
from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np


# In[12]:


#nlp.max_length = 20000000 #


# In[13]:



def readfile(file):
 fp = codecs.open(file,"rU", encoding="utf8",errors='ignore')
 text = fp.read()
 return text


# In[14]:


def lower_case(tokens):
    normal_tokens = [w.lower() for w in tokens]     #normalised tokens
    return normal_tokens


# In[15]:


def read_data(path):
    df = pd.read_csv(path)
    df['target_text'] = df['target_text'].str.lower()
    return df


# In[16]:


#resultfolder='D:/benchmark/cikmresult/'
#path ='D:/benchmark/final_data/test.csv'

resultfolder = input("Enter folder to save results:")
path=input("Enter path to csv file")


data = read_data(path)
# display(data.head())

#display(data.head())


# In[17]:


import pandas as pd
import random
#display(data['target_text'])


# In[18]:



sent=[]
import re
#sampledata=sampledata.split(" ")  #this is just for sample data
for i in data['target_text']:
#for i in sampledata:
   try: 
    tagged = nltk.pos_tag(nltk.word_tokenize(i))

    spl=re.split(r'((?<!\d)\.(?!\d))', str(i))   # split the sentence with dot except numericals
    sent.append(spl)
    #print(tagged[0])
   except:
    pass


# In[19]:



concat_list = [j for i in sent for j in i]

for element in concat_list:
    if element == '':
        concat_list.remove(element)
    if element == '.':
        concat_list.remove(element)
while("" in concat_list) :
    concat_list.remove("")


# In[20]:


### extracts terms from the sentence

terms=[]
for i in data['target_text']:
    tagged = nltk.pos_tag(nltk.word_tokenize((i)))
    try:
        if(tagged[0][1]=='NNS' or tagged[0][1]=='NN'):     
           terms.append(tagged[0][0]) 
    except:
        pass
file = open(resultfolder+'//TermsExtracted.txt', 'w',encoding="utf-8")
file.write(str(terms))
file.close()    
#print(terms)


# In[ ]:


import nltk
from spacy import displacy
import re
import pandas as pd
import bs4
import requests
import spacy
from spacy import displacy
nlp = spacy.load('en_core_web_sm')

sub = nlp(str(concat_list))

#individual declaration
individual=[]
classes=[]
ind=[]
cl=[]
for ent in sub.ents:
  
   classd= spacy.explain(ent.label_)
   #print(classd)
   x=classd.split(",")
   #print(x[0])
   #print(ent.text,spacy.explain(ent.label_))
   #print("ClassAssertion(",x,")")
   #print(x[0],',',ent.text)
    
    #extract classes individuals
   if(x[0]=='Countries' or x[0]=='Nationalities' or x[0]== 'Monetary values' or x[0]=='Any named language' or x[0]=='Buildings' or x[0]=='Companies' or x[0]=='Non-GPE locations' or x[0]== 'Titles of books'):
     classes.append(ent.text)
   if(x[0]=='People' or x[0]=='Titles' or x[0]=='Numerals' or x[0]=='Measurements' or x[0]=='Objects' or x[0]=='Absolute or relative dates or periods' or x[0]=='Numerals that do not fall under another type'):
        ind.append(ent.text)
        cl.append(x[0])
        
with open(resultfolder+'//classAssertion.txt', 'w',encoding="utf-8") as f:
          
    for i in classes:
         print('Declaration(ClassAssertion('+str(i)+')))', file=f)
         #print(str(i), file=f)

with open(resultfolder+'//Individual.txt', 'w',encoding="utf-8") as f:
          
    for ind1,cl1 in zip(ind,cl):
     print('Declaration(NamedIndividual('+str(ind1)+')))',file=f)
     #print(str(ind1), file=f)

    
with open(resultfolder+'//ClassnameOfIndividual.txt', 'w',encoding="utf-8") as f:      
    for ind1,cl1 in zip(ind,cl):
     print('Declaration(ClassAssertion('+str(cl1)+','+str(ind1)+')))',file=f)
     #print(str(cl1), file=f)

     
#print(classes)

#print("Individual is:",ent.text,", Class is:",x[0])
   



# In[ ]:





# In[71]:


#Pattern wise searching for subclass axioms
  
import re
tokens = nltk.word_tokenize(str(concat_list))

# parts of speech tagging
tagged = nltk.pos_tag(tokens)# Token and Tag

with open(resultfolder+'//SubclassAxioms1.txt', 'w',encoding="utf-8") as f:
 for tag in range(len(tagged)):
   if(tag<len(tagged)-4):
       if(tagged[tag][1]=='NNP' and tagged[tag+1][1]=='VBZ' and tagged[tag+2][1]=='NN' and tagged[tag+3][1]=='IN' and tagged[tag+4][1]=='JJ'):
            a= tagged[tag][0].lstrip('\'')
            print('SubClassOf(',a,tagged[tag+4][0],')',file=f)
       if(tagged[tag][1]=='NNP' and tagged[tag+1][1]=='VBZ' and tagged[tag+2][1]=='NN' and tagged[tag+3][1]=='IN' and tagged[tag+4][1]=='NN'):
            b = tagged[tag][0].lstrip('\'')
            print('SubClassOf(',b,tagged[tag+4][0],')',file=f)
       if(tagged[tag][1]=='NNP' and tagged[tag+1][1]=='VBZ' and tagged[tag+2][1]=='DT' and tagged[tag+3][1]=='NN'):
            c = tagged[tag][0].lstrip('\'')
            print('SubClassOf(',c,tagged[tag+3][0],')',file=f)
       if(tagged[tag][1]=='NNP' and tagged[tag+1][1]=='VBZ' and tagged[tag+2][1]=='DT' and tagged[tag+3][1]=='JJ'):
            d = tagged[tag][0].lstrip('\'')
            print('SubClassOf(',d,tagged[tag+3][0],')',file=f)
       if(tagged[tag][1]=='NNP' and tagged[tag+1][1]=='VBZ' and tagged[tag+2][1]=='PT' and tagged[tag+3][1]=='JJ'):
            e = tagged[tag][0].lstrip('\'')
            print('SubClassOf(',e,tagged[tag+3][0],')',file=f)
       if(tagged[tag][1]=='NN' and tagged[tag+1][1]=='VBZ' and tagged[tag+2][1]=='NNP'):
            f = tagged[tag][0].lstrip('\'')
            print('SubClassOf(',e,tagged[tag+2][0],')',file=f)
       if(tagged[tag][1]=='NNP' and tagged[tag+1][1]=='NNP' and tagged[tag+2][1]=='NN' and tagged[tag+3][1]=='VBZ' and tagged[tag+4][1]=='NNP'):
            g = tagged[tag][0].lstrip('\'')
            print('SubClassOf(',g,tagged[tag+2][0],')',file=f)
       if(tagged[tag][1]=='NNP' and tagged[tag+1][1]=='VBZ' and tagged[tag+2][1]=='NNP' and tagged[tag+3][1]=='IN' and tagged[tag+4][1]=='NNP'):
            h = tagged[tag][0].lstrip('\'')
            print('SubClassOf(',h,tagged[tag+4][0],')',file=f)
                            
           


# In[ ]:





# In[72]:


from nltk.corpus import wordnet as wn


# In[73]:


def get_hypernyms(ent):
    path_list = []
    ent_synset = wn.synsets(ent)   
    if len(ent_synset) == 0:
        return
#     print('total possible senses ',len(ent_synset),'\n',ent_synset,'\n')
#     word_sense(ent_synset)  #just for simplicity going with first sense only
#     print(ent)
    target_ent = ent_synset[0:1][0]
#     print(target_ent)
#     print(target_ent.hypernyms())
    ent_path = target_ent.hypernym_paths()
    #print(ent_path)
    for i,path in enumerate(ent_path):
#         print('hypernym path:',i+1)
        for e in path:
#             print(str(e.name()).split('.')[0],' -->> ',end='')
            path_list.append(str(e.name()).split('.')[0])
#         print('\n')
        break
    return path_list

sclass=[]
for entity in classes:
   path1=(get_hypernyms(entity))
   if(path1!=None):
       #print(path1)
       sclass.append(path1[-2:])
#print(sclass)        
subclass=[]
for x in sclass:
    if x not in subclass:
        subclass.append(x)
#print(subclass)
with open(resultfolder+'//SubclassAxioms1.txt', 'at',encoding="utf-8") as f:      

 try:
    for i in subclass:
        #print(i[0],i[1])
            print('SubClassOf(',i[1],i[0],')',file=f)
 except:
    pass
f.close()
   #print(x[0])
   #print(ent.text,spacy.explain(ent.label_))
   #print("ClassAssertion(",x,")")
   #print(x[0],',',ent.text)
   
     


# In[74]:


#for ine in sent:
 #   #print(ine)
 # for i in ine:
 #   i=(i.split(" "))
 #   for ent in sub.ents:
 #      classd= spacy.explain(ent.label_)
 #      x=classd.split(",")
        
    
   
   #print(x[0])
   #print(ent.text,spacy.explain(ent.label_))
   #print("ClassAssertion(",x,")")
   #print(x[0],',',ent.text)
   
     


# In[76]:




output_file = open(resultfolder+'//SubclassAxioms.txt', 'w',encoding="utf-8")

import hashlib
completed_lines_hash = set()

for line in open(resultfolder+'//SubclassAxioms1.txt', 'r',encoding="utf-8") :
    hashValue = hashlib.md5(line.rstrip().encode('utf-8')).hexdigest()
    
    if hashValue not in completed_lines_hash:
        output_file.write(line)
        completed_lines_hash.add(hashValue)

output_file.close()
import os
os.remove(resultfolder+'//SubclassAxioms1.txt')


# In[ ]:





# In[ ]:




