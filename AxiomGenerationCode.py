#!/usr/bin/env python
# coding: utf-8

# In[118]:


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


# In[119]:


#nlp.max_length = 20000000 #


# In[120]:



def readfile(file):
 fp = codecs.open(file,"rU", encoding="utf8",errors='ignore')
 text = fp.read()
 return text


# In[121]:


def lower_case(tokens):
    normal_tokens = [w.lower() for w in tokens]     #normalised tokens
    return normal_tokens


# In[122]:


def read_data(path):
    df = pd.read_csv(path)
    df['target_text'] = df['target_text'].str.lower()
    return df


# In[123]:


#resultfolder='D:/benchmark/cikmresult/'
#path ='D:/benchmark/final_data/test.csv'

resultfolder = input("Enter folder to save results:")
path=input("Enter path to csv file")


data = read_data(path)
# display(data.head())

#display(data.head())


# In[124]:


import pandas as pd
import random
#display(data['target_text'])


# In[125]:



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


# In[126]:



concat_list = [j for i in sent for j in i]

for element in concat_list:
    if element == '':
        concat_list.remove(element)
    if element == '.':
        concat_list.remove(element)
while("" in concat_list) :
    concat_list.remove("")


# In[127]:


### extracts terms from the sentence

terms=[]
for i in data['target_text']:
    tagged = nltk.pos_tag(nltk.word_tokenize(str(i)))
    try:
        if(tagged[0][1]=='NNS' or tagged[0][1]=='NN'):     
           terms.append(tagged[0][0]) 
    except:
        pass
file = open(resultfolder+'//TermsExtracted.txt', 'w',encoding="utf-8")
file.write(str(terms))
file.close()    
#print(terms)


# In[128]:


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
   



# In[129]:


def get_pos(string):
    string = nltk.word_tokenize(string)
    pos_string = nltk.pos_tag(string)
    return pos_string


with open(resultfolder+'//SubclassAxioms.txt', 'w',encoding="utf-8") as f:      

 for i in range(0,len(sent)):
    pos=get_pos(str(sent[i]))
    try:
        if(pos[i][1]=='NNP' and pos[i+1][1]=='VBZ' and pos[i+2][1]=='DT' and pos[i+3][1]=='NN'):
            print(sentee)
            print('SubClassOf(',pos[i][1],pos[i+3][0],')',file=f)

        if(pos[i][1]=='NNP' and pos[i+1][1]=='VBZ' and pos[i+2][1]=='DT' and pos[i+3][1]=='JJ'):
            print(sentee)
            print('SubClassOf(',pos[i][1],pos[i+3][0],')',file=f)

        if(pos[i][1]=='NNP' and pos[i+1][1]=='VBZ' and pos[i+2][1]=='DT' and pos[i+3][1]=='NN'):
            print(sentee)
            print('SubClassOf(',pos[i][1],pos[i+3][0],')',file=f)

        if(pos[i][1]=='NNP' and pos[i+1][1]=='VBZ' and pos[i+2][1]=='PT' and pos[i+3][1]=='NN'):
            print(sentee)
            print('SubClassOf(',pos[i][1],pos[i+3][0],')',file=f)

    except:
        pass
    

    
   
   


# In[130]:


from nltk.corpus import wordnet as wn


# In[131]:


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
with open(resultfolder+'//SubclassAxioms.txt', 'w',encoding="utf-8") as f:      

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
   
     


# In[132]:


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
   
     


# In[ ]:




