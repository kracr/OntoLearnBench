# OntoLearnBench
A benchmark for ontology learning and enrichment

# Introduction
# About the Repository
The project repository mainly consists of two main directories, triplet to text translation and axiom generation.

# Dataset 
The primary dataset we have used in our experiments is used in WebNLG 2020 challenge. The dataset consists of triplets and text. These triplets are extracted from DBpedia and mapped text is a grammatically correct sentence in the English language. One of the examples is shown below.
Input text: Alan_Bean | nationality | United_States && Alan_Bean | birthPlace | Wheeler_Texas
Target text (sample 1): American Alan Bean was born in Wheeler Texas.
Target text (sample 2): Alan Bean was an American, who was born in Wheeler, Texas.

To increase the size of the dataset we used text articles from the IMDb dataset and Wikipedia dataset. The IMDb dataset contains multiple sub-datasets such as movies dataset, rating dataset, movies dataset, names dataset, and title principals dataset. We have used the bio part from the names dataset part which is unstructured text in nature. From the Wikipedia dataset, we selected random Wikipedia articles using a python script, and then with the help of ‘wikipedia’ package in python we extracted the summary of the randomly selected article.

But the unstructured text couldn’t be used directly for transformer training therefore we used Coreference resolver and OpenIE tools to get the text in the required format. An example of coreference resolution can be seen below;

Original text: “I like ice cream because it’s sweet”, Amy said.
Coreference resolved text: “Amy likes ice cream because ice cream is sweet”, Amy said.

To extract triplets from coreference resolved text we have used Stanford’s OpenIE tool.
An example of extracted text can be seen below.

Fred Astaire was born in Omaha, Nebraska, to Johanna (Geilus) and Fritz Austerlitz, a brewer.

|- {'subject': 'Fred Astaire', 'relation': 'was born to', 'object': 'Geilus, Johanna'}
|- {'subject': 'Fred Astaire', 'relation': 'was', 'object': ''born, born in Omaha to Johanna'}
|- {'subject': 'Fred Astaire', 'relation': 'was born in', 'object': 'Omaha, Nebraska'} 


# Triplet to text translation: 
This directory contains code and data files. Before we move to 

improvements_to_allennlp_cr.py: This code takes unstructured text as input, for example wiki_summary.csv or bio.csv from Imdb dataset, and produced coreference resolved text. 
The code used for coreference resolution can be found here https://github.com/NeuroSYS-pl/coreference-resolution. All information regarding installation and running can be found on the given link.

wiki_summary.py: This code is used to read random articles from wikipedia, you need to specify the number of the articles to be read from wikipedia and their summary would be stored in a csv file, you can also specify the number of lines to be fethced from each wikipedia article.

##Packages required:
1. urlopen from urllib.request 
2. BeautifulSoup from bs4 
3. webbrowser
4. pandas
5. wikipedia

text_to_triplets.py: This code takes corefercne resolved text as input and then extract the triplets from each sentence using OpenIE tool form standford. All you need is to specify the input file path, eg. path to wiki_summary_resolved.csv, and the name of the output file which will have extracted triplets and their corresponding sentences named as input_text and target_text respectively. 

##Packages required:
1. pandas 
2. spacy
3. StanfordOpenIE from openie  

merge_data.py: This file takes input a list of csv files having triplets and sentences and make a combined csv file from them having input_text and target_text columns only. This combined csv file is then used for transformer training.

Transformer_triplet_to_text.py: This code takes combined.csv file as input and trains the transformer model to convert triplet to text. All information regarding running this file can be found on below link, https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch. 


#How to run axiom generation code

Required library: nltk, wordnet, sklearn, spacy

1. Run AxiomGenerationCode.py in python

2. It will ask to two inputs:
Enter folder to save results: Give folder path in which you have to save the results
Enter path to csv file:  give path of csv file for generation of csv

3. Output: After running the code , the folder should contains 5 text files: 
TermsExtracted.txt, ClassAssertion.txt, ClassnameOfIndividual.txt, Individual.txt, SubclassAxioms.txt 
