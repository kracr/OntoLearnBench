# OntoLearnBench
A benchmark for ontology learning and enrichment



#How to run axiom generation code

Required library: nltk, wordnet, sklearn, spacy

1. Run AxiomGenerationCode.py in python

2. It will ask to two inputs:
Enter folder to save results: Give folder path in which you have to save the results
Enter path to csv file:  give path of csv file for generation of csv

3. Output: After running the code , the folder should contains 5 text files: 
TermsExtracted.txt, ClassAssertion.txt, ClassnameOfIndividual.txt, Individual.txt, SubclassAxioms.txt 