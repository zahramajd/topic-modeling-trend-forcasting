import glob
import os

def read_documents():

    docs_per_year = {}
    path = 'documents'

    for folder in os.listdir(path):
        files=glob.glob(path+'/'+folder+'/*.txt')
        docs_per_year[folder]=[]
        for file in files:
            f=open(file, 'r')  
            c =f.read() 
            docs_per_year[folder].append(c)
            f.close() 
    
    return docs_per_year

#TODO: preprocessing
def normalize():
    return

#TODO: binary vector representation
def bag_of_word_doc():
    return

#TODO: topic discovery
# assume that each doc contains keywords
def topic_discovery(docs_per_year):
    docs_topic_per_year = docs_per_year
    for year in docs_topic_per_year:
        for i in range(len(docs_per_year[year])):
            tokenized = docs_topic_per_year[year][i].split('\n')
            docs_topic_per_year[year][i]= []
            for t in tokenized:
                docs_topic_per_year[year][i].append(t.replace(' ','_'))

    return docs_topic_per_year

#TODO: topic incidence matrix 
def generate_topic_incidence_matrix():

    return

#TODO: temporal topic correlation
def generate_topic_correlation():
    return

#TODO: topic trend forecasting
def topic_forecast():
    return

docs_per_year = read_documents()
docs_topic_per_year = topic_discovery(docs_per_year)
topic_incidence_matrix = generate_topic_incidence_matrix(docs_topic_per_year)