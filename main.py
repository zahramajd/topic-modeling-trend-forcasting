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
def topic_discovery():
    return

# assume that each doc contains keywords
def generate_topic_incidence_matrix(docs_per_year):

    topic_incidence_matrix = {}
    topics = set()
    docs_topic_per_year = docs_per_year

    for year in docs_topic_per_year:
        for i in range(len(docs_per_year[year])):
            # tokenized is actually an array of the topics of a doc
            tokenized = docs_topic_per_year[year][i].split('\n')
            docs_topic_per_year[year][i]= []
            for t in tokenized:
                topic = t.replace(' ','_')
                docs_topic_per_year[year][i].append(topic)
                
                if not topic in topics:
                    topics.add(topic)
                    topic_incidence_matrix[topic] = {}

                if not year in topic_incidence_matrix[topic]:
                    topic_incidence_matrix[topic][year] = 0

                topic_incidence_matrix[topic][year] = topic_incidence_matrix[topic][year]+1

    for topic in topic_incidence_matrix:
            for year in docs_topic_per_year:
                if not year in topic_incidence_matrix[topic]:
                    topic_incidence_matrix[topic][year] = 0

    return docs_topic_per_year, topics, topic_incidence_matrix

#TODO: temporal topic correlation
def generate_topic_correlation():
    return

#TODO: topic trend forecasting
def topic_forecast():
    return

docs_per_year = read_documents()
topic_incidence_matrix, topics, topic_incidence_matrix = generate_topic_incidence_matrix(docs_per_year)