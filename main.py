import glob
import os

def read_documents():

    docs_per_yaer = {}
    path = 'documents'

    for folder in os.listdir(path):
        files=glob.glob(path+'/'+folder+'/*.txt')
        docs_per_yaer[folder]=[]
        for file in files:
            f=open(file, 'r')  
            c =f.read() 
            docs_per_yaer[folder].append(c)
            f.close() 
    
    return docs_per_yaer

#TODO: preprocessing
def normalize():
    return

#TODO: binary vector representation
def bag_of_word_doc():
    return

#TODO: topic discovery
def topic_discovery():
    return

#TODO: topic incidence matrix 
def generate_topic_incidence_matrix():
    return

#TODO: temporal topic correlation
def generate_topic_correlation():
    return

#TODO: topic trend forecasting
def topic_forecast():
    return

docs_per_yaer = read_documents()
