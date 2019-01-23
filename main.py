import glob
import os
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import numpy as np

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
def topic_discovery(doc):
    topics = doc.split('\n')
    return topics

def generate_topic_incidence_matrix(docs_per_year):

    topic_incidence_matrix = {}
    topics = set()
    docs_topic_per_year = docs_per_year

    for year in docs_topic_per_year:
        for i in range(len(docs_per_year[year])):
            topics_of_doc = topic_discovery(docs_topic_per_year[year][i])
            docs_topic_per_year[year][i]= []
            for t in topics_of_doc:
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

        topic_incidence_matrix[topic] = dict(sorted(topic_incidence_matrix[topic].items()))

    return docs_topic_per_year, topics, topic_incidence_matrix

#TODO: temporal topic correlation
def generate_topic_correlation():
    return

#TODO: linear regression
#TODO: support vector regression
#TODO: ensemble

def topic_forecast(topic_incidence_matrix):

    def plot_topic_year(topic_year_actual,topic_year_predicted):
        plt.plot(list(topic_year_actual.keys()), list(topic_year_actual.values()), color='red')
        plt.plot(list(topic_year_predicted.keys()), list(topic_year_predicted.values()), color='blue')
        return

    def linear_regression(topic_year_actual):
        topic_year_predicted = {}

        years = []
        years_ = []
        for year in topic_year_actual:

            if len(years)>1 :
                regr = linear_model.LinearRegression()
                regr.fit(np.array(years), np.array([topic_year_actual[k] for k in years_ if k in topic_year_actual]))
                topic_year_predicted[year] = regr.predict(np.array([[float(year)]]))


            years.append([float(year)])
            years_.append(year)

        return topic_year_predicted

    def support_vector_regression(topic_year_actual):
        topic_year_predicted = {}
        return topic_year_predicted

    def ensemble(topic_year_actual):
        topic_year_predicted = {}
        return topic_year_predicted
    
    topic_year_predicted = linear_regression(topic_incidence_matrix['Machine_learning_algorithms'])
    plot_topic_year(topic_incidence_matrix['Machine_learning_algorithms'],topic_year_predicted)
    
    return

docs_per_year = read_documents()
topic_incidence_matrix, topics, topic_incidence_matrix = generate_topic_incidence_matrix(docs_per_year)
topic_forecast(topic_incidence_matrix)
# plt.show()