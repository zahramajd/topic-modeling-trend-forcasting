import glob
import os
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
import numpy as np
from scipy.stats import pearsonr

#TODO: run 6 scenario
#TODO: MSE
#TODO: debug lr


def read_documents():

    docs_per_year = {}
    path = 'documents'

    for folder in os.listdir(path):
        if(not folder=='.DS_Store'):
            files=glob.glob(path+'/'+folder+'/*.txt')
            docs_per_year[folder]=[]
            for file in files:
                f=open(file, 'r')  
                c =f.read() 
                docs_per_year[folder].append(c)
                f.close() 
    
    return docs_per_year

def topic_discovery(doc):
    # assume that each doc contains keywords
    doc = doc.replace('; ','\n')
    topics = doc.split('\n')

    for topic in topics:
        if( topic == ''):
            topics.remove(topic)

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

def generate_topic_correlation(topic_incidence_matrix, topic):
    topic_correlations = []
    for another_topic in topic_incidence_matrix:
        if not another_topic == topic:
            corr, p = pearsonr(list(topic_incidence_matrix[topic].values()),list(topic_incidence_matrix[another_topic].values()))
            topic_correlations.append((another_topic,corr))
    
    return topic_correlations

def generate_other_topics(topic_correlations, topic, scenario):

    other_topics = []

    if(scenario == 'single_topic'):
        other_topics = []

    if(scenario == 'highest_correlated'):
        tmp = -1
        highest = ()
        for topic in topic_correlations:
            if( topic[1] > tmp):
                tmp = topic[1]
                highest = topic
        
        other_topics.append(highest[0])

    if(scenario == 'highly_correlated'):
        threshold = 0.5
        other_topics = []
        for topic in topic_correlations:
            if(topic[1] > threshold):
                other_topics.append(topic[0])

    if(scenario == 'highest_negatively_correlated'):
        tmp = 1
        highest = ()
        for topic in topic_correlations:
            if( topic[1] < tmp):
                tmp = topic[1]
                highest = topic

        other_topics.append(highest[0])

    if(scenario == 'inversely_correlated'):
        threshold = -0.3
        other_topics = []
        for topic in topic_correlations:
            if(topic[1] < threshold):
                other_topics.append(topic[0])

    if(scenario == 'random'):
        other_topics = list(np.random.choice([t[0] for t in topic_correlations], 5, replace=False))

    return other_topics

def topic_forecast(topic_incidence_matrix, topic, other_topics):

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
        years = []
        years_ = []
        
        for year in topic_year_actual:
            if len(years)>1 :
                clf = SVR(gamma='scale', C=1.0, epsilon=0.2)
                clf.fit(np.array(years), np.array([topic_year_actual[k] for k in years_ if k in topic_year_actual]))
                topic_year_predicted[year] = clf.predict(np.array([[float(year)]]))

            years.append([float(year)])
            years_.append(year)

        return topic_year_predicted

    def ensemble(topic_year_actual):
        topic_year_predicted = {}

        lr_result = linear_regression(topic_year_actual)
        svr_result = support_vector_regression(topic_year_actual)

        for year in lr_result:
            topic_year_predicted[year] = (lr_result[year] + svr_result[year]) / 2

        return topic_year_predicted
    
    topic_year_predicted = ensemble(topic_incidence_matrix[topic])
    plot_topic_year(topic_incidence_matrix[topic],topic_year_predicted)
    
    return

topic='Lung_cancer'

docs_per_year = read_documents()

topic_incidence_matrix, topics, topic_incidence_matrix = generate_topic_incidence_matrix(docs_per_year)

topic_correlations = generate_topic_correlation(topic_incidence_matrix, topic)

other_topics = generate_other_topics(topic_correlations, topic, scenario='single_topic')

topic_forecast(topic_incidence_matrix, topic, other_topics)

plt.show()