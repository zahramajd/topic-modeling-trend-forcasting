import glob
import os
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
import numpy as np
from scipy.stats import pearsonr


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

def generate_other_topics(topic_correlations, topic, topic_incidence_matrix, scenario):

    other_topics = []

    if(scenario == 'single_topic'):
        other_topics = []

    if(scenario == 'highest_correlated'):
        tmp = -1
        highest = ()
        for t in topic_correlations:
            if( t[1] > tmp):
                tmp = t[1]
                highest = t
        
        other_topics.append(highest[0])

    if(scenario == 'highly_correlated'):
        threshold = 0.5
        other_topics = []
        for t in topic_correlations:
            if(t[1] > threshold):
                other_topics.append(t[0])

    if(scenario == 'highest_negatively_correlated'):
        tmp = 1
        highest = ()
        for t in topic_correlations:
            if( t[1] < tmp):
                tmp = t[1]
                highest = t

        other_topics.append(highest[0])

    if(scenario == 'inversely_correlated'):
        threshold = -0.3
        other_topics = []
        for t in topic_correlations:
            if(t[1] < threshold):
                other_topics.append(t[0])

    if(scenario == 'random'):
        other_topics = list(np.random.choice([t[0] for t in topic_correlations], 5, replace=False))

    other_topics_per_year = {}
    for year in topic_incidence_matrix[topic]:
        other_topics_per_year[year] = []
        for tp in other_topics:
            other_topics_per_year[year].append(topic_incidence_matrix[tp][year])

    return  other_topics_per_year

def topic_forecast(topic_incidence_matrix, topic, other_topics_per_year, level, alg):

    def plot_topic_year(topic_year_actual,topic_year_predicted):
        plt.plot(list(topic_year_actual.keys()), list(topic_year_actual.values()), color='red')
        plt.plot(list(topic_year_predicted.keys()), list(topic_year_predicted.values()), color='blue')
        return

    def linear_regression(topic_year_actual, other_topics_per_year, level):
        topic_year_predicted = {}
        if level=='single':
            years = []
            years_ = []
            for year in topic_year_actual:
                if len(years)>1 :
                    regr = linear_model.LinearRegression()
                    regr.fit(np.array(years), np.array([topic_year_actual[k] for k in years_ if k in topic_year_actual]))
                    topic_year_predicted[year] = regr.predict(np.array([[float(year)]]))
                    if topic_year_predicted[year] < 0 :
                        topic_year_predicted[year] = 0

                years.append([float(year)])
                years_.append(year)
        
        if level=='multi':
            xs = []
            years_ = []
            for year in topic_year_actual:
                if len(xs)>1 :
                    regr = linear_model.LinearRegression()
                    regr.fit(np.array(xs), np.array([topic_year_actual[k] for k in years_ if k in topic_year_actual]))
                    topic_year_predicted[year] = regr.predict(np.array([other_topics_per_year[year]]))
                    if topic_year_predicted[year] < 0 :
                        topic_year_predicted[year] = 0

                xs.append(other_topics_per_year[year])
                years_.append(year)

        return topic_year_predicted

    def support_vector_regression(topic_year_actual, other_topics_per_year, level):
        topic_year_predicted = {}
        if level=='single':
            years = []
            years_ = []
            for year in topic_year_actual:
                if len(years)>1 :
                    clf = SVR(gamma='scale', C=1.0, epsilon=0.2)
                    clf.fit(np.array(years), np.array([topic_year_actual[k] for k in years_ if k in topic_year_actual]))
                    topic_year_predicted[year] = clf.predict(np.array([[float(year)]]))
                    if topic_year_predicted[year] < 0 :
                        topic_year_predicted[year] = 0

                years.append([float(year)])
                years_.append(year)
        
        if level=='multi':
            xs = []
            years_ = []
            for year in topic_year_actual:
                if len(xs)>1 :
                    clf = SVR(gamma='scale', C=1.0, epsilon=0.2)
                    clf.fit(np.array(xs), np.array([topic_year_actual[k] for k in years_ if k in topic_year_actual]))
                    topic_year_predicted[year] = clf.predict(np.array([other_topics_per_year[year]]))
                    if topic_year_predicted[year] < 0 :
                        topic_year_predicted[year] = 0

                xs.append(other_topics_per_year[year])
                years_.append(year)

        return topic_year_predicted

    def ensemble(topic_year_actual, other_topics_per_year, level):
        topic_year_predicted = {}

        lr_result = linear_regression(topic_year_actual, other_topics_per_year, level)
        svr_result = support_vector_regression(topic_year_actual, other_topics_per_year, level)

        for year in lr_result:
            topic_year_predicted[year] = (lr_result[year] + svr_result[year]) / 2

        return topic_year_predicted
    
    def evaluate(topic_year_actual,topic_year_predicted):
        
        mse = mean_squared_error(list(topic_year_actual.values())[2:], list(topic_year_predicted.values()))
        mae = mean_absolute_error(list(topic_year_actual.values())[2:], list(topic_year_predicted.values()))
        rmse = mse**(1/2)

        return mse, mae, rmse

    if(alg == 'lr'):
        topic_year_predicted = linear_regression(topic_incidence_matrix[topic], other_topics_per_year=other_topics_per_year, level=level)
    if(alg == 'svr'):
        topic_year_predicted = support_vector_regression(topic_incidence_matrix[topic], other_topics_per_year=other_topics_per_year, level=level)
    if(alg == 'en'):
        topic_year_predicted = ensemble(topic_incidence_matrix[topic], other_topics_per_year=other_topics_per_year, level=level)


    mse, mae, rmse = evaluate(topic_incidence_matrix[topic],topic_year_predicted)
    # plot_topic_year(topic_incidence_matrix[topic],topic_year_predicted)

    return mae, rmse

topic='Anthracofibrosis'

scenario='highly_correlated'

docs_per_year = read_documents()

docs_topic_per_year, topics, topic_incidence_matrix = generate_topic_incidence_matrix(docs_per_year)

topic_correlations = generate_topic_correlation(topic_incidence_matrix, topic)

# other_topics_per_year = generate_other_topics(topic_correlations, topic, topic_incidence_matrix, scenario=scenario)

# if(scenario == 'single_topic'):
#     level = 'single'
# else : level = 'multi'

# topic_forecast(topic_incidence_matrix, topic, other_topics_per_year, level, alg='lr')

# plt.show()

scenarios = ['single_topic','highest_correlated','highly_correlated',
'highest_negatively_correlated','inversely_correlated','random']

algs = ['lr','svr','en']
errors_rmse = {}

for alg in algs:
    errors_rmse[alg]=[]
    for s in scenarios:
        other_topics_per_year = generate_other_topics(topic_correlations, topic, topic_incidence_matrix, scenario=s)

        if(s == 'single_topic'):
            level = 'single'
        else : level = 'multi'

        mae, rmse = topic_forecast(topic_incidence_matrix, topic, other_topics_per_year, level, alg)
        errors_rmse[alg].append(rmse)

plt.plot(scenarios, errors_rmse['lr'], color='red')
plt.plot(scenarios, errors_rmse['svr'], color='blue')
plt.plot(scenarios, errors_rmse['en'], color='black')

plt.show()
