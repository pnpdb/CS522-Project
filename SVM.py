#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from yellowbrick.text import FreqDistVisualizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, hamming_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from Common.DataCenter import data_center

ModelsPerformance = {}
# Creating method for model evaluation
def metricsReport(modelName, test_labels, predictions):
    macro_f1 = f1_score(test_labels, predictions, average='macro')

    #micro_f1 = f1_score(test_labels, predictions, average='micro')
    
    weighted_f1 = f1_score(test_labels, predictions, average='weighted')
    
    macro_precision = precision_score(test_labels, predictions, average='macro')
    
    macro_recall = recall_score(test_labels, predictions, average='macro')   

    # hamLoss = hamming_loss(test_labels, predictions)
    ModelsPerformance[modelName] = (macro_f1, weighted_f1, macro_precision, macro_recall)

def Evaluate_Models(X_train, y_train, X_test, y_test, bSVMOnly = True):
    #Initializing Vectorization of Climate posts
    vectorizer = TfidfVectorizer()
    vectorised_train_documents = vectorizer.fit_transform(X_train)
    vectorised_test_documents = vectorizer.transform(X_test)
    # print("##########################################")
    # print("vectorised train documents:")
    # print(vectorised_train_documents)

    # Using the One vs All concept, I am changing the labels to vectors (4 x 1) each
    mlb = MultiLabelBinarizer()
    train_labels = mlb.fit_transform(map(str, y_train))
    test_labels = mlb.transform(map(str, y_test))

    if(not bSVMOnly):
        #Distribution of most common words
        features = vectorizer.get_feature_names_out()
        visualizer = FreqDistVisualizer(features=features, orient='v')
        visualizer.fit(vectorised_train_documents)
        visualizer.show()

        #first weak classifier
        bagClassifier = OneVsRestClassifier(BaggingClassifier(n_jobs=-1))
        bagClassifier.fit(vectorised_train_documents, train_labels)
        bagPreds = bagClassifier.predict(vectorised_test_documents)
        metricsReport(bagClassifier, test_labels, bagPreds)

        #Model: K-nearest-Neighbors
        knnClf = KNeighborsClassifier()

        knnClf.fit(vectorised_train_documents, train_labels)
        knnPredictions = knnClf.predict(vectorised_test_documents)
        metricsReport(knnClf, test_labels, knnPredictions)

        #Model: Decision Tree
        dtClassifier = DecisionTreeClassifier()
        dtClassifier.fit(vectorised_train_documents, train_labels)
        dtPreds = dtClassifier.predict(vectorised_test_documents)
        metricsReport(dtClassifier, test_labels, dtPreds)

        #Model: Random Forest
        rfClassifier = RandomForestClassifier(n_jobs=-1)
        rfClassifier.fit(vectorised_train_documents, train_labels)
        rfPreds = rfClassifier.predict(vectorised_test_documents)
        metricsReport(rfClassifier, test_labels, rfPreds)

        #Model: Gradient Boosting
        boostClassifier = OneVsRestClassifier(GradientBoostingClassifier())
        boostClassifier.fit(vectorised_train_documents, train_labels)
        boostPreds = boostClassifier.predict(vectorised_test_documents)
        metricsReport(boostClassifier, test_labels, boostPreds)

        #Model: Multinominal Naive Bayes
        nbClassifier = OneVsRestClassifier(MultinomialNB())
        nbClassifier.fit(vectorised_train_documents, train_labels)
        nbPreds = nbClassifier.predict(vectorised_test_documents)
        metricsReport(nbClassifier, test_labels, nbPreds)

    #Model: Linear Support Vector Machine
    svmClassifier = OneVsRestClassifier(LinearSVC(), n_jobs=-1)
    svmClassifier.fit(vectorised_train_documents, train_labels)

    svmPreds = svmClassifier.predict(vectorised_test_documents)
    metricsReport(svmClassifier, test_labels, svmPreds)

    print(svmClassifier, " macro_f1: %.4f , weighted_f1: %.4f, macro_precision: %.4f, macro_recall: %.4f" %
          (ModelsPerformance[svmClassifier][0], 
           ModelsPerformance[svmClassifier][1],
           ModelsPerformance[svmClassifier][2],
           ModelsPerformance[svmClassifier][3]))


if __name__ == '__main__':
    dataset = pd.read_csv("twitter_sentiment_data.csv")
    dataset.head()

    dc = data_center("twitter_sentiment_data.csv", test_size = 0.2, noisy_size = 0.2)

    print("####################################################")
    print("Total data size: ", dc.get_len())
    print("Total train data size: ", dc.get_train_len())
    print("Total test data size: ",  dc.get_test_len())

    X_test, y_test = dc.get_test()
    for size in [400, 800, 1600, 3200, 4000, 8000, 15000, 20000]:
        print("-----------------------------------------------")
        X_train, y_train = dc.get_train(size/dc.get_train_len())
        print("Train data size %.1f%% (%d samples): " % (len(y_train)/dc.get_train_len()*100, len(X_train)))
        Evaluate_Models(X_train, y_train, X_test, y_test)
