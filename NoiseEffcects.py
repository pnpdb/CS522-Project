#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from cleanlab.classification import LearningWithNoisyLabels
from sklearn.ensemble import IsolationForest
from IPython.display import display
import pickle
from Common.DataCenter import data_center
from Common.preprocessor import normalize_preprocessing
from Common.UtilFuncs import print_evaluation, Evaluator, Lab

#
Ev  = Evaluator()

# Text preprocessing
# parameter: original X of training set and test set
# return:  vectorised X of training set and test set
def text_preprocessing(X_train, X_test):
    # preprocessing with traditional NLP methodology
    X_train_normalized = normalize_preprocessing(X_train)
    X_test_normalized  = normalize_preprocessing(X_test)

    # Convert texts to vectors
    vectorizer   = TfidfVectorizer(ngram_range=(1,2))
    X_train_vec  = vectorizer.fit_transform(X_train_normalized)
    X_test_vec   = vectorizer.transform(X_test_normalized)
    return X_train_vec, X_test_vec

# One-hot encoding, convert the labels to vectors (4 x 1) each
# parameter: original y of training set, original y of test set
# return:  encoded y of training set and test set
def one_hot_encoding(y_train, y_test):
    mlb          = MultiLabelBinarizer()
    y_train_vec  = mlb.fit_transform(map(str, y_train))
    y_test_vec   = mlb.transform(map(str, y_test))
    return y_train_vec, y_test_vec

# Run SVM
# parameter:  vectorised X and encoded y of training set and test set
def run_SVM(X_train_vec, y_train_vec, X_test_vec, y_test_vec):
    # Run SVM - fit and predict
    SVM             = OneVsRestClassifier(LinearSVC(dual=False, class_weight='balanced'), n_jobs=-1)
    SVM.fit(X_train_vec, y_train_vec)
    y_pred          = SVM.predict(X_test_vec)
    return  y_pred

# do an experiment without denoising
# Parameter: training set and test set
# Return evaluation info
def do_experiment(train_df, test_df):
    X_train, y_train = data_center.Xy(train_df)
    X_test, y_test   = data_center.Xy(test_df)

    # Convert texts to vectors
    X_train_vec, X_test_vec = text_preprocessing(X_train, X_test)
    y_train_vec, y_test_vec = one_hot_encoding(y_train, y_test)

    # Run SVM and evaluate the results
    y_pred = run_SVM(X_train_vec, y_train_vec, X_test_vec, y_test_vec)

    # Print the evaluation
    print_evaluation(y_test_vec, y_pred, labels=[0,1,2,3])
    evaluateDF = Ev.evaluate(y_test_vec, y_pred)
    return evaluateDF

# do an experiment denoised by ConfidentLearning
# Parameter: training set and test set
# Return evaluation info, denoising info
def do_experiment_denoised_by_ConfidentLearning(train_df, test_df):
    X_train, y_train = data_center.Xy(train_df)
    X_test, y_test   = data_center.Xy(test_df)

    # Convert texts to vectors
    X_train_vec, X_test_vec = text_preprocessing(X_train, X_test)
    y_train_vec, y_test_vec = one_hot_encoding(y_train, y_test)

    # LearningWithNoisyLabels require the classifier has the entry predict_proba()
    # So, use CalibratedClassifierCV to wrap LinearSVC
    SVM = CalibratedClassifierCV(LinearSVC(dual=False, class_weight='balanced'))
    rp = LearningWithNoisyLabels(clf=SVM, seed=522)
    rp.fit(X_train_vec, np.array(y_train))
    y_pred = rp.predict(X_test_vec)

    # Print the evaluation
    # One hot encoding for print_evaluation()
    _, y_pred = one_hot_encoding(y_train, y_pred)
    print_evaluation(y_test_vec, y_pred, labels=[0,1,2,3])

    evaluateDF = Ev.evaluate(y_test_vec, y_pred)
    return evaluateDF, None

# do an experiment denoised by IsolationForest
# Parameter: training set and test set
# Return evaluation info, denoising info
def do_experiment_denoised_by_IsolationForest(train_df, test_df):
    X_train, y_train = data_center.Xy(train_df)
    X_test, y_test   = data_center.Xy(test_df)

    # Convert texts to vectors
    X_train_vec, X_test_vec = text_preprocessing(X_train, X_test)
    y_train_vec, y_test_vec = one_hot_encoding(y_train, y_test)

    X = X_train_vec

    # IsolationForest
    # n_estimators is the number of trees, try bigger values
    iforest = IsolationForest(n_estimators=100000, max_samples='auto',
                              contamination=0.1, max_features=3,
                              bootstrap=False, n_jobs=-1, random_state=1)

    df = train_df.copy()
    df['label'] = iforest.fit_predict(X)

    # give the anomaly score
    df['scores'] = iforest.decision_function(X)
    df.sort_values(by="scores", inplace=True, ascending=True)

    denoisedDF = df[df.label!=-1]
    X_train, y_train = data_center.Xy(denoisedDF)

    # Convert texts to vectors
    X_train_vec, X_test_vec = text_preprocessing(X_train, X_test)
    y_train_vec, y_test_vec = one_hot_encoding(y_train, y_test)

    # Run SVM and evaluate the results
    y_pred = run_SVM(X_train_vec, y_train_vec, X_test_vec, y_test_vec)

    print_evaluation(y_test_vec, y_pred, labels=[0,1,2,3])
    evaluateDF = Ev.evaluate(y_test_vec, y_pred)

    return evaluateDF, denoisedDF

if __name__ == '__main__':
    # The settings of the noise sources.
    # Each item: source -> (size, distribution)
    noisy_set_sizes0 = {
        'mislabeled' : (8600, None),                   # max size: 15000
        # 'irrelevant' : (8600, [0.25,0.25,0.25,0.25]),  # max size: 34259
        # 'translated' : (8600, "reserve_labels"),       # max size: 5000
    }

    noisy_set_sizes1 = {
        # 'mislabeled' : (8600, None),                   # max size: 15000
        'irrelevant' : (8600, [0.25,0.25,0.25,0.25]),  # max size: 34259
        # 'translated' : (8600, "reserve_labels"),       # max size: 5000
    }

    noisy_set_sizes2 = {
        # 'mislabeled' : (8600, None),                   # max size: 15000
        # 'irrelevant' : (8600, [0.25,0.25,0.25,0.25]),  # max size: 34259
        'translated' : (8600, 0),       # max size: 5000
    }

    noisy_set_sizes3 = {
        # 'mislabeled' : (8600, None),                   # max size: 15000
        # 'irrelevant' : (8600, [0.25,0.25,0.25,0.25]),  # max size: 34259
        'translated' : (8600, 0.75),                   # max size: 5000
    }

    noisy_set_sizes4 = {
        # 'mislabeled' : (8600, None),                   # max size: 15000
        # 'irrelevant' : (8600, [0.25,0.25,0.25,0.25]),  # max size: 34259
        'translated' : (8600, 1),                   # max size: 5000
    }

    # dc = data_center("twitter_sentiment_data_clean.csv", train_size = 20000, test_size = 4000, validation_size = 1000,
    #                  noisy_size = noisy_set_sizes2)
    #
    # print(dc.get_noisy_df(8600).sort_values(by="tweetid", ascending=True))

    # To see the data features via a demo
    # train_df = dc.get_train_with_noisy_df(1000,1000)
    # data_center.print_data(train_df.head(15))

    # dc = data_center("twitter_sentiment_data_clean.csv", train_size = 20000, test_size = 4000, validation_size = 1000,
    #                  noisy_size = noisy_set_sizes3)
    #
    # print(dc.get_noisy_df(8600).sort_values(by="tweetid", ascending=True))
    # To see the data features via a demo
    # train_df = dc.get_train_with_noisy_df(1000,1000)
    # data_center.print_data(train_df.head(15))

    # exit()

    # Choose a experiment without denoising
    # Each item: name -> (funcion, whether choose) note:only the first active one will be used
    experiment_without_denoising = {
        'SVM' : (do_experiment, 1),
    }

    # Choose a experiment with denoising
    # Each item: name -> (funcion, whether choose) note:only the first active one will be used
    # experiment_with_denoising = {
    #     'Confident Learning' : (do_experiment_denoised_by_ConfidentLearning, 1),
    #     'Isolation Forest'   : (do_experiment_denoised_by_IsolationForest,   0),
    # }

    # The training set of each experiment
    origin_train_set_sizes = [2000, 4000, 5000, 8000, 10000, 15000, 20000]
    # origin_train_set_sizes = [5000, 10000, 15000, 20000]
    noisy_train_set_sizes  = [(4000, 1000), (8000, 2000), (12000,3000), (15000, 5000)]
    # noisy_train_set_sizes  = [(1000, 2000)]

    RUN = 1      #1/0:  Run new experiments / Read results made by previous experiments
    if RUN:
        # Run new experiments
        # Initialize the lab, which will run a serial of experiments

        lab = Lab("twitter_sentiment_data_clean.csv", noisy_sources = noisy_set_sizes0, total_train_size = 20000, total_test_size = 4000)
        lab.set_experiment_no_denoising(experiment_without_denoising)

        # Run experiment on original data
        lab.do_batch_experiments(origin_train_set_sizes)

        # Run experiment on noisy data -- mislabed noise
        lab.set_noisy_sources(noisy_set_sizes0)
        lab.do_batch_experiments(noisy_train_set_sizes)

        # Run experiment on noisy data -- irrelevant noise
        lab.set_noisy_sources(noisy_set_sizes1)
        lab.do_batch_experiments(noisy_train_set_sizes)

        # Run experiment on noisy data -- translated noise
        lab.set_noisy_sources(noisy_set_sizes2)
        lab.do_batch_experiments(noisy_train_set_sizes)

        # Run experiment on noisy data -- mislabeled translated noise
        lab.set_noisy_sources(noisy_set_sizes3)
        lab.do_batch_experiments(noisy_train_set_sizes)

        # Run experiment on noisy data -- part-mislabeled translated noise
        lab.set_noisy_sources(noisy_set_sizes4)
        lab.do_batch_experiments(noisy_train_set_sizes)

        # Save the results
        lab.save("noise_effect.pk")
    else:
        # Read evaluations saved by previous experiments
        lab = Lab.load("noise_effect.pk")

    # Show evaluations in a form
    lab.print()

    # Plot the evaluations
    lab.plot()

    # Plot training set size vs. Macro F1
    # x coordinate
    # xValue  = "x['Origin']+x['Noise']"
    # xLabel  = "Training set total size\nnoisy sets: %s" % \
    #           str([str(x[0])+'+'+str(x[1]) for x in noisy_train_set_sizes]).replace("\'","")

    xValue  = "x['Origin']"
    xLabel  = "Training set origin part size\nnoisy sets: %s" % \
              str([str(x[0])+'+'+str(x[1]) for x in noisy_train_set_sizes]).replace("\'","")

    # y coordinate
    yValue  = "y['Macro F1']"

    # Divide experiments into several groups, each will be plotted as a line
    len1 = len(origin_train_set_sizes)
    len2 = len(noisy_train_set_sizes)
    lines = { # each item: name, filter
        'Original Data':       "int((df['Experiment']-1)/%d)==0"%len1,
        'Mislabeled Noise':    "int((df['Experiment']-1-%d)/%d)==0 and df['Experiment']-1-%d>=0"%(len1,len2,len1),
        'Irrelevant Noise':    "int((df['Experiment']-1-%d)/%d)==1"%(len1,len2),
        'Translated Noise(0% mislabeled)':    "int((df['Experiment']-1-%d)/%d)==2"%(len1,len2),
        'Translated Noise(25% mislabeled)':   "int((df['Experiment']-1-%d)/%d)==3"%(len1,len2),
        'Translated Noise(100% mislabeled)':  "int((df['Experiment']-1-%d)/%d)==4"%(len1,len2),
    }

    # Do plot
    lab.Ev.plot(xValue = xValue, yValue = yValue, lines = lines,
                xLabel = xLabel, title = "SVM effected by various noises")

