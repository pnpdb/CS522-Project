#!/usr/bin/env python
# coding: utf-8
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.svm import LinearSVC
from Common.DataCenter import data_center

# Text preprocessing
def text_preprocessing(X_train, X_test):
    # Convert texts to vectors
    vectorizer  = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)
    return X_train_vec, X_test_vec

# One-hot encoding, convert the labels to vectors (4 x 1) each
def one_hot_encoding(y_train, y_test):
    mlb = MultiLabelBinarizer()
    y_train_vec  = mlb.fit_transform(map(str, y_train))
    y_test_vec   = mlb.transform(map(str, y_test))
    return y_train_vec, y_test_vec

# Run SVM and evaluate the results
def evaluate_SVM(X_train_vec, y_train_vec, X_test_vec, y_test_vec):
    # Run SVM - fit and predict
    SVM = OneVsRestClassifier(LinearSVC(), n_jobs=-1)
    SVM.fit(X_train_vec, y_train_vec)
    prediction = SVM.predict(X_test_vec)

    # Evaluate the results
    macro_f1 = f1_score(y_test_vec, prediction, average='macro')
    weighted_f1 = f1_score(y_test_vec, prediction, average='weighted')
    macro_precision = precision_score(y_test_vec, prediction, average='macro')
    macro_recall = recall_score(y_test_vec, prediction, average='macro')

    return macro_f1, weighted_f1, macro_precision, macro_recall

# do an experiment
def do_experiment(X_train, y_train, X_test, y_test):
    # Convert texts to vectors
    X_train_vec, X_test_vec = text_preprocessing(X_train, X_test)
    y_train_vec, y_test_vec = one_hot_encoding(y_train, y_test)

    # Run SVM and evaluate the results
    macro_f1, weighted_f1, macro_precision, macro_recall = \
        evaluate_SVM(X_train_vec, y_train_vec, X_test_vec, y_test_vec)

    # Show the indicators
    print(" macro_f1: %.4f , weighted_f1: %.4f, macro_precision: %.4f, macro_recall: %.4f" %
          (macro_f1, weighted_f1, macro_precision, macro_recall))

if __name__ == '__main__':
    # Load the database and split it into training set, test set, noisy set, validation set
    dc = data_center("twitter_sentiment_data.csv", test_size=8000, noisy_size=8000) # sizes represented in absolute values

    print("####################################################")
    print("Total data size: ",       dc.get_len())
    print("Total train data size: ", dc.get_train_len())
    print("Total test data size: ",  dc.get_test_len())

    # Get the test set for evaluation
    X_test, y_test = dc.get_test()

    # Run experiments with different training set, and use the same test set.
    print("-----------------------------------------------")
    for size in [2000, 2500, 4000, 5000, 7500, 10000]:
        # Get training set without noisy data
        X_train, y_train = dc.get_train(size)
        print("Training set size: %d samples (%.1f%%): " % (len(X_train), len(y_train)/dc.get_train_len()*100))

        # Do experiment
        do_experiment(X_train, y_train, X_test, y_test)

    print("-----------------------------------------------")
    for size in [(2000, 500), (4000, 1000), (7500, 2500)]:
        # Get noisy training set
        X_train, y_train = dc.get_train_with_noisy(size[0], size[1])
        print("Noisy training set size: %d samples (%d original, %d noisy)" % (len(y_train), size[0], size[1]))

        # Do experiment
        do_experiment(X_train, y_train, X_test, y_test)
