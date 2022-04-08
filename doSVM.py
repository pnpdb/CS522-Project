#!/usr/bin/env python
# coding: utf-8
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.svm import LinearSVC
from Common.DataCenter import data_center

# Text preprocessing
# parameter: original X of training set and test set
# return:  vectorised X of training set and test set
def text_preprocessing(X_train, X_test):
    # Convert texts to vectors
    vectorizer   = TfidfVectorizer(ngram_range=(1,2))
    X_train_vec  = vectorizer.fit_transform(X_train)
    X_test_vec   = vectorizer.transform(X_test)
    return X_train_vec, X_test_vec

# One-hot encoding, convert the labels to vectors (4 x 1) each
# parameter: original y of training set, original y of test set
# return:  encoded y of training set and test set
def one_hot_encoding(y_train, y_test):
    mlb          = MultiLabelBinarizer()
    y_train_vec  = mlb.fit_transform(map(str, y_train))
    y_test_vec   = mlb.transform(map(str, y_test))
    return y_train_vec, y_test_vec

# Run SVM and evaluate the results
# parameter:  vectorised X and encoded y of training set and test set
def evaluate_SVM(X_train_vec, y_train_vec, X_test_vec, y_test_vec):
    # Run SVM - fit and predict
    SVM             = OneVsRestClassifier(LinearSVC(dual=False, class_weight='balanced'), n_jobs=-1)
    SVM.fit(X_train_vec, y_train_vec)
    prediction      = SVM.predict(X_test_vec)

    # Evaluate the results
    macro_f1        = f1_score(y_test_vec, prediction, average='macro')
    weighted_f1     = f1_score(y_test_vec, prediction, average='weighted')
    macro_precision = precision_score(y_test_vec, prediction, average='macro')
    macro_recall    = recall_score(y_test_vec, prediction, average='macro')

    return macro_f1, weighted_f1, macro_precision, macro_recall

# do an experiment
# Parameter: original X,y of training set and test set
def do_experiment(X_train, y_train, X_test, y_test):
    # Convert texts to vectors
    X_train_vec, X_test_vec = text_preprocessing(X_train, X_test)
    y_train_vec, y_test_vec = one_hot_encoding(y_train, y_test)

    # Run SVM and evaluate the results
    macro_f1, weighted_f1, macro_precision, macro_recall = \
        evaluate_SVM(X_train_vec, y_train_vec, X_test_vec, y_test_vec)

    # Show the indicators
    print("macro_f1: %.4f , weighted_f1: %.4f, macro_precision: %.4f, macro_recall: %.4f" %
          (macro_f1, weighted_f1, macro_precision, macro_recall))

# print the distribution of labels
def print_distribution(hint, y):
    df = data_center.df((y, y))
    c = df['sentiment'].value_counts(sort = False)
    l = len(df)
    print("%s: %s" % (hint, ("%.1f%%, "*(len(c)-1)+"%.1f%%") % tuple([x*100/l for x in list(c)])))

if __name__ == '__main__':
    # Load the database and split it into training set, test set, noisy set, validation set
    dc = data_center("twitter_sentiment_data_clean.csv", test_size=4000, noisy_size=3000)

    print("####################################################")
    print("Total data size: ",       dc.get_len())
    print("Total train data size: ", dc.get_train_len())
    print("Total test data size: ",  dc.get_test_len())

    X_train, y_train              = dc.get_train()
    print_distribution("Sentiment distribution of the whole training set", y_train)

    # Get the test set for evaluation
    X_test, y_test = dc.get_test()

    # distribution of training set
    train_distribution = None

    # distribution of external noisy
    external_noisy_distribution = [0.25, 0.25, 0.25, 0.25]

    # add the external noisy data
    dc.add_noisy(noisy_source="irrelevant", distribution = external_noisy_distribution, size = 5000)

    # Run experiments with different training sets, and use the same test set.
    print("-----------------------------------------------")

    for size in [3000, 6000, 7500, 12000, 15000, 22500, 30000]:
        # Get a training set without noisy data
        X_train, y_train = dc.get_train(size, train_distribution)

        print("Training set size: %5d samples" % (len(y_train)))
        print_distribution("Sentiment distribution", y_train)

        # Do an experiment
        do_experiment(X_train, y_train, X_test, y_test)

    print("-----------------------------------------------")
    for size in [(6000, 1500), (12000, 3000), (22500, 7500)]:
        # Get a noisy training set
        X_train, y_train = dc.get_train_with_noisy(size[0], size[1], train_distribution)
        print("Noisy training set size: %d samples (%d original, %d noisy)" % (len(y_train), size[0], size[1]))
        print_distribution("Sentiment distribution", y_train)

        # Do an experiment
        do_experiment(X_train, y_train, X_test, y_test)
