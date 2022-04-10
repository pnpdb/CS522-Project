#!/usr/bin/env python
# coding: utf-8
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from Common.DataCenter import data_center
from Common.preprocessor import normalize_preprocessing
from Common.UtilFuncs import print_evaluation, print_distribution

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

# do an experiment
# Parameter: original X,y of training set and test set
def do_experiment(X_train, y_train, X_test, y_test):
    # Convert texts to vectors
    X_train_vec, X_test_vec = text_preprocessing(X_train, X_test)
    y_train_vec, y_test_vec = one_hot_encoding(y_train, y_test)

    # Run SVM and evaluate the results
    y_pred = run_SVM(X_train_vec, y_train_vec, X_test_vec, y_test_vec)

    # Print the evaluation
    print_evaluation(y_test_vec, y_pred, labels=[0,1,2,3])

if __name__ == '__main__':
    # The size of the noise sources
    noisy_set_sizes = {
        'mislabeled' : 5000,   # max size: 15000
        'irrelevant' : 5000,   # max size: 34259
        'translated' : 5000,   # max size: 5000
    }

    # Load the database and split it into training set, test set, noisy set, validation set
    dc = data_center("twitter_sentiment_data_clean.csv", train_size = 20000, test_size = 4000, validation_size = 1000,
                     noisy_size = noisy_set_sizes['mislabeled'] if 'mislabeled' in noisy_set_sizes.keys() else 0)

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

    # Prepare noisy data sets
    lstNoisyInfo    = []
    if 'mislabeled' in noisy_set_sizes.keys() and noisy_set_sizes['mislabeled'] > 0:
        lstNoisyInfo.append(("mislabeled",dc.get_noisy_len()))
        print("%d noisy samples of '%s' added" % (dc.get_noisy_len(), 'mislabeled'))

    # add the external noisy data (irrelevant texts)
    if 'irrelevant' in noisy_set_sizes.keys() and noisy_set_sizes['irrelevant'] > 0:
        # distribution of irrelevant noisy
        irrelevant_noisy_distribution = [0.25, 0.25, 0.25, 0.25]    # None, if use the distribution of original set
        added_size = dc.add_noisy(noisy_source="irrelevant", distribution = irrelevant_noisy_distribution,
                                  size = noisy_set_sizes['irrelevant'])
        print("%d noisy samples of '%s' added"  % (added_size, 'irrelevant'))
        lstNoisyInfo.append(("irrelevant",added_size))

    # add the external noisy data (translated texts). use the labels of each noisy data
    if 'translated' in noisy_set_sizes.keys() and noisy_set_sizes['translated'] > 0:
        added_size = dc.add_noisy(noisy_source="translated", distribution = "reserve_labels",
                                  size = noisy_set_sizes['translated'])
        print("%d noisy samples of '%s' added"  % (added_size, 'translated'))
        lstNoisyInfo.append(("translated",added_size))

    print("The total size of noisy data is %d"                % dc.get_noisy_len())

    # Run experiments with different training sets, and use the same test set.
    print("-------------- No noisy training sets ----------")
    for size in [2000, 4000, 5000, 8000, 10000, 15000, 20000]:
        # Get a training set without noisy data
        X_train, y_train = dc.get_train(size, train_distribution)

        print("* Training set size: %d samples" % (len(y_train)))
        print_distribution("  Sentiment distribution", y_train)

        # Do an experiment
        do_experiment(X_train, y_train, X_test, y_test)

    print("-------------- Noisy training sets -------------")
    print("The proportions of the noise sources %s: " % [x[0] for x in lstNoisyInfo],
          [round(x[1]*100/dc.get_noisy_len(),1) for x in lstNoisyInfo])
    for size in [(4000, 1000), (8000, 2000), (15000, 5000)]:
        # Get a noisy training set
        X_train, y_train = dc.get_train_with_noisy(size[0], size[1], train_distribution)
        print("* Noisy training set size: %d samples (%d original, %d noisy)" % (len(y_train), size[0], size[1]))
        print_distribution("  Sentiment distribution", y_train)

        # Do an experiment
        do_experiment(X_train, y_train, X_test, y_test)
