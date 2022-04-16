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
from IPython.display import display
from Common.DataCenter import data_center
from Common.preprocessor import normalize_preprocessing
from Common.UtilFuncs import print_evaluation, Evaluator

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

# do an experiment withou denoise
# Parameter: original X,y of training set and test set
def do_experiment(X_train, y_train, X_test, y_test):
    # Convert texts to vectors
    X_train_vec, X_test_vec = text_preprocessing(X_train, X_test)
    y_train_vec, y_test_vec = one_hot_encoding(y_train, y_test)

    # Run SVM and evaluate the results
    y_pred = run_SVM(X_train_vec, y_train_vec, X_test_vec, y_test_vec)

    # Print the evaluation
    print_evaluation(y_test_vec, y_pred, labels=[0,1,2,3])
    evaluateDF = Ev.evaluate(y_test_vec, y_pred)
    return evaluateDF

# do an experiment with denoise
# Parameter: original X,y of training set and test set
def do_experiment_denoise(X_train, y_train, X_test, y_test):
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
    return evaluateDF

if __name__ == '__main__':
    # The settings of the noise sources. source : (size, distribution)
    noisy_set_sizes = {
        'mislabeled' : (8000, None),                   # max size: 15000
        'irrelevant' : (2000, [0.25,0.25,0.25,0.25]),  # max size: 34259
        'translated' : (2000, "reserve_labels"),       # max size: 5000
    }

    # Load the database and split it into training set, test set, noisy set, validation set
    # dc = data_center("twitter_sentiment_data_clean.csv", train_size = 20000, test_size = 4000, validation_size = 1000,
    #                  noisy_size = noisy_set_sizes['mislabeled'][0] if 'mislabeled' in noisy_set_sizes.keys() else 0)

    dc = data_center("twitter_sentiment_data_clean.csv", train_size = 20000, test_size = 4000, validation_size = 1000,
                     noisy_size = noisy_set_sizes)

    # Show the summary of the whole data
    dc.print_summary()

    # To see the data features via a demo
    train_df = dc.get_train_with_noisy_df(1000,1000)
    data_center.print_data(train_df.head(15))

    # Get the test set for evaluation
    X_test, y_test = dc.get_test()

    # distribution of training set
    train_distribution = None

    LOAD_OLD_RESULT    = False

    # Run experiments with different training sets, and use the same test set.
    expriment_no    = 1
    print("-------------- No noisy training sets ----------")
    for size in [2000, 4000, 5000, 8000, 10000, 15000, 20000]:
        if LOAD_OLD_RESULT:
            break
        # Get a training set without noisy data
        X_train, y_train = dc.get_train(size, train_distribution)

        print("* Training set size: %d samples" % (len(y_train)))
        data_center.print_distribution("  Sentiments", y_train)

        # Do an experiment
        dfResult = do_experiment(X_train, y_train, X_test, y_test)
        Ev.add_evaluation(dfResult, size, 0, "-",
                                  data_center.calc_distribution_str(y_train, 'sentiment', [0,1,2,3]),
                                  "-", expriment_no
                                  )
        expriment_no += 1

    print("-------------- Noisy training sets -------------")
    dc.print_noise_source_distribution("General noise source distribution")

    lstSizes = [(4000, 1000), (8000, 2000), (15000, 5000)]
    # lstSizes = [(500, 500),(1000, 500),(1500, 500)]
    for size in lstSizes:
        if LOAD_OLD_RESULT:
            break
        # Get a noisy training set
        train_df         = dc.get_train_with_noisy_df(size[0], size[1], train_distribution)
        X_train, y_train = data_center.Xy(train_df)
        X_noisy          = train_df[train_df['noise'] != 0]

        print("* Noisy training set size: %d samples (%d original, %d noisy)" % (len(y_train), size[0], size[1]))
        data_center.print_distribution("  Sentiments", y_train)
        dc.print_noise_source_distribution("  Noise sources")

        # Do an experiment without de-noising
        print("  Before de-noising:")
        dfResult = do_experiment(X_train, y_train, X_test, y_test)
        Ev.add_evaluation(dfResult, size[0], size[1], "N",
                                data_center.calc_distribution_str(y_train, 'sentiment', [0,1,2,3]),
                                data_center.calc_distribution_str(X_noisy, 'noise', [1,2,3]),
                                expriment_no
                                )

        # Do an experiment with de-noising
        print("  After de-noising:")
        dfResult = do_experiment_denoise(X_train, y_train, X_test, y_test)
        Ev.add_evaluation(dfResult, size[0], size[1], "Y",
                                data_center.calc_distribution_str(y_train, 'sentiment', [0,1,2,3]),
                                data_center.calc_distribution_str(X_noisy, 'noise', [1,2,3]),
                                expriment_no + len(lstSizes)
                                )
        expriment_no += 1

    if(LOAD_OLD_RESULT):
        df = pd.read_csv("tmpeval.csv")[[ 'Experiment', 'Origin', 'Noise', 'Denoised', 'Micro F1', 'Macro F1',
                                         'Weighted F1', 'Macro Precision', 'Macro Recall', 'F1 of classes',
                                         'Sentiments distribution', 'Noise sources distribution' ]]
        df.sort_values(by="Experiment", inplace=True, ascending=True)
        df.set_index("Experiment", inplace=True)
        display(df)
    else:
        # Show evaluations in a form
        Ev.print()
        df = Ev.get_evaluate()
        df.reset_index().to_csv("tmpeval.csv")
        df = None

    # Plot training set size vs. Macro F1
    # x coordinate
    xValue  = "x['Origin']+x['Noise']"
    # y coordinate
    yValue  = "y['Macro F1']"

    # Divide experiments into several groups, which will be plot as lines
    lines = { # each item: name, filter
        'Original Data':      "df['Denoised']=='-'",
        'Noisy Data':       "df['Denoised']=='N'",
        'Denoised Data':    "df['Denoised']=='Y'",
    }

    # Do plot
    Ev.plot(xValue = xValue, yValue = yValue, lines = lines,
            title = 'SVM using confident learning for de-noising', df = df)

