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
    noisy_set_sizes = {
        'mislabeled' : (8000, None),                   # max size: 15000
        'irrelevant' : (0000, [0.25,0.25,0.25,0.25]),  # max size: 34259
        'translated' : (0000, "reserve_labels"),       # max size: 5000
    }

    # Choose a denoising method
    # Each item: name -> (funcion, whether choose) note:only the first active one will be used
    denoising_method = {
        'Confident Learning' : (do_experiment_denoised_by_ConfidentLearning, 0),
        'Isolation Forest'   : (do_experiment_denoised_by_IsolationForest,   1),
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
    test_df = dc.get_test_df()

    # distribution of training set
    train_distribution = None

    LOAD_OLD_RESULT    = 1

    # Run experiments with different training sets, and use the same test set.
    expriment_no    = 1
    print("-------------- No noisy training sets ----------")
    for size in [2000, 4000, 5000, 8000, 10000, 15000, 20000]:
        if LOAD_OLD_RESULT:
            break
        # Get a training set without noisy data
        train_df = dc.get_train_df(size, train_distribution)
        print("*%2d> Training set size: %d samples" % (expriment_no, len(train_df)))
        data_center.print_distribution("  Sentiments", train_df['sentiment'])

        # Do an experiment
        dfResult = do_experiment(train_df, test_df)
        Ev.add_evaluation(dfResult, size, 0, "-",
                          data_center.calc_distribution_str(train_df['sentiment'], 'sentiment', [0,1,2,3]),
                          "-", expriment_no
                         )
        expriment_no += 1

    print("-------------- Noisy training sets -------------")
    dc.print_noise_source_distribution("General noise source distribution")

    do_experiment_denoise  = None
    for denoise_name, v in denoising_method.items():
        if v[1] == True:
            do_experiment_denoise = v[0]
            break
    if do_experiment_denoise is None:
        print("No denoising function selected.")

    lstNoiseSizes = [(4000, 1000), (8000, 2000), (15000, 5000)]
    for size in lstNoiseSizes:
        if LOAD_OLD_RESULT:
            break
        # Get a noisy training set
        train_df         = dc.get_train_with_noisy_df(size[0], size[1], train_distribution)
        X_noisy          = train_df[train_df['noise'] != 0]

        print("*%2d> Noisy training set size: %d samples (%d original, %d noisy)"
              % (expriment_no, len(train_df), size[0], size[1]))
        data_center.print_distribution("  Sentiments", train_df['sentiment'])
        dc.print_noise_source_distribution("  Noise sources")

        # Do an experiment without de-noising
        print("  Before de-noising:")
        dfResult = do_experiment(train_df, test_df)
        Ev.add_evaluation(dfResult, size[0], size[1], "N",
                                data_center.calc_distribution_str(train_df['sentiment'], 'sentiment', [0,1,2,3]),
                                data_center.calc_distribution_str(X_noisy, 'noise', [1,2,3]),
                                expriment_no
                                )

        # Do an experiment with de-noising
        if do_experiment_denoise is None:
            continue

        print("  After de-noising:")
        dfResult, _ = do_experiment_denoise(train_df, test_df)

        Ev.add_evaluation(dfResult, size[0], size[1], "Y",
                                data_center.calc_distribution_str(train_df['sentiment'], 'sentiment', [0,1,2,3]),
                                data_center.calc_distribution_str(X_noisy, 'noise', [1,2,3]),
                                expriment_no + len(lstNoiseSizes)
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
    xLabel  = "Size of training sets\nnoisy sets: %s" % str(lstNoiseSizes)
    # y coordinate
    yValue  = "y['Macro F1']"

    # Divide experiments into several groups, each will be plotted as a line
    lines = { # each item: name, filter
        'Original Data':    "df['Denoised']=='-'",
        'Noisy Data':       "df['Denoised']=='N'",
        'Denoised Data':    "df['Denoised']=='Y'",
    }

    # Do plot
    Ev.plot(xValue = xValue, yValue = yValue, lines = lines,
            xLabel = xLabel,
            title = 'SVM using %s for de-noising' % denoise_name,
            subtitle = data_center.distribution2str(
                        "noise sources: ", dc.get_noise_source_distribution(), 3),
            df = df)

