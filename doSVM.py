#!/usr/bin/env python
# coding: utf-8
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

# Our base common modules
from Common.DataCenter import data_center
from Common.preprocessor import text_preprocessing_tfidf, one_hot_encoding
from Common.UtilFuncs import print_evaluation, Evaluator, Lab

# Denoising Methodes
import Common.IsolationForestMethod as IsolationForestMethod
import Common.ConfidentLearningMethod as ConfidentLearningMethod
import Common.LocalOutlierFactorMethod as LocalOutlierFactorMethod

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
    X_train_vec, X_test_vec = text_preprocessing_tfidf(X_train, X_test)
    y_train_vec, y_test_vec = one_hot_encoding(y_train, y_test)

    # Run SVM and evaluate the results
    y_pred = run_SVM(X_train_vec, y_train_vec, X_test_vec, y_test_vec)

    # Print the evaluation
    print_evaluation(y_test_vec, y_pred, labels=[0,1,2,3])
    evaluateDF = Evaluator.do_evaluate(y_test_vec, y_pred)
    return evaluateDF

if __name__ == '__main__':
    # The settings of the noise sources.
    # Each item: source -> (size, distribution)
    noisy_set_sizes = {
        # 'mislabeled' : (8600, None),                   # max size: 15000
        'irrelevant' : (8600, [0.25,0.25,0.25,0.25]),  # max size: 34259
        # 'translated' : (8600, 100),       # max size: 5000
    }

    # Choose a experiment without denoising
    # Each item: name -> (funcion, whether choose) note:only the first active one will be used
    experiment_without_denoising = {
        'SVM without denoising' : (do_experiment, 1),
    }

    # Choose a experiment with denoising
    # Each item: name -> (funcion, whether choose) note:only the first active one will be used
    experiment_with_denoising = {
        'Confident Learning' : (ConfidentLearningMethod.do_experiment_with_denoising_for_SVM,   1),
        'Isolation Forest'   : (IsolationForestMethod.do_experiment_with_denoising_for_SVM,     1),
        'LocalOutlierFactor' : (LocalOutlierFactorMethod.do_experiment_with_denoising_for_SVM,  1),
    }

    # The training set of each experiment
    # origin_train_set_sizes = [2000, 4000, 5000, 8000, 10000, 15000, 20000]
    origin_train_set_sizes = [2000, 4000, 10000, 15000, 20000]
    noisy_train_set_sizes  = [(4000, 1000), (8000, 2000), (15000, 5000)]

    # Initialize the lab, which will run a serial of experiments
    lab = Lab("twitter_sentiment_data_clean.csv", noisy_sources = noisy_set_sizes, total_train_size = 20000, total_test_size = 4000)

    # To see the data features via a demo
    train_df = lab.dc.get_train_with_noisy_df(15000, 5000)
    data_center.print_data(train_df.head(15))

    # Calculate the filename for save the lab.
    lab_filename = Lab.get_active_experiment_name(experiment_with_denoising)
    if lab_filename is None:
        lab_filename  = Lab.get_active_experiment_name(experiment_without_denoising)
    if lab_filename is None:
        print("Nothing to do.")
        exit(0)
    lab_filename += str(noisy_train_set_sizes) + ".pk"

    # Run new experiments (or just review the evaluations saved by previous experiments)
    RUN = 1
    if RUN:     # Run new experiments
        # Set the function to classify data without denoising
        lab.set_experiment_no_denoising(experiment_without_denoising)

        # Set the function to classify data with denoising
        lab.set_experiment_with_denoising(experiment_with_denoising)

        print("-------------- No noisy training sets ----------")
        lab.do_batch_experiments(origin_train_set_sizes)

        print("-------------- Noisy training sets -------------")
        lab.do_batch_experiments(noisy_train_set_sizes)

        # Save the evaluations of lab
        lab.save(lab_filename)

    else:       # Load evaluations saved by previous experiments
        # Read evaluations saved by previous experiments
        lab = Lab.load(lab_filename)

    # Show evaluations in a form
    lab.print()

    # Plot the evaluations
    lab.plot()

