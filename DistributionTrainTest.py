#!/usr/bin/env python
# coding: utf-8
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

# Our base common modules
from Common.DataCenter import data_center
from Common.preprocessor import text_preprocessing_tfidf, one_hot_encoding
from Common.UtilFuncs import print_evaluation, Evaluator, Lab

# Run SVM
# parameter:  vectorised X and encoded y of training set and test set
def run_SVM(X_train_vec, y_train_vec, X_test_vec, y_test_vec):
    # Run SVM - fit and predict
    SVM     = OneVsRestClassifier(LinearSVC(dual=False, class_weight='balanced'), n_jobs=-1)
    SVM.fit(X_train_vec, y_train_vec)
    y_pred  = SVM.predict(X_test_vec)
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
        'mislabeled' : (2000, None),                   # max size: 15000
        'irrelevant' : (2000, [0.25,0.25,0.25,0.25]),  # max size: 34259
        'translated' : (2000, "reserve_labels"),       # max size: 5000
    }

    # Choose a experiment without denoising
    # Each item: name -> (funcion, whether choose) note:only the first active one will be used
    experiment_without_denoising = {
        'SVM' : (do_experiment, 1),
    }

    # The training set of each experiment
    # origin_train_set_sizes = [2000, 4000, 5000, 8000, 10000, 15000, 20000]
    origin_train_set_sizes = [2000, 4000, 6000, 8000]
    noisy_train_set_sizes  = [(1500, 500), (3000, 1000), (4500,1500), (6000, 2000)]

    distributions = [(None, None),
                     (None,[0.25,0.25,0.25,0.25]),
                     ([0.25,0.25,0.25,0.25],None),
                     ([0.25,0.25,0.25,0.25],[0.25,0.25,0.25,0.25]),
                     ]

    lab_filename = "saving/" + str(noisy_train_set_sizes) + ".pk"

    RUN = 0      #1/0:  Run new experiments / Read results made by previous experiments
    if RUN:
        # Run new experiments
        # Initialize the lab, which will run a serial of experiments
        lab = Lab("twitter_sentiment_data_clean.csv", noisy_sources = noisy_set_sizes,
                  total_train_size = 25000, total_test_size = 12500, validation_size=0)

        # Set the function to classify data without denoising
        lab.set_experiment_no_denoising(experiment_without_denoising)

        for n in range(2):
            for i in range(len(distributions)):
                # Set distributions of training and test
                train_dist, test_dist = distributions[i]
                lab.set_distribution(train_dist, test_dist)

                # Run experiment on original data
                lab.do_batch_experiments(origin_train_set_sizes if n == 0 else noisy_train_set_sizes)

        # Save the results
        lab.save(lab_filename)
    else:
        # Read evaluations saved by previous experiments
        lab = Lab.load(lab_filename)

    # Show evaluations in a form
    lab.print()

    # # Plot the evaluations
    # lab.plot()

    # Plot training set size vs. Macro F1

    # x coordinate
    xValue  = "x['Origin']+x['Noise']"
    # y coordinate
    yValue  = "y['Macro F1']"
    len1 = len(origin_train_set_sizes)
    # len2 = len(noisy_train_set_sizes)   # assume len1 == len2 for convenience

    xLabel  = "Training set size\nno noise"

    # Divide experiments into several groups, each will be plotted as a line
    lines = { # each item: name, filter
        'Even-Even':         "int((df['Experiment']-1)/%d)==3"%len1,
        'Origin-Origin':     "int((df['Experiment']-1)/%d)==0"%len1,
        'Origin-Even':       "int((df['Experiment']-1)/%d)==1"%len1,
        'Even-Origin':       "int((df['Experiment']-1)/%d)==2"%len1,
    }

    # Do plot
    lab.Ev.plot(xValue = xValue, yValue = yValue, lines = lines,
                xLabel = xLabel, title = "SVM effected by different distribution")


    xLabel  = "Training set size\nnoisy sets: %s" % \
              str([str(x[0])+'+'+str(x[1]) for x in noisy_train_set_sizes]).replace("\'","")

    # Divide experiments into several groups, each will be plotted as a line
    lines = { # each item: name, filter
        'Even-Even':         "int((df['Experiment']-1)/%d)==7"%len1,
        'Origin-Origin':     "int((df['Experiment']-1)/%d)==4"%len1,
        'Origin-Even':       "int((df['Experiment']-1)/%d)==5"%len1,
        'Even-Origin':       "int((df['Experiment']-1)/%d)==6"%len1,
    }

    # Do plot
    lab.Ev.plot(xValue = xValue, yValue = yValue, lines = lines,
                xLabel = xLabel, title = "SVM effected by different distribution")


