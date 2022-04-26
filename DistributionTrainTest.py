#!/usr/bin/env python
# coding: utf-8
# Our base common modules
from Common.UtilFuncs import Evaluator, Lab

# Classifiers without denoising
import Common.SvmMethod as SvmMethod
import Common.BERTModel as BERTModel

# Denoising Methodes
import Common.IsolationForestMethod as IsolationForestMethod
import Common.ConfidentLearningMethod as ConfidentLearningMethod
import Common.LocalOutlierFactorMethod as LocalOutlierFactorMethod

# The settings of the noise sources.
# Each item: source -> (size, distribution)
noisy_set_sizes = {
    'mislabeled' : (2000, None),                   # max size: 15000
    'irrelevant' : (2000, None),  # max size: 34259
    'translated' : (2000, "reserve_labels"),       # max size: 5000
}

# Initialize the lab, which will run a serial of experiments
lab = Lab("twitter_sentiment_data_clean.csv", noisy_sources = noisy_set_sizes,
          total_train_size = 25000, total_test_size = 12500, validation_size=0)

# Choose a experiment without denoising
# Each item: name -> (funcion, whether choose) note:only the first active one will be used
experiment_without_denoising = {
    'SVM' : (SvmMethod.do_experiment, 0),
    'BERT' : (BERTModel.do_experiment_BERT, lab.dc.get_validation_df(), 1)
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

if __name__ == '__main__':
    lab_filename = "saving/" + str(noisy_train_set_sizes) + ".pk"

    RUN = 1      #1/0:  Run new experiments / Read results made by previous experiments
    if RUN:
        # Run new experiments
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


    lab.Ev.get_evaluate().to_csv("saving/distri_step.csv")