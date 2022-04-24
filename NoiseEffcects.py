#!/usr/bin/env python
# coding: utf-8
# Our base common modules
from Common.UtilFuncs import Evaluator, Lab

# Classifiers without denoising
import Common.SvmMethod as SvmMethod

# Denoising Methodes
import Common.IsolationForestMethod as IsolationForestMethod
import Common.ConfidentLearningMethod as ConfidentLearningMethod
import Common.LocalOutlierFactorMethod as LocalOutlierFactorMethod
from Common.BERTModel import do_experiment_BERT

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
    'translated' : (8600, 0.50),                   # max size: 5000
}

noisy_set_sizes4 = {
    # 'mislabeled' : (8600, None),                   # max size: 15000
    # 'irrelevant' : (8600, [0.25,0.25,0.25,0.25]),  # max size: 34259
    'translated' : (8600, 1),                   # max size: 5000
}

# Choose a experiment without denoising
# Each item: name -> (funcion, whether choose) note:only the first active one will be used
experiment_without_denoising = {
    'SVM' : (SvmMethod.do_experiment, 0),
    'BERT' : (do_experiment_BERT, 1),
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

RUN = 0      #1/0:  Run new experiments / Read results made by previous experiments

if __name__ == '__main__':
    if RUN:
        # Run new experiments
        # Initialize the lab, which will run a serial of experiments

        lab = Lab("twitter_sentiment_data_clean.csv", noisy_sources = noisy_set_sizes0, total_train_size = 20000, total_test_size = 4000)
        experiment_without_denoising['BERT'] = (do_experiment_BERT, (lab,), 1)
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
        lab.save("saving/noise_effect.pk")
    else:
        # Read evaluations saved by previous experiments
        lab = Lab.load("saving/noise_effect.pk")

    # Show evaluations in a form
    lab.print()

    # Plot the evaluations
    # lab.plot()

    for i in range(2):
        # Plot training set size vs. Macro F1

        # x coordinate
        if i == 0:
            xValue  = "x['Origin']+x['Noise']"
            xLabel  = "Training set total size\nnoisy sets: %s" % \
                      str([str(x[0])+'+'+str(x[1]) for x in noisy_train_set_sizes]).replace("\'","")
        else:
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
            'Translated Noise(50% mislabeled)':   "int((df['Experiment']-1-%d)/%d)==3"%(len1,len2),
            'Translated Noise(100% mislabeled)':  "int((df['Experiment']-1-%d)/%d)==4"%(len1,len2),
        }

        # Do plot
        lab.Ev.plot(xValue = xValue, yValue = yValue, lines = lines, ymin = 0.5, ymax = 0.72,
                    xLabel = xLabel, title = "SVM effected by various noises")

    lab.Ev.get_evaluate().to_csv("saving/noise_effect.csv")
