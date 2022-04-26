#!/usr/bin/env python
# coding: utf-8
# Our base common modules
from Common.DataCenter import data_center
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
    'mislabeled' : (8600, None),                   # max size: 15000
    # 'irrelevant' : (8600, [0.25,0.25,0.25,0.25]),  # max size: 34259
    # 'translated' : (8600, 100),       # max size: 5000
}

# Initialize the lab, which will run a serial of experiments
# Split the database into training set, test set, noisy set, validation set
lab = Lab("twitter_sentiment_data_clean.csv", noisy_sources = noisy_set_sizes, total_train_size = 20000, total_test_size = 4000)

# Choose a experiment without denoising
# Each item: name -> (function, args-optional, whether choose) note:only the first active one will be used
experiment_without_denoising = {
    'SVM without denoising' : (SvmMethod.do_experiment, 0),
    'BERT  without denoising' : (BERTModel.do_experiment_BERT, lab.dc.get_validation_df(), 1)
}

# Choose a experiment with denoising
# Each item: name -> (funcion, args-optional, whether choose) note:only the first active one will be used
experiment_with_denoising = {
    'Confident Learning' : (ConfidentLearningMethod.do_experiment_with_denoising_for_SVM,   1),
    'Isolation Forest'   : (IsolationForestMethod.do_experiment_with_denoising_for_SVM,     0),
    'LocalOutlierFactor' : (LocalOutlierFactorMethod.do_experiment_with_denoising_for_SVM,  1),
}

# The training set of each experiment
origin_train_set_sizes = [2000, 4000, 5000, 8000, 10000, 15000, 20000]
# origin_train_set_sizes = [2000, 4000, 10000, 15000, 20000]
noisy_train_set_sizes  = [(4000, 1000), (8000, 2000), (15000, 5000)]

if __name__ == '__main__':
    # Review the summary of the whole data
    lab.dc.print_summary()

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
    lab_filename = "saving/" + lab_filename + str(noisy_train_set_sizes) + ".pk"

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

