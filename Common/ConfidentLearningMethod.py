import numpy as np
from Common.DataCenter import data_center
from Common.preprocessor import text_preprocessing_tfidf, one_hot_encoding
from Common.UtilFuncs import print_evaluation, Evaluator
from sklearn.svm import LinearSVC
from cleanlab.classification import LearningWithNoisyLabels
from sklearn.calibration import CalibratedClassifierCV

# do an experiment denoised by ConfidentLearning
# Parameter: training set and test set
# Return evaluation info, denoising info
def do_experiment_with_denoising_for_SVM(train_df, test_df):
    X_train, y_train = data_center.Xy(train_df)
    X_test, y_test   = data_center.Xy(test_df)

    # Convert texts to vectors
    X_train_vec, X_test_vec = text_preprocessing_tfidf(X_train, X_test)
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

    evaluateDF = Evaluator.do_evaluate(y_test_vec, y_pred)
    return evaluateDF, None
