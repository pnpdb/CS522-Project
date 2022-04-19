import numpy as np
from Common.DataCenter import data_center
from Common.preprocessor import text_preprocessing_tfidf, one_hot_encoding
from Common.UtilFuncs import print_evaluation, Evaluator
from sklearn.ensemble import IsolationForest
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier

# do an experiment denoised by IsolationForest
# Parameter: training set and test set
# Return evaluation info, denoising info
def do_experiment_with_denoising_for_SVM(train_df, test_df):
    X_train, y_train = data_center.Xy(train_df)
    X_test, y_test   = data_center.Xy(test_df)

    # Convert texts to vectors
    X_train_vec, X_test_vec = text_preprocessing_tfidf(X_train, X_test)
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
    X_train_vec, X_test_vec = text_preprocessing_tfidf(X_train, X_test)
    y_train_vec, y_test_vec = one_hot_encoding(y_train, y_test)

    # Run SVM - fit and predict
    SVM             = OneVsRestClassifier(LinearSVC(dual=False, class_weight='balanced'), n_jobs=-1)
    SVM.fit(X_train_vec, y_train_vec)
    y_pred          = SVM.predict(X_test_vec)

    # Evaluate the results
    print_evaluation(y_test_vec, y_pred, labels=[0,1,2,3])
    evaluateDF = Evaluator.do_evaluate(y_test_vec, y_pred)

    return evaluateDF, denoisedDF