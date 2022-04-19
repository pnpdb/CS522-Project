import numpy as np
import pandas as pd
from Common.DataCenter import data_center
from Common.preprocessor import text_preprocessing_tfidf, one_hot_encoding
from Common.UtilFuncs import print_evaluation, Evaluator
from sklearn.neighbors import LocalOutlierFactor
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

class LocalOutlierFactorMethod:
    def __init__(self, neighbours = 2):
        self.Neighbours = neighbours
        pass
    
    def CleanDFNoise(self, trainDF, X_train_vec, y_train):
        trainDFs = []
        y_train = np.array(y_train)

        for sentiment in [0,1,2,3]:
            # detect noise
            trainDFForSentiment = self.DetectNoise(trainDF, X_train_vec, y_train, sentiment)
            trainDFs.append(trainDFForSentiment)
        # merge the dataframes
        mergedDF = pd.concat(trainDFs)
        # sort by index
        mergedDF = mergedDF.sort_index()
        # finally I drop the rows marked as noise
        cleanDF = mergedDF[mergedDF['markasnoise'] == 0]
        new_trainVec = X_train_vec[mergedDF['markasnoise'] == 0]
        new_ytrain = y_train[mergedDF['markasnoise'] == 0]
        return (cleanDF, new_trainVec, new_ytrain)
        
    def DetectNoise( self, trainDF, trainVec, y_train, sentimentType):
        pick_trainX = trainVec[y_train==sentimentType]
        clf = LocalOutlierFactor(n_neighbors=2)
        label_result = clf.fit_predict(pick_trainX)
        label_result = np.array(label_result)
        pick_trainDF = trainDF[y_train == sentimentType]
        # create result data frame
        resultDF = pd.DataFrame(label_result, columns=['markasnoise'], index = pick_trainDF.index)
        # change the flag 0: not noise 1: noise
        resultDF[resultDF['markasnoise'] == 1] = 0
        resultDF[resultDF['markasnoise']==-1] = 1
        # append markasnoise column to the train data frame
        newTrainDF = pick_trainDF.join(resultDF)
        return newTrainDF

# do an experiment denoised by LocalOutlierFactor
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
    #SVM = CalibratedClassifierCV(LinearSVC(dual=False, class_weight='balanced'))
    SVM             = OneVsRestClassifier(LinearSVC(dual=False, class_weight='balanced'), n_jobs=-1)

    # clean noise in noisy data frame
    denoiseMethod = LocalOutlierFactorMethod()
    cleanTrainDF, clean_X_train_vec, clean_y_train = denoiseMethod.CleanDFNoise(train_df, X_train_vec, y_train )

    SVM.fit(clean_X_train_vec, clean_y_train)
    y_pred          = SVM.predict(X_test_vec)

    # Print the evaluation
    # One hot encoding for print_evaluation()
    _, y_pred = one_hot_encoding(y_train, y_pred)
    print_evaluation(y_test_vec, y_pred, labels=[0,1,2,3])

    evaluateDF = Evaluator.do_evaluate(y_test_vec, y_pred)
    return evaluateDF, None
