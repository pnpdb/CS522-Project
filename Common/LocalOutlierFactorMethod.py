import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
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
        
    