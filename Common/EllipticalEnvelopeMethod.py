from heapq import merge
from Common.SVDUtils import SVD
from sklearn.covariance import EllipticEnvelope
import numpy as np
import pandas as pd
class EllipticalEnvelopeMethod:
    def __init__(self, dimension):
        self.Dimension = dimension
        pass
    
    def CleanDFNoise(self, trainDF, X_train_vec, y_train):
        '''
        Clean Noise in Data Frame By EllipticEnvelope method
        '''
        trainDFs = []
        y_train = np.array(y_train)

        for sentiment in [0,1,2,3]:
            # detect noise
            trainDFForSentiment = self.DetectNoise(trainDF, X_train_vec, y_train, sentiment, self.Dimension)
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
    
    def DetectNoise( self, trainDF, trainVec, y_train, sentimentType, dimension):
        # I only pick the data with the specific type
        pick_trainX = trainVec[y_train==sentimentType]    
        svd = SVD(dimension)
        svd.Process(pick_trainX)
        # now I try to remove the noise
        # instantiate model
        model = EllipticEnvelope(contamination = 0.1, support_fraction=0.8) 
        # fit model
        result = model.fit_predict(svd.U)
        pick_trainDF = trainDF[y_train == sentimentType]
        # create result data frame
        resultDF = pd.DataFrame(result, columns=['markasnoise'], index = pick_trainDF.index)
        # change the flag 0: not noise 1: noise
        resultDF[resultDF['markasnoise'] == 1] = 0
        resultDF[resultDF['markasnoise']==-1] = 1
        # append markasnoise column to the train data frame
        newTrainDF = pick_trainDF.join(resultDF)
        return newTrainDF
    