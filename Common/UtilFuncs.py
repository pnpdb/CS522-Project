
class DataSize:
    # get the training size of baseline
    @staticmethod
    def GetTrainSizeBaseline():
        return [3000, 6000, 7500, 12000, 15000, 22500, 30000]

    # get training size with noisy data
    @staticmethod
    def GetTrainSizeWithNoisyData():
        #return [(6000, 1500), (12000, 3000), (22500, 7500)]
        return [(6000, 1500), (12000, 3000)]
    @staticmethod
    def GetTestDataSize():
        return 4000
    @staticmethod
    def GetNoiseDataSize():
        return 3000
    @staticmethod
    def GetValidationDataSize():
        return 1000