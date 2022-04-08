# get the training size of baseline
def GetTrainSizeBaseline():
    return [6000, 7500, 12000, 15000, 22500, 30000]

# get training size with noisy data
def GetTrainSizeWithNoisyData():
    return [(6000, 1500), (12000, 3000), (22500, 7500)]