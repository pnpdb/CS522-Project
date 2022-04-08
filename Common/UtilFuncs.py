from sklearn.metrics import precision_score, recall_score, f1_score
# Print the evaluation
def print_evaluation(y_true, y_pred, labels=[0,1,2,3]):
    f1scores = []
    for i in range(len(labels)):
        f1scores.append(round(f1_score(y_true, y_pred, labels=[labels[i]], average='macro'),3))
    print("  f1 of classes: %s" % (str(f1scores)))
    micro_f1        = f1_score(y_true, y_pred, average='micro', zero_division=0)
    macro_f1        = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1     = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_recall    = recall_score(y_true, y_pred, average='macro', zero_division=0)

    print("  micro_f1: %.3f , macro_f1: %.3f , weighted_f1: %.3f, macro_precision: %.3f, macro_recall: %.3f" %
          (micro_f1, macro_f1, weighted_f1, macro_precision, macro_recall))

class DataSize:
    # get the training size of baseline
    @staticmethod
    def GetTrainSizeBaseline():
        return [3000, 6000, 7500, 12000, 15000, 22500, 30000]

    # get training size with noisy data
    @staticmethod
    def GetTrainSizeWithNoisyData():
        return [(6000, 1500), (12000, 3000), (22500, 7500)]

    @staticmethod
    def GetTestDataSize():
        return 4000
    @staticmethod
    def GetNoiseDataSize():
        return 3000
    @staticmethod
    def GetValidationDataSize():
        return 1000

