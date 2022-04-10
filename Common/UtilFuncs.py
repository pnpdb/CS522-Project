from sklearn.metrics import precision_score, recall_score, f1_score
from Common.DataCenter import data_center
import pandas as pd
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

def EvaluationToDF(title, y_true, y_pred, labels=[0,1,2,3]):
    f1scores = []
    for i in range(len(labels)):
        f1scores.append(round(f1_score(y_true, y_pred, labels=[labels[i]], average='macro'),3))
    #print("  f1 of classes: %s" % (str(f1scores)))
    micro_f1        = f1_score(y_true, y_pred, average='micro', zero_division=0)
    macro_f1        = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1     = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_recall    = recall_score(y_true, y_pred, average='macro', zero_division=0)
    
    df = pd.DataFrame({"Data Size":[title], "Micro F1":[micro_f1], "Macro F1":[macro_f1], "Weighted F1":[weighted_f1], "Macro Precision": [macro_precision],
                    "Macro Recall":[macro_recall], "F1 of class 0":[f1scores[0]], "F1 of class 1":[f1scores[1]], 
                  "F1 of class 2":[f1scores[2]], "F1 of class 3":[f1scores[3]]})
    return df
    
# print the distribution of labels
def print_distribution(hint, y):
    df = data_center.df((y, y))
    l  = len(df)
    c  = [y.count(x) for x in range(len(df.iloc[:,1].value_counts(sort = False)))]
    print("%s: %s" % (hint, ("%.1f%%, "*(len(c)-1)+"%.1f%%") % tuple([x*100/l for x in list(c)])))

class DataSize:
    # get the training size of baseline
    @staticmethod
    def GetTrainSizeBaseline():
        return [2000, 4000, 5000, 8000, 10000, 15000, 20000]

    # get training size with noisy data
    @staticmethod
    def GetTrainSizeWithNoisyData():
        return [(4000, 1000), (8000, 2000), (15000, 5000)]

    @staticmethod
    def GetTestDataSize():
        return 4000
    @staticmethod
    def GetNoiseDataSize():
        return 3000
    @staticmethod
    def GetValidationDataSize():
        return 1000

