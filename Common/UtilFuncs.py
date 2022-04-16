from sklearn.metrics import precision_score, recall_score, f1_score
from IPython.display import display
import pandas as pd
import matplotlib.pyplot as plt

# Print the evaluation
def print_evaluation(y_true, y_pred, labels=[0,1,2,3]):
    f1scores = []
    for i in range(len(labels)):
        f1scores.append(round(f1_score(y_true, y_pred, labels=[labels[i]], average='macro'),3))
    print("    f1 of classes: %s" % (str(f1scores)))
    micro_f1        = f1_score(y_true, y_pred, average='micro', zero_division=0)
    macro_f1        = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1     = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_recall    = recall_score(y_true, y_pred, average='macro', zero_division=0)

    print("    micro_f1: %.3f , macro_f1: %.3f , weighted_f1: %.3f, macro_precision: %.3f, macro_recall: %.3f" %
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
    df = pd.DataFrame({'sentiment':y})
    l  = len(df)
    c  = [y.count(x) for x in range(len(df.iloc[:,1].value_counts(sort = False)))]
    print("%s: %s" % (hint, ("%.1f%%, "*(len(c)-1)+"%.1f%%") % tuple([x*100/l for x in list(c)])))

# for general distribution
def show_distribution(hint, keys, distribution):
    print( "%s %s: %s" % (hint, str(tuple(keys)),
                          ("%.1f%%, "*(len(keys)-1)+"%.1f%%") %
                          tuple([x*100 for x in distribution])))

class Evaluator():
    def __init__(self):
        self.evaluateDF  = None
        self.last_index  = 0

    def get_evaluate(self):
        return self.evaluateDF.set_index("Experiment")

    def clear(self):
        self.evaluateDF = None

    def evaluate(self, y_true, y_pred):
        labels=[0,1,2,3]
        f1scores = []
        for i in range(len(labels)):
            f1scores.append(round(f1_score(y_true, y_pred, labels=[labels[i]], average='macro'),3))
        micro_f1        = f1_score(y_true, y_pred, average='micro', zero_division=0)
        macro_f1        = f1_score(y_true, y_pred, average='macro', zero_division=0)
        weighted_f1     = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        macro_recall    = recall_score(y_true, y_pred, average='macro', zero_division=0)

        df = pd.DataFrame({"Micro F1":[micro_f1], "Macro F1":[macro_f1], "Weighted F1":[weighted_f1], "Macro Precision": [macro_precision],
                           "Macro Recall":[macro_recall], "F1 of classes":[f1scores]})
        return df

    def add_evaluation(self, df, original_size, noisy_size, denoised, sentiments, noise_sources, index=None):
        if index is None:
            self.last_index += 1
            index   = self.last_index
        else:
            self.last_index = index

        df["Experiment"]    = int(index)
        df["Origin"]        = original_size
        df["Noise"]         = noisy_size
        df["Denoised"]      = denoised
        df["Sentiments distribution"]    = str(sentiments)
        df["Noise sources distribution"] = str(noise_sources)

        if self.evaluateDF is None:
            self.evaluateDF = df
        else:
            self.evaluateDF = pd.concat([self.evaluateDF,df],axis=0)
        self.evaluateDF.sort_values(by="Experiment", inplace=True, ascending=True)

    def print(self):
        if self.evaluateDF is not None:
            df = self.evaluateDF[[ 'Experiment', 'Origin', 'Noise', 'Denoised', 'Micro F1', 'Macro F1',
                                   'Weighted F1', 'Macro Precision', 'Macro Recall', 'F1 of classes',
                                   'Sentiments distribution', 'Noise sources distribution' ]]
            display(df.set_index("Experiment"))

    def plot(self, xValue, yValue, lines, title="Plot", xLabel=None, yLabel=None, colors = None, df=None):
        if colors is None:
            colors = ['green', 'red', 'blue', 'brown', 'pink', 'black']
        if xLabel is None:
            xLabel = xValue.replace("x['","")
            xLabel = xLabel.replace("']","")
        if yLabel is None:
            yLabel = yValue.replace("y['","")
            yLabel = yLabel.replace("']","")
        if df is None:
            df = self.evaluateDF.set_index("Experiment")

        fig, ax = plt.subplots()
        plt.title(title)
        i = 0
        for k, v in lines.items():
            index_line = df.apply(lambda df: eval(v), axis=1)
            df_line    = df[index_line]
            x = df_line.apply(lambda x: eval(xValue), axis=1)
            y = df_line.apply(lambda y: eval(yValue), axis=1)
            ax.plot(x, y, colors[i], label=k)
            i = (i+1) % len(colors)

        plt.legend()
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.show()

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
