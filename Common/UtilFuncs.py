from sklearn.metrics import precision_score, recall_score, f1_score
from IPython.display import display
from Common.DataCenter import data_center
import pandas as pd
import pickle
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
        self.clear()

    def get_evaluate(self):
        return self.evaluateDF

    def clear(self):
        self.evaluateDF = None
        self.last_index  = 0

    def evaluate(self, y_true, y_pred, labels=[0,1,2,3]):
        return Evaluator.do_evaluate(y_true, y_pred, labels)

    @staticmethod
    def do_evaluate(y_true, y_pred, labels=[0,1,2,3]):
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

    def get_last_index(self):
        return self.last_index

    def print(self):
        if self.evaluateDF is not None:
            df = self.evaluateDF[[ 'Experiment', 'Origin', 'Noise', 'Denoised', 'Micro F1', 'Macro F1',
                                   'Weighted F1', 'Macro Precision', 'Macro Recall', 'F1 of classes',
                                   'Sentiments distribution', 'Noise sources distribution' ]]
            display(df.set_index("Experiment"))

    def plot(self, xValue, yValue, lines, title="Plot", xLabel=None, yLabel=None, colors = None, df=None, subtitle=None):
        if colors is None:
            colors = ['green', 'red', 'blue', 'brown', 'pink', 'black']
        if xLabel is None:
            xLabel = xValue.replace("x['","")
            xLabel = xLabel.replace("']","")
        if yLabel is None:
            yLabel = yValue.replace("y['","")
            yLabel = yLabel.replace("']","")
        if df is None:
            df = self.evaluateDF

        fig, ax = plt.subplots()
        plt.title(title+("" if subtitle is None else ("\n"+subtitle)))
        i = 0
        for k, v in lines.items():
            index_line = df.apply(lambda df: eval(v), axis=1)
            df_line    = df[index_line]
            if(len(df_line) == 0):
                continue
            x = df_line.apply(lambda x: eval(xValue), axis=1)
            y = df_line.apply(lambda y: eval(yValue), axis=1)
            ax.plot(x, y, colors[i], label=k)
            plt.scatter(x, y, marker="o", s=10)
            i = (i+1) % len(colors)

        plt.legend()
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.show()

class Lab():
    def __init__(self, data_file, noisy_sources=8000, total_train_size = 20000, total_test_size = 4000, validation_size = 1000):
        self.Ev  = Evaluator()
        self.clear()
        self.data_file                  = data_file
        self.noisy_sources              = noisy_sources
        self.total_train_size           = total_train_size
        self.total_test_size            = total_test_size
        self.validation_size            = validation_size
        self.experiment                 = None
        self.experiment_name            = ""
        self.experiment_denoising       = None
        self.experiment_denoising_name  = ""
        self.last_index                 = 0
        self.noisy_train_set_sizes      = None
        self.__create_sets()

    def suffle(self, rseed):
        self.dc.shuffle(rseed)

    def set_noisy_sources(self, noisy_sources):
        self.noisy_sources  = noisy_sources
        self.__create_sets()

    def clear(self):
        self.Ev.clear()

    def __create_sets(self):
        self.dc =  data_center(self.data_file,
                               train_size = self.total_train_size, test_size = self.total_test_size,
                               validation_size = self.validation_size,
                               noisy_size = self.noisy_sources)

    # Set the experiment without denoising
    # Paramter experiment is like: def do_experiment(train_df, test_df) which return EV:
    def set_experiment_no_denoising(self, experiments):
        self.experiment = None
        for name, v in experiments.items():
            if v[1] == True:
                self.experiment         = v[0]
                self.experiment_name    = name
                break

    # Set the experiment with denoising
    # Paramter experiment is like: def do_experiment_denoising(train_df, test_df) which return EV and other_info:
    def set_experiment_with_denoising(self, experiments_denoising):
        self.experiment_denoising = None
        for name, v in experiments_denoising.items():
            if v[1] == True:
                self.experiment_denoising         = v[0]
                self.experiment_denoising_name    = name
                break

    def do_batch_experiments(self, train_set_sizes):
        dc = self.dc
        experiment_no   = self.Ev.get_last_index() + 1

        # Get the test set for evaluation
        test_df = dc.get_test_df()

        train_distribution  = None

        # Run experiments with different training sets, and use the same test set.
        for size in train_set_sizes:
            bNoisy  = False
            if isinstance(size, tuple) or isinstance(size, list):
                bNoisy  = True

            if bNoisy:
                train_df = dc.get_train_with_noisy_df(size[0], size[1], train_distribution)
                X_noisy          = train_df[train_df['noise'] != 0]
                print("*%2d> Noisy training set size: %d samples (%d original, %d noisy)"
                      % (experiment_no, len(train_df), size[0], size[1]))
                data_center.print_distribution("  Sentiments", train_df['sentiment'])
                dc.print_noise_source_distribution("  Noise sources")
                print("  Before de-noising:")
            else:
                train_df = dc.get_train_df(size, train_distribution)
                print("*%2d> Training set size: %d samples" % (experiment_no, len(train_df)))
                data_center.print_distribution("  Sentiments", train_df['sentiment'])

            # Do an experiment
            dfResult = self.experiment(train_df, test_df)
            if bNoisy:
                self.Ev.add_evaluation(dfResult, size[0], size[1], "N",
                                  data_center.calc_distribution_str(train_df['sentiment'], 'sentiment', [0,1,2,3]),
                                  data_center.calc_distribution_str(X_noisy, 'noise', [1,2,3]),
                                  experiment_no)
            else:
                self.Ev.add_evaluation(dfResult, size, 0, "-",
                                  data_center.calc_distribution_str(train_df['sentiment'], 'sentiment', [0,1,2,3]),
                                  "-", experiment_no)

            if bNoisy and self.experiment_denoising:
                print("  After de-noising:")
                # Do an experiment with de-noising first
                dfResult, _ = self.experiment_denoising(train_df, test_df)
                self.Ev.add_evaluation( dfResult, size[0], size[1], "Y",
                                   data_center.calc_distribution_str(train_df['sentiment'], 'sentiment', [0,1,2,3]),
                                   data_center.calc_distribution_str(X_noisy, 'noise', [1,2,3]),
                                   experiment_no + len(train_set_sizes))

                self.noisy_train_set_sizes  = train_set_sizes

            experiment_no += 1

        return self.Ev

    def print(self):
        self.Ev.print()

    def plot(self):
        # Plot training set size vs. Macro F1
        # x coordinate
        xValue  = "x['Origin']+x['Noise']"
        if self.noisy_train_set_sizes is not None:
            xLabel  = "Training set size\nnoisy sets: %s" % \
                      str([str(x[0])+'+'+str(x[1]) for x in self.noisy_train_set_sizes]).replace("\'","")
        else:
            xLabel  = "Training set size"

        # y coordinate
        yValue  = "y['Macro F1']"

        # Divide experiments into several groups, each will be plotted as a line
        lines = { # each item: name, filter
            'Original Data':    "df['Denoised']=='-'",
            'Noisy Data':       "df['Denoised']=='N'",
            'Denoised Data':    "df['Denoised']=='Y'",
        }

        # Do plot
        self.Ev.plot(xValue = xValue, yValue = yValue, lines = lines,
                xLabel = xLabel,
                title = 'SVM using %s for de-noising' % self.experiment_denoising_name,
                subtitle = data_center.distribution2str(
                          "noise sources: ", self.dc.get_noise_source_distribution(), 3)
                )

    @staticmethod
    def get_active_experiment_name(experiments):
        for name, v in experiments.items():
            if v[1] == True:
                return name
        return None

    def save(self, filename):
        with open(filename , "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(filename):
        with open(filename , "rb") as fh:
            lab = pickle.load(fh)
        return lab

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
