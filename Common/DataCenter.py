import pandas as pd
import random
import sklearn
from sklearn.model_selection import train_test_split

# A class manager the data, which loads the data, generates a test set and train sets with different sizes,
# and add noisy data to these train sets.
class data_center():
    # parameter: file name, size of the test set, size of the noisy set
    def __init__(self, filename, test_size, noisy_size, validation_size = 0):
        # load the original data set
        df          = pd.read_csv(filename, encoding='UTF-8')
        self.dfRaw  = df                                     # the raw set just load from the dataset file.

        # do some cleaning, such as drop the NAs, drop the duplicates
        df                = df.copy()
        df['sentiment'] = df['sentiment'].astype("category").cat.codes
        df = df[df['message'] != None]
        df.dropna(inplace=True)
        df = df.drop_duplicates(subset=['message'],keep='first')

        self.dfOriginal         = df                           # let the cleaned set be the original set
        self.shuffle(rseed = 522)                       # shuffle the original set
        self.train_size         = len(df) - test_size - noisy_size - validation_size
        self.test_size          = test_size
        self.validation_size    = validation_size
        self.noisy_size         = noisy_size

    def shuffle(self, rseed):
        self.rseed       = rseed                        # the seed for random
        df               = sklearn.utils.shuffle(self.dfOriginal, random_state=self.rseed)

        # Split into 4 groups by 'sentiment'
        self.dfs         = []
        for i in range(4):
            self.dfs.append(df[df['sentiment'] == i])

    # Get a subset with the same 'sentiment' distribution as the whole original set
    # start: start position in the whole uniformed original set
    # size:  size of the subset
    def __get_uniformed_subset(self, start, size):
        df = self.dfOriginal[:0]
        for i in range(4):
            ratio   = len(self.dfs[i]) / len(self.dfOriginal)
            s       = int(round(start * ratio))
            c       = int(round(size * ratio))
            if c > 0:
                df  = pd.concat([df, self.dfs[i][s : s + c]])
        df = sklearn.utils.shuffle(df[:size], random_state=self.rseed)
        return df

    # Get the train set
    # size: size of the train set
    # return: X and y of training set
    def get_train(self, size=None):
        if size is None:
            size = self.train_size
        if size > self.train_size:
            raise Exception("The size is large than the max train size!")
        df = self.__get_uniformed_subset(0, size)
        return list(df['message']), list(df['sentiment'])

    # Get a test set
    # size: size of the test set
    # return: X and y of test set
    def get_test(self, size=None):
        if size is None:
            size = self.test_size
        if size > self.test_size:
            raise Exception("The size is large than the max test size!")
        df = self.__get_uniformed_subset(self.train_size, size)
        return list(df['message']), list(df['sentiment'])

    # Get a validation set
    # size: size of the validation set
    # return: X and y of noisy set
    def get_validation(self, size=None):
        if size is None:
            size = self.validation_size
        if size > self.validation_size:
            raise Exception("The size is large than the max validation size!")
        df = self.__get_uniformed_subset(self.train_size + self.test_size, size)
        return list(df['message']), list(df['sentiment'])

    # Get a noisy set
    # size: size of the noisy set
    # return: X and y of noisy set
    def get_noisy(self, size=None):
        if size is None:
            size = self.noisy_size
        if size > self.noisy_size:
            raise Exception("The size is large than the max noisy size!")

        df = self.__get_uniformed_subset(self.train_size + self.test_size + self.validation_size, size)

        # Change labels to make noise
        random.seed(self.rseed)
        df['sentiment'] = list(map(lambda x: (int(x) + random.randint(1, 3)) % 4, df['sentiment']))

        return list(df['message']), list(df['sentiment'])

    # Get the train set with noisy data
    # original_size: size of the data from the original train set the set
    # noisy_size:    size of the noisy data
    def get_train_with_noisy(self, original_size = None, noisy_size = None):
        X_train, y_train   = self.get_train(original_size)
        if noisy_size is None:
            return X_train, y_train
        X_noisy, y_noisy   = self.get_noisy(noisy_size)
        df = pd.DataFrame(X_train + X_noisy, columns=['message'])
        df['sentiment'] = y_train + y_noisy
        df = sklearn.utils.shuffle(df, random_state=self.rseed)  #shuffle
        return list(df['message']), list(df['sentiment'])

    # Get the size of the raw set
    def get_raw_len(self):
        return len(self.dfRaw)

    # Get the size of the whole original set
    def get_len(self):
        return len(self.dfOriginal)

    # Get the size of the whole train set
    def get_train_len(self):
        return self.train_size

    # Get the size of the whole test set
    def get_test_len(self):
        return self.test_size

    # Get the size of the whole validation set
    def get_validation_len(self):
        return self.validation_size

    # Get the size of the whole noisy set
    def get_noisy_len(self):
        return self.noisy_size

    # Get the subset with the specific size
    # return: X y of subset
    def get_sub_set(self, x, y, size=None):
        X1, y1, X2, y2 = self.split_set(x, y, size)
        return X1, y1

    # Split the dataset into two parts
    # size: the size of subset 1
    # return: X y of subset 1 and that of subset 2
    def split_set(self, x, y, size=None):
        if size is None or size == 1:
            return x, y, None, None

        if size > len(y):
            raise Exception("Size of the subset is large than the set!")

        # Size is represented by absolute value
        X1, X2, y1, y2 = \
            train_test_split(x, y, test_size=1 - (size + 1) / len(y), random_state=self.rseed, stratify=y)
        if len(y1) == size:
            return X1, y1, X2, y2
        else:
            return X1[:size], y1[:size], X2 + X1[-1:], y2 + y1[-1:]

    @staticmethod
    def df(Xy):
        return pd.DataFrame({'message':Xy[0] , 'sentiment':Xy[1]})

if __name__ == '__main__':
    def print_distribution(hint, y):
        df = data_center.df((y, y))
        c = df['sentiment'].value_counts()
        l = len(df)
        print("%s: ( 0 : %.1f%%, 1 : %.1f%%, 2: %.1f%%, 3 : %.1f%%)"
              % (hint, c[0] * 100 / l, c[1] * 100 / l, c[2] * 100 / l, c[3] * 100 / l))

    #Split the original data set into 3 parts: training set, test set, noisy set
    dc = data_center("twitter_sentiment_data.csv", test_size=8000, noisy_size=8000, validation_size=5000)
    dc.shuffle(rseed = 522) # Shuffle to setup a series of experiments
    X_train, y_train              = dc.get_train()

    print("-----------------------------------------------------------------------")
    print("Raw set (not cleaned) size is %d"    % dc.get_raw_len())
    print("Original set size is %d"             % dc.get_len())
    print("Training set size is %d"             % dc.get_train_len())
    print("Test set size is %d"                 % dc.get_test_len())
    print("Noisy set size is %d"                % dc.get_noisy_len())
    print("Validation set size is %d"           % len(dc.get_validation()[0]))
    print_distribution("Sentiment distribution of the whole training set", y_train)

    print("\nGenerate training set with different sizes:")
    print("-----------------------------------------------------------------------")
    for size in [1000, 2000, 2500]:     # training set sizes represented in absolute values
        X_train, y_train = dc.get_train(size)
        print("Training set size: %5d samples" % (len(y_train)))
        print_distribution("Sentiment distribution", y_train)

    print("-----------------------------------------------------------------------")
    for size in [(2000, 500), (4000, 1000), (7500, 2500)]:     # training set sizes represented in absolute values
        X_train, y_train = dc.get_train_with_noisy(size[0], size[1])
        print("Noisy training set size: %5d samples (%d + %d)" % (len(X_train), size[0], size[1]))
        print_distribution("Sentiment distribution", y_train)

    # X_train1, y_train1 = dc.get_train_with_noisy(10, 5)
    # # X_train1, y_train1 = dc.get_train(15)
    # print(X_train1)
    # X_train2, y_train2 = dc.get_train(10)
    # print(X_train2)
    # print(set(X_train1)-set(X_train2))

    # df1 = data_center.df(dc.get_test(8000))
    # df2 = data_center.df(dc.get_train(4000))
    # print(df1.sentiment.value_counts())
    # print(df2.sentiment.value_counts())
    # dd = df2.sentiment.value_counts()
    # print(dd[0],dd[1],dd[2],dd[3])
    # print(len(set(df2['message'])-set(df1['message'])))
