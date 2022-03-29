import pandas as pd
import random
import sklearn
from sklearn.model_selection import train_test_split

# A class manager the data, which loads the data, generates a test set and train sets with different sizes,
# and add noisy data to these train sets.
class data_center():
    def __init__(self, filename, test_size, noisy_size):  # file name, size of the test set, size of the noisy set
        # load the original data set
        df = pd.read_csv(filename, encoding='latin-1')
        df['encoded_cat'] = df.sentiment.astype("category").cat.codes
        df = df[df['message'] != None]
        df.dropna(inplace=True)

        self.rseed = 42                 # the seed for random
        self.dfOriginal = df            # the original set
        # self.X_train, self.y_train    # X,y of the training set
        # self.X_noisy, self.y_noisy    # X,y of the noisy set
        # self.X_test, y1, self.y_test  # X,y of the test set

        y = list(df['encoded_cat'])
        X = list(df['message'])

        if test_size <= 1:
            test_size  = int(self.get_len() * test_size)
        if noisy_size <= 1:
            noisy_size = int(self.get_len() * noisy_size)

        # Split a part out of the original set as the test set
        self.X_test, self.y_test, X1, y1, \
            = self.__split_set(X, y, test_size)

        # Split the rest part into two parts as the training set and the noisy set
        self.X_train, self.y_train, self.X_noisy, self.y_noisy\
            = self.__split_set(X1, y1, self.get_len() - test_size - noisy_size)

        print(len(y1), len(self.y_train), len(self.y_noisy))
        print(len(X1), len(self.X_train), len(self.X_noisy))

        # Change labels of noisy set
        random.seed(self.rseed)
        self.y_noisy = list(map(lambda x: (int(x) + random.randint(1, 3)) % 4, self.y_noisy))

    # Get the test set
    # size: size of the set, represented in proportion (if <= 1) or absolute value (if > 1)
    # return: X and y of test set
    def get_test(self, size=None):
        return self.__get_sub_set(self.X_test, self.y_test, size)

    # Get the noisy set
    # size: size of the set, represented in proportion (if <= 1) or absolute value (if > 1)
    # return: X and y of noisy set
    def get_noisy(self, size=None):
        return self.__get_sub_set(self.X_noisy, self.y_noisy, size)

    # Get the train set
    # size: size of the set, represented in proportion (if <= 1) or absolute value (if > 1)
    # return: X and y of training set
    def get_train(self, size=None):
        return self.__get_sub_set(self.X_train, self.y_train, size)

    # Get the train set with noisy data
    # original_size: size of the data from the original train set the set
    # noisy_size:    size of the noisy data
    def get_train_with_noisy(self, original_size = None, noisy_size = None):
        X_train, y_train   = self.get_train(original_size)
        if noisy_size is None:
            return X_train, y_train
        X_noisy, y_noisy   = self.get_noisy(noisy_size)
        df = pd.DataFrame(X_train + X_noisy, columns=['X'])
        df['y'] = y_train + y_noisy
        df = sklearn.utils.shuffle(df, self.rseed)  #shuffle
        return list(df['X']), list(df['y'])

    # Get the size of the whole original set
    def get_len(self):
        return len(self.dfOriginal)

    # Get the size of the whole train set
    def get_train_len(self):
        return len(self.y_train)

    # Get the size of the whole test set
    def get_test_len(self):
        return len(self.y_test)

    # Get the size of the whole noisy set
    def get_noisy_len(self):
        return len(self.y_noisy)

    # Get the subset with the specific size
    # return: X y of subset
    def __get_sub_set(self, x, y, size=None):
        X1, y1, X2, y2 = self.__split_set(x, y, size)
        return X1, y1

    # Split the dataset into two parts
    # size: the size of subset 1
    # return: X y of subset 1 and that of subset 2
    def __split_set(self, x, y, size=None):
        if size is None or size == 1:
            return x, y, None, None

        # Size is represented by proportion
        if size < 1:
            X1, X2, y1, y2 = \
                train_test_split(x, y, test_size=1 - size, random_state=self.rseed, stratify=y)
            return X1, y1, X2, y2

        # Size is represented by absolute value
        X1, X2, y1, y2 = \
            train_test_split(x, y, test_size=1 - (size + 1) / len(y), random_state=self.rseed, stratify=y)
        if len(y1) == size:
            return X1, y1, X2, y2
        else:
            return X1[:size], y1[:size], X2 + X1[-1:], y2 + y1[-1:]

if __name__ == '__main__':
    #Split the original data set into 3 parts: training set, test set, noisy set
    # dc = data_center("twitter_sentiment_data.csv", test_size=0.2, noisy_size=0.2) # sizes represented in proportions
    dc = data_center("twitter_sentiment_data.csv", test_size=8000, noisy_size=8000) # sizes represented in absolute values
    X_test, y_test              = dc.get_test()
    X_noisy, y_noisy            = dc.get_noisy()

    print("-----------------------------------------------")
    print("Original size is %d" % dc.get_len())
    print("Training set size is %d" % dc.get_train_len())
    print("Test set size is %d" % dc.get_test_len())
    print("Noisy set size is %d" % dc.get_noisy_len())

    print("\nGenerate training set with different sizes:")
    print("-----------------------------------------------")
    for size in [0.1, 0.5, 1]:         # training set sizes represented in proportions
        X_train, y_train = dc.get_train(size)
        print("Training set size: %.1f%% (%d samples) " % (len(y_train)/dc.get_train_len()*100, len(X_train)))

    print("-----------------------------------------------")
    for size in [1000, 2000, 2500]:     # training set sizes represented in absolute values
        X_train, y_train = dc.get_train(size)
        print("Training set size: %.1f%% (%d samples) " % (len(y_train)/dc.get_train_len()*100, len(X_train)))

    print("-----------------------------------------------")
    for size in [(2000, 500), (4000, 1000), (7500, 2500)]:     # training set sizes represented in absolute values
        X_train, y_train = dc.get_train_with_noisy(size[0], size[1])
        print("Noisy training set size %d samples " % len(X_train))
