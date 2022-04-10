import pandas as pd
import os
import random
import sklearn
from sklearn.model_selection import train_test_split

# A class manager the data, which loads the data, generates a test set and train sets with different sizes,
# and add noisy data to these train sets.
class data_center():
    # parameter: file name, size of the test set, size of the noisy set
    def __init__(self, filename, test_size, noisy_size, validation_size = 0, train_size = 20000):
        # load the original data set
        df          = self.__read_csv_safe(filename)
        self.dfRaw  = df                                     # the raw set just load from the dataset file.

        # do some cleaning, such as drop the NAs, drop the duplicates
        df                = df.copy()
        # df['sentiment'] = df['sentiment'].astype("category").cat.codes
        df['sentiment'] = df['sentiment'].astype("int")
        df['sentiment'] = df['sentiment'].apply(lambda x: x+1)

        df = df[df['message'] != None]
        df.dropna(inplace=True)
        df = df.drop_duplicates(subset=['message'],keep="first")
        df = df.drop_duplicates(subset=['tweetid'],keep="first")
        self.class_count = len(df['sentiment'].value_counts(sort = False))

        self.dfOriginal         = df                    # let the cleaned set be the original set
        self.shuffle(rseed = 522)                       # shuffle the original set
        self.train_size         = train_size if train_size is not None else len(df) - test_size - noisy_size - validation_size
        self.test_size          = test_size
        self.validation_size    = validation_size
        self.noisy_size         = noisy_size
        if self.train_size + self.test_size + self.validation_size + self.noisy_size > len(df):
            raise Exception("The sum of the size of all sets is bigger than the size of the whole data set!" % (size, self.train_size))

        self.dfNoisy            = None                  # the noisy set

        # the distribution of the whole original set
        # self.distribution       = [x/len(df) for x in list(df['sentiment'].value_counts(sort = False))]
        self.distribution       = [list(df['sentiment']).count(x)/len(df) for x in range(self.class_count)]

    # Shuffle to generate all the data sets in a new way.
    def shuffle(self, rseed):
        self.rseed       = rseed                        # the seed for random
        df               = sklearn.utils.shuffle(self.dfOriginal, random_state=self.rseed)

        # Split into several groups by 'sentiment'
        self.dfs         = []
        for i in range(self.class_count):
            self.dfs.append(df[df['sentiment'] == i])

    # start: start position in the whole uniformed original set
    # size:  size of the subset
    # distribution: the 'sentiment' distribution. None if use the distribution of the whole original set
    def __get_subset(self, start, set_size, size, distribution = None):
        dist = self.distribution if distribution is None else distribution
        df = self.dfOriginal[:0]
        for i in range(self.class_count):
            s = int(round(start * self.distribution[i]))    # Use original distribution for getting the start position
            c = int(round(size * dist[i]))                  # Use specific distribution to calc the size of every label
            if i == self.class_count - 1:
                c = size - len(df)
            if c > int(round(set_size * self.distribution[i])+1):
                print("Cannot generate the set with size %d and distribution %f in a %d size set!" \
                      % (size, dist[i], set_size))
                exit(-1)
            if c > 0:
                df  = pd.concat([df, self.dfs[i][s : s + c]])
        df = sklearn.utils.shuffle(df[:size], random_state=self.rseed)
        return df

    # Get the train set in dataframe format
    def __get_train_df(self, size=None, distribution = None):
        if size is None:
            size = self.train_size
        if size > self.train_size:
            raise Exception("The size %d is large than the max train size %d!" % (size, self.train_size))
        df = self.__get_subset(0, self.train_size, size, distribution)
        return df

    # Get the train set
    # size: size of the train set
    # return: X and y of training set
    # distribution: the 'sentiment' distribution. None if use the distribution of the whole original set
    def get_train(self, size=None, distribution = None):
        df = self.__get_train_df(size, distribution)
        return list(df['message']), list(df['sentiment'])

    # Get a test set
    # size: size of the test set
    # return: X and y of test set
    # distribution: the 'sentiment' distribution. None if use the distribution of the whole original set
    def get_test(self, size=None, distribution = None):
        df = self.__get_test_df(size, distribution)
        return list(df['message']), list(df['sentiment'])

    #  Get a test set, in dataframe format
    def __get_test_df(self, size=None, distribution = None):
        if size is None:
            size = self.test_size
        if size > self.test_size:
            raise Exception("The size %d is large than the max test size %d!" % (size, self.test_size))
        df = self.__get_subset(self.train_size, self.test_size, size, distribution)
        return df

    # Get a validation set
    # size: size of the validation set
    # return: X and y of noisy set
    # distribution: the 'sentiment' distribution. None if use the distribution of the whole original set
    def get_validation(self, size=None, distribution = None):
        if size is None:
            size = self.validation_size
        if size > self.validation_size:
            raise Exception("The size %d is large than the max validation size %d!" % (size, self.validation_size))
        df = self.__get_subset(self.train_size + self.test_size, self.validation_size, size, distribution)
        return list(df['message']), list(df['sentiment'])

    # Get a original noisy set with is generated form the origianl set by mislabeling
    # size: size of the noisy set
    # return: X and y of noisy set
    def get_original_noisy(self, size=None):
        if size is None:
            size = self.noisy_size
        if size > self.noisy_size:
            raise Exception("The size %d is large than the max original noisy size %d!" % (size, self.noisy_size))

        df = self.__get_subset(self.train_size + self.test_size + self.validation_size, self.noisy_size, size)

        # Change labels to make noise
        random.seed(self.rseed)
        df['sentiment'] = list(map(lambda x: (int(x) + random.randint(1, self.class_count-1)) % self.class_count, df['sentiment']))

        return df

    # Get a noisy set which is mixed by the original noisy, irrelevant noisy, translated noisy, etc.
    # size: size of the noisy set
    # return: X and y of noisy set
    def get_noisy(self, size = None):
        if self.dfNoisy is None:
            self.dfNoisy = self.get_original_noisy()
        if size is not None and size > len(self.dfNoisy):
            raise Exception("The size %d is large than the total noisy size %d!" % (size, len(self.dfNoisy)))
        df = sklearn.utils.shuffle(self.dfNoisy, random_state=self.rseed)
        if size is not None:
            df = df[:size]
        return list(df['message']), list(df['sentiment'])

    # Reset the noisy set to the origianl noisy set
    def reset_noisy(self):
        self.dfNoisy    = None

    # Enumate .csv files in a directory
    def __enum_files(self, dir):
        lstFile = []
        ls = os.listdir(dir)
        for fn in ls:
            pos = fn.find('.')
            if (pos >= 1):
                surfix = fn[pos + 1:]
                if(surfix == 'csv'):
                    lstFile.append(fn)
        return lstFile

    # load a dataframe from csv
    def __read_csv_safe(self, file):
        lst = ['UTF-8','unicode_escape','utf-16','gb18030','ANSI','latin',"gbk"]
        for encodeing in lst:
            try:
                df = pd.read_csv(file, encoding=encodeing)
                return df
            except:
                pass
        return pd.read_csv(file)

    # Add the external noisy data to the noisy set
    # None indicates use the same distribution as that of the whole original data
    # noisy_source: name of noisy source, like: irrelevant, translated
    # distribution: distribution of the noisy.
    #               None: use the distribution of original set
    #               "reserve_labels": use the labels of each noisy data.
    # size: size of noisy to be add. None indicates all noisy data will be added
    def add_noisy(self, noisy_source, distribution = None, size = None):
        reserve_labels = False
        if distribution is None:
            distribution = self.distribution
        elif isinstance(distribution, str) and distribution == "reserve_labels":
            reserve_labels = True

        lstMessage     = []
        lstSentiment   = []
        lstTweetid     = []
        dir = './Noisy/' + noisy_source + '/'
        lstFile = self.__enum_files(dir)
        for file in lstFile:
            df = self.__read_csv_safe(dir + file)
            lstMessage  += list(df['message'])
            if reserve_labels:
                lstSentiment   += list(df['sentiment'])
                if 'tweetid' in df.columns.values:
                    lstTweetid  += list(df['tweetid'])

        length = len(lstMessage)
        if not reserve_labels:
            lstMessage = sklearn.utils.shuffle(lstMessage, random_state=self.rseed)
            lstSentiment = []
            for i in range(self.class_count):
                c = int(round(length * distribution[i])) # Use specific distribution to calc the size of every label
                if i == self.class_count - 1:
                    c = length - len(lstSentiment)
                lstSentiment += ([i]*c)
            lstSentiment = sklearn.utils.shuffle(lstSentiment, random_state=self.rseed)

            df = pd.DataFrame(lstSentiment, columns=['sentiment'])
            df['sentiment'] = df['sentiment'].astype("category")
            df['message']   = lstMessage
        else:
            df = pd.DataFrame(lstSentiment, columns=['sentiment'])
            # df['sentiment'] = df['sentiment'].astype("category").cat.codes
            df['sentiment'] = df['sentiment'].astype("int")
            df['sentiment'] = df['sentiment'].apply(lambda x: x+1)
            df['message']   = lstMessage

            if len(lstTweetid) == length:
                # use tweetid to drop samples which exists in test set
                df['tweetid']   = lstTweetid
                dfForDrop   = pd.concat([self.__get_test_df()[['sentiment','message','tweetid']], df])
                dfForDrop   = dfForDrop.drop_duplicates(subset=['tweetid'],keep="first")
                df          = dfForDrop[self.get_test_len():]

            df = sklearn.utils.shuffle(df, random_state=self.rseed)

        df = df[df['message'] != None]
        df.dropna(inplace=True)
        df = df.drop_duplicates(subset=['message'],keep="first")

        if size is not None:
            if size > length:
                raise Exception("The size is large than the size of noisy source!")
            df  = df[:size]

        if 'tweetid' not in df.columns.values:
            df['tweetid']   = [-1] * len(df)

        if self.dfNoisy is None:
            self.dfNoisy = self.get_original_noisy()
        if len(self.dfNoisy):
            self.dfNoisy    = pd.concat([self.dfNoisy, df])
        else:
            self.dfNoisy    = df
        return len(df)

    # Get the train set with noisy data
    # original_size: size of the data from the original train set
    # noisy_size:    size of the noisy data
    # distribution: the 'sentiment' of original train set. None if use the distribution of the whole original set
    def get_train_with_noisy(self, original_size = None, noisy_size = None, distribution = None):
        dfTrain   = self.__get_train_df(original_size, distribution)
        if noisy_size is None:
            return list(dfTrain['message']), list(dfTrain['sentiment'])

        if self.dfNoisy is None:
            self.dfNoisy = self.get_original_noisy()

        dfNoisy  = self.dfNoisy[~self.dfNoisy['tweetid'].isin(dfTrain['tweetid'])]
        if noisy_size > len(dfNoisy):
            raise Exception("Requiring %d noisy data, but only %d available!" % (noisy_size, len(dfNoisy)))
        dfNoisy = sklearn.utils.shuffle(dfNoisy, random_state=self.rseed)  #shuffle
        dfNoisy = dfNoisy[:noisy_size]

        df = pd.concat([dfTrain, dfNoisy])
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

    # Get the size of the whole original noisy set. External noisy not included
    def get_original_noisy_len(self):
        return self.noisy_size

    # Get the size of the whole noisy set, Including the external noisy
    def get_noisy_len(self):
        if self.dfNoisy is None:
            return self.get_original_noisy_len()
        return len(self.dfNoisy)

    # Get the subset with the specific size
    # return: X y of subset
    def get_sub_set(self, x, y, size=None):
        X1, y1, X2, y2 = self.split_set(x, y, size)
        return X1, y1

    # Split a dataset into two parts
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

    # Get the distribution of the whole original set
    def get_original_distribution(self):
        return self.distribution

    @staticmethod
    # Combine X,y as in dataframe format
    def df(Xy):
        return pd.DataFrame({'message':Xy[0] , 'sentiment':Xy[1]})

if __name__ == '__main__':
    def print_distribution(hint, y):
        df = data_center.df((y, y))
        c = df['sentiment'].value_counts(sort = False)
        l = len(df)
        print("%s: %s" % (hint, ("%.1f%%, "*(len(c)-1)+"%.1f%%") % tuple([x*100/l for x in list(c)])))

    #Split the original data set into 3 parts: training set, test set, noisy set
    dc = data_center("twitter_sentiment_data.csv", test_size=4000, noisy_size=3000)
    dc.shuffle(rseed = 522) # Shuffle to setup a new series of experiments
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

    # distribution of training set. None indicates use the same distribution as that of the whole original data
    train_distribution = None

    # distribution of external noisy
    external_noisy_distribution = [0.25, 0.25, 0.25, 0.25]

    # add the external noisy data (irrelevant texts)
    added_size = dc.add_noisy(noisy_source="irrelevant",
                              distribution = external_noisy_distribution, size = None) # max size: 36000
    print("%d noisy samples added" % added_size)

    # add the external noisy data (translated texts)
    added_size = dc.add_noisy(noisy_source="translated",
                              distribution = "reserve_labels", size = None) # max size: 8000
    print("%d noisy samples added" % added_size)

    print("Noisy set new size is %d"                % dc.get_noisy_len())

    for size in [6000, 7500, 12000, 15000, 22500, 30000]:     # training set sizes represented in absolute values
        X_train, y_train = dc.get_train(size, train_distribution)
        print("* Training set size: %5d samples" % (len(y_train)))
        print_distribution("  Sentiment distribution", y_train)

    print("-----------------------------------------------------------------------")
    for size in [(6000, 1500), (12000, 3000), (22500, 7500)]:     # training set sizes represented in absolute values
        X_train, y_train = dc.get_train_with_noisy(size[0], size[1], train_distribution)
        print("* Noisy training set size: %5d samples (%d + %d)" % (len(X_train), size[0], size[1]))
        print_distribution("  Sentiment distribution", y_train)
