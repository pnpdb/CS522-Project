import pandas as pd
import os
import random
import sklearn
from sklearn.model_selection import train_test_split
from IPython.display import display

# A class manager the data, which loads the data, generates a test set and train sets with different sizes,
# and add noisy data to these train sets.
class data_center():
    noise_sources   = {"none":0, "mislabeled":1, "irrelevant":2, "translated":3}

    # parameter: file name, size of the test set, size of the noisy set
    def __init__(self, filename, test_size, noisy_size, validation_size = 0, train_size = 20000, do_verb=0):
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
        self.class_count = len(df['sentiment'].value_counts(sort=False))

        if isinstance(noisy_size, dict):
            noisy_set_sizes  = noisy_size
            if 'mislabeled' in noisy_set_sizes.keys():
                self.noisy_size = noisy_size['mislabeled'][0]
            else:
                self.noisy_size = 0
        else:
            noisy_set_sizes  = None
            self.noisy_size  = noisy_size

        self.dfOriginal         = df                    # let the cleaned set be the original set
        self.shuffle(rseed = 522)                       # shuffle the original set
        self.train_size         = train_size if train_size is not None else len(df) - test_size - noisy_size - validation_size
        self.test_size          = test_size
        self.validation_size    = validation_size

        if self.train_size + self.test_size + self.validation_size + self.noisy_size > len(df):
            raise Exception("The sum of the size of all sets is bigger than the size of the whole data set!" % (size, self.train_size))

        self.dfNoisy            = None                  # the noisy set
        self.noise_source_distribution  = None

        # the distribution of the whole original set
        # self.distribution       = [x/len(df) for x in list(df['sentiment'].value_counts(sort = False))]
        self.distribution       = [list(df['sentiment']).count(x)/len(df) for x in range(self.class_count)]

        self.noise_source_distribution = (1,0,0)

        if noisy_set_sizes:
            # Prepare noisy data sets
            if 'mislabeled' in noisy_set_sizes.keys() and noisy_set_sizes['mislabeled'][0] > 0:
                if do_verb:
                    print("%d noisy samples of '%s' added" % (self.get_noisy_len(), 'mislabeled'))

            # add the external noisy data (irrelevant texts)
            if 'irrelevant' in noisy_set_sizes.keys() and noisy_set_sizes['irrelevant'][0] > 0:
                # distribution of irrelevant noisy
                added_size = self.add_noisy(noisy_source="irrelevant", distribution = noisy_set_sizes['irrelevant'][1],
                                          size = noisy_set_sizes['irrelevant'][0])
                if do_verb:
                    print("%d noisy samples of '%s' added"  % (added_size, 'irrelevant'))

            # add the external noisy data (translated texts). use the labels of each noisy data
            if 'translated' in noisy_set_sizes.keys() and noisy_set_sizes['translated'][0] > 0:
                added_size = self.add_noisy(noisy_source="translated", distribution = noisy_set_sizes['translated'][1],
                                          size = noisy_set_sizes['translated'][0])
                if do_verb:
                    print("%d noisy samples of '%s' added"  % (added_size, 'translated'))

            if do_verb:
                print("The total size of noisy data is %d"                % self.get_noisy_len())


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
    def get_train_df(self, size=None, distribution = None):
        if size is None:
            size = self.train_size
        if size > self.train_size:
            raise Exception("The size %d is large than the max train size %d!" % (size, self.train_size))
        df = self.__get_subset(0, self.train_size, size, distribution)
        df['noise_text'] = "none"
        return self.__add_noise_id_column(df)

    # Get the train set
    # size: size of the train set
    # return: X and y of training set
    # distribution: the 'sentiment' distribution. None if use the distribution of the whole original set
    def get_train(self, size=None, distribution = None):
        df = self.get_train_df(size, distribution)
        return list(df['message']), list(df['sentiment'])

    # Get a test set
    # size: size of the test set
    # return: X and y of test set
    # distribution: the 'sentiment' distribution. None if use the distribution of the whole original set
    def get_test(self, size=None, distribution = None):
        df = self.get_test_df(size, distribution)
        return list(df['message']), list(df['sentiment'])

    #  Get a test set, in dataframe format
    def get_test_df(self, size=None, distribution = None):
        if size is None:
            size = self.test_size
        if size > self.test_size:
            raise Exception("The size %d is large than the max test size %d!" % (size, self.test_size))
        df = self.__get_subset(self.train_size, self.test_size, size, distribution)
        df['noise_text'] = "none"
        return self.__add_noise_id_column(df)

    @staticmethod
    def noise_id(noise_text):
        return data_center.noise_sources[noise_text]

    @staticmethod
    def noise_text(noise_id):
        for k,v in data_center.noise_sources:
            if v == noise_id:
                return k
        return data_center.noise_sources.keys[0]

    def __add_noise_id_column(self, df):
        df['noise'] = df['noise_text'].apply(lambda x: int(data_center.noise_id(x)))
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
        df['origin'] = df['sentiment']
        df['sentiment'] = list(map(lambda x: (int(x) + random.randint(1, self.class_count-1)) % self.class_count, df['sentiment']))
        df['noise_text'] = "mislabeled"

        return df

    # Get a noisy set which is mixed by the original noisy, irrelevant noisy, translated noisy, etc.
    # size: size of the noisy set
    # return: X and y of noisy set
    def get_noisy_df(self, size = None):
        if self.dfNoisy is None:
            self.dfNoisy = self.get_original_noisy()
        if size is not None and size > len(self.dfNoisy):
            raise Exception("The size %d is large than the total noisy size %d!" % (size, len(self.dfNoisy)))
        df = sklearn.utils.shuffle(self.dfNoisy, random_state=self.rseed)
        if size is not None:
            df = df[:size]
        return self.__add_noise_id_column(df)

    def get_noisy(self, size = None):
        df = self.get_noisy_df(size)
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
    #               0~1: ratio to mislabel the data(if has label), 0 is same as "reserve_labels", 1 means mialabel all
    # size: size of noisy to be add. None indicates all noisy data will be added
    def add_noisy(self, noisy_source, distribution = None, size = None):
        if distribution is None:
            distribution = self.distribution
        elif isinstance(distribution, list) or isinstance(distribution, tuple):
            pass
        elif isinstance(distribution, str) and distribution == "reserve_labels":
            mis_ratio = 0
            distribution = None
        else:
            mis_ratio = float(distribution)
            distribution = None

        lstMessage     = []
        lstSentiment   = []
        lstTweetid     = []
        dir = './Noisy/' + noisy_source + '/'
        lstFile = self.__enum_files(dir)
        for file in lstFile:
            df = self.__read_csv_safe(dir + file)
            lstMessage  += list(df['message'])
            if distribution is None:    #donot use specific distribution
                if 'sentiment' in df.columns.values:
                    lstSentiment    += list(df['sentiment'])
            if 'tweetid' in df.columns.values:
                lstTweetid      += list(df['tweetid'])

        length = len(lstMessage)

        df = pd.DataFrame(lstMessage, columns=['message'])
        if len(lstTweetid) == length:
            df['tweetid']   = lstTweetid

        if len(lstSentiment):   # if source data have labels, save to 'origin'
            df['origin'] = lstSentiment
            df['origin'] = df['origin'].astype("int")
            df['origin'] = df['origin'].apply(lambda x: x+1)
            random.seed(self.rseed)
            df = sklearn.utils.shuffle(df, random_state=self.rseed)

            mis_len = int(length*mis_ratio)
            _ = df['origin'][:mis_len]
            _ = list(map(lambda x: (int(x) + random.randint(1, self.class_count-1)) % self.class_count, _))
            df['sentiment'] = list(_) + list(df[mis_len:]['origin'])
        else:  # if source data have no label, generate randomly according to the distribution specified
            for i in range(self.class_count):
                c = int(round(length * distribution[i])) # Use specific distribution to calc the size of every label
                if i == self.class_count - 1:
                    c = length - len(lstSentiment)
                lstSentiment += ([i]*c)
            df['sentiment'] = lstSentiment
            df['sentiment'] = df['sentiment'].astype("int")
            df['message']   = lstMessage

        # else:
        #     df = pd.DataFrame(lstSentiment, columns=['sentiment'])
        #     df['sentiment'] = df['sentiment'].astype("int")
        #     df['sentiment'] = df['sentiment'].apply(lambda x: x+1)
        #     df['origin']    = df['sentiment']
        #     df['message']   = lstMessage

        if len(lstTweetid) == length:
            # use tweetid to drop samples which exists in test set
            dfForDrop   = pd.concat([self.get_test_df()[['sentiment','message','tweetid']], df])
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

        if 'origin' not in df.columns.values:
            df['origin']   = None

        df['noise_text'] = noisy_source

        if self.dfNoisy is None:
            self.dfNoisy = self.get_original_noisy()
        if len(self.dfNoisy):
            self.dfNoisy    = pd.concat([self.dfNoisy, df])
        else:
            self.dfNoisy    = df

        dftmp = self.__add_noise_id_column(self.dfNoisy )
        self.noise_source_distribution = tuple([list(dftmp['noise']).count(x)*1./len(dftmp)
                                            for x in set(data_center.noise_sources.values())-set([0])])

        return len(df)

    # Get the train set with noisy data
    # original_size: size of the data from the original train set
    # noisy_size:    size of the noisy data
    # distribution: the 'sentiment' of original train set. None if use the distribution of the whole original set
    # return : A dataframe with columns : message, sentiment, tweetid, noise
    def get_train_with_noisy_df(self, original_size = None, noisy_size = None, distribution = None):
        dfTrain   = self.get_train_df(original_size, distribution)
        if noisy_size is None:
            return self.__add_noise_id_column(dfTrain)

        if self.dfNoisy is None:
            self.dfNoisy = self.get_original_noisy()

        # Noise should not be already in training set.
        # "reserve_labels". Currently, noise_text=translated
        random.seed(noisy_size+self.rseed)
        dfNoisy12 = self.dfNoisy[self.dfNoisy['noise_text']!='translated']
        dfNoisy12 = sklearn.utils.shuffle(dfNoisy12, random_state=self.rseed)

        small_delta = 0.02  # let the distribtions to fluctuate in a samll range
        translated_size = int(noisy_size * self.noise_source_distribution[2] * random.uniform(1-small_delta,1+small_delta))
        translated_size = max(translated_size, noisy_size-len(dfNoisy12))

        dftmp     = self.dfNoisy[self.dfNoisy['noise_text']=='translated']
        dfNoisy3  = dftmp[~dftmp['tweetid'].isin(dfTrain['tweetid'])]
        if translated_size * (1-small_delta-0.01) < len(dfNoisy3) < translated_size:
            translated_size = len(dfNoisy3)
        if(len(dfNoisy3) < translated_size):
            raise Exception("Requiring %d no conflict translated noisy data, but only %d available!"
                            % (translated_size, len(dfNoisy3)))
        dfNoisy3  = sklearn.utils.shuffle(dfNoisy3, random_state=self.rseed)
        dfNoisy3  = dfNoisy3[:translated_size]

        dfNoisy12 = dfNoisy12[:noisy_size-translated_size]
        dfNoisy   = pd.concat([dfNoisy12, dfNoisy3])
        if noisy_size > len(dfNoisy):
            raise Exception("Requiring %d noisy data, but only %d available!" % (noisy_size, len(dfNoisy)))

        dfNoisy = sklearn.utils.shuffle(dfNoisy, random_state=self.rseed)
        dfNoisy = self.__add_noise_id_column(dfNoisy)
        dfNoisy = dfNoisy[:noisy_size]

        # dfNoisy = dfNoisy.reset_index()
        # split = StratifiedShuffleSplit(n_splits=1, test_size=noisy_size, random_state=self.rseed)
        # for train_index, test_index in split.split(dfNoisy, dfNoisy["noise"]):
        #     dfNoisy = dfNoisy.loc[test_index,]
        #     break

        df = pd.concat([dfTrain, dfNoisy])
        df = sklearn.utils.shuffle(df, random_state=self.rseed)  #shuffle

        return self.__add_noise_id_column(df).reset_index()

    # Similary as get_train_with_noisy_df, but return X, y
    def get_train_with_noisy(self, original_size = None, noisy_size = None, distribution = None):
        df = self.get_train_with_noisy_df(original_size, noisy_size, distribution)
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

    def get_noise_source_distribution(self):
        return self.noise_source_distribution

    def print_noise_source_distribution(self, hint = ""):
        data_center.show_distribution(hint, tuple(data_center.noise_sources.keys())[1:], self.noise_source_distribution)

    def print_summary(self):
        print("###################################### Data Summary #############################################")
        # print("  Raw set (not cleaned) size: %d"    % self.get_raw_len())
        print("  Total data size: %d"             % self.get_len())
        data_center.show_distribution("      sentiments", ['Anti','Neutral','Pro','News'], self.distribution)
        print("  Training data size: %d"             % self.get_train_len())
        print("  Test data data: %d"                 % self.get_test_len())
        print("  Noisy data data: %d"                % self.get_noisy_len())
        if len(self.get_validation()[0]):
            print("  Validation data size: %d"           % len(self.get_validation()[0]))
        self.print_noise_source_distribution("      noise sources")
        print("##################################################################################################")

    @staticmethod
    # Combine X,y as in dataframe format
    def df(Xy):
        return pd.DataFrame({'message':Xy[0] , 'sentiment':Xy[1]})

    @staticmethod
    # Split df to two columns
    def Xy(df):
        return list(df['message']), list(df['sentiment'])

    @staticmethod
    # distribution of sentiments
    def print_distribution(hint, y, print_flag=True):
        dist = data_center.calc_distribution(y, 'sentiment', [0,1,2,3])
        # dist = tuple([x*100 for x in dist])
        # df = data_center.df((y, y))
        # class_count = len(df['sentiment'].value_counts(sort = False))
        # dist0 = tuple([list(df['sentiment']).count(x)/len(df) for x in range(class_count)])
        # dist = tuple([x*100 for x in dist0])
        if print_flag:
            data_center.show_distribution(hint, ['Anti','Neutral','Pro','News'], dist)
        return dist

    @staticmethod
    def calc_distribution(y, column="sentiment", labels=[0,1,2,3]):
        if not isinstance(y, pd.DataFrame):
            df = data_center.df((y, y))
        else:
            df = y
        dist = tuple([list(df[column]).count(x)/len(df) for x in labels])
        return dist

    @staticmethod
    def calc_distribution_str(y, column="sentiment", labels=[0,1,2,3]):
        dist = data_center.calc_distribution(y, column, labels)
        return data_center.distribution2str(None, dist, len(labels))

    @staticmethod
    # for general distribution
    def show_distribution(hint, keys, distribution):
        print( "%s %s: %s" % (hint, str(tuple(keys)),
                              ("%.1f%%, "*(len(keys)-1)+"%.1f%%") %
                              tuple([x*100 for x in distribution])))

    @staticmethod
    # conver distribution to a string
    def distribution2str(hint, distribution, class_count):
        return  "%s[%s]" % ("" if hint is None else hint,
            ("%.1f%%, "*(class_count-1)+"%.1f%%") % tuple([x*100 for x in distribution]))

    @staticmethod
    def print_data(df):
        df = df.copy()
        df['tweetid...']  = df['tweetid'].apply(lambda x: str(int(x/100000000)) if x > 0 else "-")
        df['message...']  = df['message'].apply(lambda x: x[:30])
        if 'origin' not in df.columns.values:
            df['origin(sentiment)'] = "-"
        else:
            df['origin(sentiment)'] = df['origin'].apply(lambda x: str(int(x)) if pd.notnull(x) else "-")

        df = df[['noise','noise_text','sentiment','origin(sentiment)','tweetid...','message...']]
        if(data_center.is_ipython()):
            display(df)
        else:
            print(df.to_string(index=False))

    @staticmethod
    # if in ipython
    def is_ipython():
        try:
            __IPYTHON__
            return True
        except NameError:
            return False

if __name__ == '__main__':
    print("\nPrepare data:")
    print("----------------------------------------------------------------------------------")

    #Split the original data set into 3 parts: training set, test set, noisy set
    dc = data_center("twitter_sentiment_data.csv", test_size=3000, noisy_size=6000)
    dc.shuffle(rseed = 522) # Shuffle to setup a new series of experiments

    # distribution of training set. None indicates use the same distribution as that of the whole original data
    train_distribution = None

    # distribution of external noisy
    external_noisy_distribution = [0.25, 0.25, 0.25, 0.25]

    # add the external noisy data (irrelevant texts)
    added_size = dc.add_noisy(noisy_source="irrelevant",
                              distribution = external_noisy_distribution, size = 6000) # max size: 36000
    print("%d irrelevant noisy samples added" % added_size)

    # add the external noisy data (translated texts)
    added_size = dc.add_noisy(noisy_source="translated",
                              distribution = "reserve_labels", size = 6000) # max size: 8000
    print("%d translated noisy samples added" % added_size)

    # Show the summary of the whole data
    dc.print_summary()

    # A demo to show the data features
    train_df = dc.get_train_with_noisy_df(1000,1000)
    data_center.print_data(train_df.head(10))

    print("\nGenerate training set with different sizes:")
    print("----------------------------------------------------------------------------------")
    for size in [2000, 4000, 5000, 8000, 10000, 15000, 20000]:     # training set sizes represented in absolute values
        X_train, y_train = dc.get_train(size, train_distribution)
        print("* Training set size: %5d samples" % (len(y_train)))
        data_center.print_distribution("  Sentiments", y_train)

    print("----------------------------------------------------------------------------------")
    for size in [(4000, 14000), (8000, 2000), (15000, 5000)]:     # training set sizes represented in absolute values
        train_df         = dc.get_train_with_noisy_df(size[0], size[1], train_distribution)
        X_train, y_train = data_center.Xy(train_df)

        # data_center.print_data(train_df.head())

        print("* Noisy training set size: %5d samples (%d + %d)" % (len(X_train), size[0], size[1]))
        data_center.print_distribution("  Sentiments", y_train)
        dc.print_noise_source_distribution("  Noise sources")
