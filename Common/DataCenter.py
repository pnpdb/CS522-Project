import pandas as pd
from sklearn.model_selection import train_test_split

class data_center():
    def __init__(self, filename, test_size, noisy_size): #file name, size of test set, size of noisy set
        self.rseed = 42
        df = pd.read_csv(filename, encoding='latin-1')
        df['encoded_cat'] = df.sentiment.astype("category").cat.codes
        df = df[df['message'] != None]
        df.dropna(inplace=True)
        self.df = df

        y = list(df['encoded_cat'])
        X = list(df['message'])

        X1, self.X_test, y1, self.y_test = \
            train_test_split(X, y, test_size=test_size, random_state=self.rseed, stratify=y)
        self.X_train, self.X_noisy, self.y_train, self.y_noisy = \
            train_test_split(X1, y1, test_size=noisy_size / (1-test_size), random_state=self.rseed, stratify=y1)

    # Get test set
    def get_test(self, size = None):
        return self.__get_sub_set(self.X_test, self.y_test, size)
        
    # get test data with fixed size
    def get_test_fixedsize(self, intSize):
        return self.get_test(intSize/self.get_test_len())


    # Get noisy set
    def get_noisy(self, size = None):
        # Labels not changed, to be done...
        return self.__get_sub_set(self.X_noisy, self.y_noisy, size)

    # Get train set
    def get_train(self, size = None):
        return self.__get_sub_set(self.X_train, self.y_train, size)
    
    # get train data with fixed size
    def get_train_fixedsize(self, intSize):
        return self.get_train(intSize/self.get_train_len())


    def get_len(self):
        return len(self.df)

    def get_train_len(self):
        return len(self.y_train)

    def get_test_len(self):
        return len(self.y_test)

    def get_noisy_len(self):
        return len(self.y_noisy)

    def __get_sub_set(self, x, y, size = None):
        if size is None or size >= 1:
            return x, y
        X1, X2, y1, y2 =\
            train_test_split(x, y, test_size=1-size, random_state=self.rseed, stratify=y)
        return X1, y1

if __name__ == '__main__':
    dc = data_center("twitter_sentiment_data.csv", test_size = 0.2, noisy_size = 0.2)
    X_test,  y_test = dc.get_test()
    X_noisy, y_noisy    =dc.get_noisy()
    X_train_100, y_train_100 = dc.get_train()
    X_train_50, y_train_50 = dc.get_train(0.50)
    X_train_30, y_train_30 = dc.get_train(0.30)
    X_train_20, y_train_20 = dc.get_train(0.20)
    X_train_10, y_train_10 = dc.get_train(0.10)
    print(len(X_noisy), len(X_test), len(X_train_100), len(X_train_50), len(X_train_30), len(X_train_20), len(X_train_10))
