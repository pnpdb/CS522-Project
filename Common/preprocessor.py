import re
import string
import nltk
from nltk.stem.porter import PorterStemmer
from regex import R
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

def tweet_preprocessor(tweet):

    '''
    Takes in tweet and cleans it by removing line breaks,
    URL's, numbers, capital letters, and punctuation.
    '''

    tweet = tweet.replace('\n', ' ') # remove line breaks
    tweet = re.sub('[%s]' % re.escape(string.punctuation), ' ', tweet.lower()) # remove capital letters and punctuation
    
    return tweet

# parameter: list of tweet messages
# return: normalization of pre-processing
def normalize_preprocessing(data):
    
    messages = []
    
    # Traversal the message list
    for i in range(len(data)):
        # Lower case
        message         = data[i].lower()
        # pre-process to tweet corpus
        message         = tweet_preprocessor(message)    
        # Tokenize
        message         = nltk.word_tokenize(message)  
        # Stemming
        ps               = PorterStemmer()
        message_filtered = [ps.stem(w) for w in message]
        # Re-Combinate
        message      = ' '.join(message_filtered)

        messages.append(message)
        
    return messages

# Text preprocessing
# parameter: original X of training set and test set
# return:  vectorised X of training set and test set
def text_preprocessing_tfidf(X_train, X_test):
    # preprocessing with traditional NLP methodology
    X_train_normalized = normalize_preprocessing(X_train)
    X_test_normalized  = normalize_preprocessing(X_test)

    # Convert texts to vectors
    vectorizer   = TfidfVectorizer(ngram_range=(1,2))
    X_train_vec  = vectorizer.fit_transform(X_train_normalized)
    X_test_vec   = vectorizer.transform(X_test_normalized)
    return X_train_vec, X_test_vec

# One-hot encoding, convert the labels to vectors (4 x 1) each
# parameter: original y of training set, original y of test set
# return:  encoded y of training set and test set
def one_hot_encoding(y_train, y_test):
    mlb          = MultiLabelBinarizer()
    y_train_vec  = mlb.fit_transform(map(str, y_train))
    y_test_vec   = mlb.transform(map(str, y_test))
    return y_train_vec, y_test_vec
