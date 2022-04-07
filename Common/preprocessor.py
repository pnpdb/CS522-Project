import re
import string
import nltk
from nltk.stem.porter import PorterStemmer


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
