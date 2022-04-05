# %% [markdown]
# # SVM - Climate Sentiment Multiclass Classification
# ## CS522 Project

# %% [markdown]
# ### Dataset: 
# https://www.kaggle.com/code/luiskalckstein/climate-sentiment-multiclass-classification

# %% [markdown]
# ### Imports

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from Common.DataCenter import data_center
from Common.LSI import SKLearnLSA



# %% [markdown]
# ### Text preprocessing

# %%
# parameter: original X of training set and test set
# return:  vectorised X of training set and test set
def text_preprocessing(X_train, X_test):
    # Convert texts to vectors
    #lsiVectorizer = LSI()
    #lsiVectorizer.BuildModel(X_train)
    #X_train_vec =  lsiVectorizer.QueryList2LatentSpace(X_train)
    #X_test_vec =  lsiVectorizer.QueryList2LatentSpace(X_test)
    #vectorizer   = TfidfVectorizer()
    #X_train_vec  = vectorizer.fit_transform(X_train)
    #X_test_vec   = vectorizer.transform(X_test)
    lsa = SKLearnLSA()
    lsa.BuildModel(X_train + X_test, 300)
    X_train_vec = lsa.Query2LatentSpace(X_train)
    X_test_vec = lsa.Query2LatentSpace(X_test)
    return X_train_vec, X_test_vec


# %% [markdown]
# ### One-hot encoding, convert the labels to vectors (4 x 1) each

# %%
# parameter: original y of training set, original y of test set
# return:  encoded y of training set and test set
def one_hot_encoding(y_train, y_test):
    mlb          = MultiLabelBinarizer()
    y_train_vec  = mlb.fit_transform(map(str, y_train))
    y_test_vec   = mlb.transform(map(str, y_test))
    return y_train_vec, y_test_vec


# %% [markdown]
# ### Run SVM and evaluate the results

# %%
# parameter:  vectorised X and encoded y of training set and test set
def evaluate_SVM(X_train_vec, y_train_vec, X_test_vec, y_test_vec):
    # Run SVM - fit and predict
    SVM             = OneVsRestClassifier(LinearSVC(), n_jobs=-1)
    SVM.fit(X_train_vec, y_train_vec)
    prediction      = SVM.predict(X_test_vec)

    # Evaluate the results
    macro_f1        = f1_score(y_test_vec, prediction, average='macro')
    weighted_f1     = f1_score(y_test_vec, prediction, average='weighted')
    macro_precision = precision_score(y_test_vec, prediction, average='macro')
    macro_recall    = recall_score(y_test_vec, prediction, average='macro')

    return macro_f1, weighted_f1, macro_precision, macro_recall


# %% [markdown]
# ### Do an experiment

# %%
# Parameter: original X,y of training set and test set
def do_experiment(X_train, y_train, X_test, y_test):
    # Convert texts to vectors
    X_train_vec, X_test_vec = text_preprocessing(X_train, X_test)
    y_train_vec, y_test_vec = one_hot_encoding(y_train, y_test)

    # Run SVM and evaluate the results
    macro_f1, weighted_f1, macro_precision, macro_recall = \
        evaluate_SVM(X_train_vec, y_train_vec, X_test_vec, y_test_vec)

    # Show the indicators
    print(" macro_f1: %.4f , weighted_f1: %.4f, macro_precision: %.4f, macro_recall: %.4f" %
          (macro_f1, weighted_f1, macro_precision, macro_recall))
    return X_train_vec


# %% [markdown]
# ### Main entry

# %% [markdown]
# **Load the database and split it into training set, test set, noisy set, validation set**

# %%
dc = data_center("twitter_sentiment_data.csv", test_size=8000, noisy_size=8000, validation_size=5000)

print("####################################################")
print("Total data size: ",       dc.get_len())
print("Total train data size: ", dc.get_train_len())
print("Total test data size: ",  dc.get_test_len())

# %% [markdown]
# **Get the test set for evaluation**

# %%
X_test, y_test = dc.get_test()


# %% [markdown]
# **Run experiments with different training sets, and use the same test set.**

# %%
print("-----------------------------------------------")
for size in [2000, 2500, 4000, 5000, 7500, 10000]:
    # Get a training set without noisy data
    X_train, y_train = dc.get_train(size)
    print("Training set size: %d samples (%.1f%%): " % (len(X_train), len(y_train)/dc.get_train_len()*100))

    # Do an experiment
    do_experiment(X_train, y_train, X_test, y_test)

print("-----------------------------------------------")
xtrainvec = None
for size in [(2000, 500), (4000, 1000), (7500, 2500)]:
    # Get a noisy training set
    X_train, y_train = dc.get_train_with_noisy(size[0], size[1])
    print("Noisy training set size: %d samples (%d original, %d noisy)" % (len(y_train), size[0], size[1]))

    # Do an experiment
    xtrainvec = do_experiment(X_train, y_train, X_test, y_test)

# %%
X_train, y_train = dc.get_train(2000)
# Convert texts to vectors
X_train_vec, X_test_vec = text_preprocessing(X_train, X_test)
y_train_vec, y_test_vec = one_hot_encoding(y_train, y_test)

# Run SVM and evaluate the results
macro_f1, weighted_f1, macro_precision, macro_recall = \
    evaluate_SVM(X_train_vec, y_train_vec, X_test_vec, y_test_vec)

# Show the indicators
print(" macro_f1: %.4f , weighted_f1: %.4f, macro_precision: %.4f, macro_recall: %.4f" %
        (macro_f1, weighted_f1, macro_precision, macro_recall))

