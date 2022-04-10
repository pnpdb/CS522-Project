# import nltk
# nltk.download()  # Models - punkt, Punkt tokenizer Models

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plot
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, TrainingArguments, Trainer, TFDistilBertForSequenceClassification, TFTrainer, TFTrainingArguments
from Common.DataCenter import data_center
from Common.preprocessor import normalize_preprocessing as normalize
import time

class_names = ['Anti', 'Neutral', 'Pro', 'News']
class_types = [0, 1, 2, 3]
show_confusion_matrix = False
show_plot = False


def get_data(file_name, train_size, validation_size, test_size, noisy_size, name='', train_distribution=None, noisy_distribution=None):
    ''' Load and preprocess data
    Load file from file and remove empty message, drop NaN data, drop duplicate data and shuffle data finally.

    Args:
        file_name
        validation_size
        test_size
        noisy_size

    Returns:
        Return different data sets (train, validate, test, mixed_noisy, raw_noisy) which has tow columns(message,sentiment)
    '''
    dc = data_center(file_name, test_size, noisy_size, validation_size)
    dc.add_noisy(noisy_source="irrelevant", distribution=noisy_distribution, size=noisy_size)

    noisy_distr = None # [0.25, 0.25, 0.25, 0.25]
    if(noisy_size==0):
        X_train, y_train = dc.get_train(train_size, train_distribution)
    else:
        X_train, y_train = dc.get_train_with_noisy(train_size, noisy_size, noisy_distr)

    X_validate, y_validate = dc.get_validation()
    X_test, y_test = dc.get_test()
    X_noisy, y_noisy = dc.get_noisy()

    dists = data_center.print_distribution('', y_train, False)
    dist = {class_names[i]:"%.1f%%"%dists[i] for i in range(len(dists))}
    print('%s, TrainSet:%s, ValidateSet:%s, TestSet:%s, NoisySet:%s, SentimentDistribution(Training): %s' % (name, len(X_train), len(X_validate), len(X_test), len(X_noisy), dist))

    return X_train, y_train, X_validate, y_validate, X_test, y_test, X_noisy, y_noisy


def tokenizer(X_train,  y_train, X_test, y_test, params={}):
    ''' Tokenizie the words of the dataset(lecture 4-NLP/5-LSA/5-NLP)
    Format/language stripping
    Tokenization(language: Chinese and Japanese, Accents, language-specific, Ambiguous)
    Normalization(right-to-left, Date, Alphabet)
    Punctuation
    Numbers
    Case folding
    Thesauri and soundex
    Stop Words Removal
    *Lemmatization
    Stemming(Porter’s algorithm, Word sense Disambiguation, Machine Translation, Accent Restoration in Spanish & French, Capitalization Restoration, Text-to-Speech Synthesis, Spelling Correction) 
    Dimensionality Reduction
    Latent Semantic Analysis
    Docment:
    - Sequence Labeling as Classification/Forward Classification/Backward Classification
    - Part of Speech Tagging(Maximum Entropy Markov Model)

    Word
    - Named Entity Recognition
    - Information Extraction/The Semantic Web

    Text Annotation
    - Labeled Dependency Parsing/Dependency Trees

    pLSA/LDA
    *Word2Vec/Word2Vec Neural Networks
    Neural Networks
    BERT Deep Learning Network(BERT Word Embeddings/Text Classification/Transformer)
    '''
    # Normalization
    x_train_normalized = X_train
    x_test_normalized = X_test
    if(params["tokenizer"] == True):
        x_train_normalized = normalize(X_train)
        x_test_normalized = normalize(X_test)

    # One-hot-encoding
    y_train_encoded = y_train
    y_test_encoded = y_test
    if(params["one_hot_encoding"] == True):
        mlb = MultiLabelBinarizer()
        y_train_encoded  = mlb.fit_transform(map(str, y_train))
        y_test_encoded = mlb.transform(map(str, y_test))

    return x_train_normalized, y_train_encoded, x_test_normalized, y_test_encoded


def svm_fit(x_train, y_train, x_test, y_test, kernel='linear', dataset_type='train', params={}):
    '''Fit the data by using SVM with linear kernel.'''
    if(len(x_train) == 0):
        return None

    t1 = time.time()
    # Preprocess
    x_train, y_train, x_test, y_test = tokenizer(x_train, y_train, x_test, y_test, params)

    vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True)
    x_train_vectors = vectorizer.fit_transform(x_train)
    x_test_vectors = vectorizer.transform(x_test)
    t2 = time.time()

    # Train the data
    if (not hasattr(y_train, "shape")):
        model = svm.SVC(kernel=kernel)
        # model = OneVsRestClassifier(svm.SVC(kernel=kernel), n_jobs=-1)
    else:
        model = OneVsRestClassifier(LinearSVC(dual=False, class_weight='balanced'), n_jobs=-1)
    # model = OneVsRestClassifier(svm.SVC(kernel=kernel), n_jobs=-1)

    model.fit(x_train_vectors, y_train)
    t3 = time.time()

    # Predict the test data
    y_predict = model.predict(x_test_vectors)
    t4 = time.time()

    # Report the accurency
    report = plot_report(y_test, y_predict, {"name": "SVM-"+kernel, "data_type": dataset_type, "eval_loss": 0, "preprocess": round(t2-t1,3), "train": round(t3-t2,3), "predict": round(t4-t3,3)})
    if (not hasattr(report,"accuracy")):
        score = model.score(x_test_vectors, y_test)
        report["accuracy"] = score

    return report


def bert_fit(x_train, y_train, x_test, y_test, dataset_type='train', params={}):
    '''Fit the data by using BERT. '''

    if(len(x_train) == 0):
        return None

    # Tokenizer the train data and test data
    t1 = time.time()
    # Preprocess
    x_train, y_train, x_test, y_test = tokenizer(x_train, y_train, x_test, y_test, params)

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    x_train_tokens = tokenizer(list(x_train), truncation=True, padding=True)
    x_test_tokens = tokenizer(list(x_test), truncation=True, padding=True)
    t2 = time.time()

    train_dataset = tf.data.Dataset.from_tensor_slices((dict(x_train_tokens), y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((dict(x_test_tokens), y_test))

    # Prepare the args
    training_args = TFTrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=2,              # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=16,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.1,                # strength of weight decay
        logging_steps=100,
        eval_steps=10,
    )

    report = None
    with training_args.strategy.scope():
        trainer_model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=4)

        model = TFTrainer(
            model=trainer_model,                 # the instantiated Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            eval_dataset=test_dataset,           # evaluation dataset
        )

        # Train the data
        model.train()
        t3 = time.time()

        # Predict the test data
        predictions, y_predict, metrics = model.predict(test_dataset)
        t4 = time.time()

        # Report the accurency
        report = plot_report(y_test, y_predict, {"name": "BERT", "data_type": dataset_type, "eval_loss": metrics["eval_loss"], "preprocess": round(t2-t1, 3), "train": round(t3-t2, 3), "predict": round(t4-t3, 3)})

    return report


def plot_report(y_test, y_predict, args=None):
    report = classification_report(y_test, y_predict, output_dict=True, target_names=class_names, digits=3, zero_division=1)
    report["args"] = args
    if (show_confusion_matrix == True):
        cm = confusion_matrix(y_test, y_predict, labels=class_types)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_types)
        disp.plot()
        plot.show()
    return report


def get_report_summary(report):
    macro_avg = report["macro avg"]
    args = report["args"]
    report_simple = {
        "Name": args["name"],
        "DataType": args["data_type"],
        "TrainSize": macro_avg["support"],
        "TestSize": 0,
        "Accuracy": report["accuracy"],
        "Loss": args["eval_loss"],
        "Macro-F1": macro_avg["f1-score"],
        "Macro-precision": macro_avg["precision"],
        "Macro-recall": macro_avg["recall"],
        "Macro-F1":  report["weighted avg"]["f1-score"],
        "Time-Preprocess(s)": args["preprocess"],
        "Time-Train(s)": args["train"],
        "Time-Predict(s)": args["predict"]
    }
    return report_simple

def plot_item(report, xvalue_name, yvalue_name, title_value_name='', plot_data_type=["train"]):
    # Plot the firgues
    names = [item["Name"] for item in  report["train_report"] ]
    names = np.unique(names)

    data_types = [item["DataType"] for item in report["train_report"]]
    data_types = np.unique(data_types)
    data_size = None

    for dt in data_types:
        if dt not in plot_data_type:
            continue
        for name in names:
            data = [item for item in report["train_report"] if item["Name"] == name and item["DataType"] == dt]
            x = [item[xvalue_name] for item in data]
            y = [item[yvalue_name] for item in data]
            if data_size == None:
                data_size = x
            plot.plot(x, y, label=name+'-'+dt)

    plot.title('%s %s with NoisySet %s' % (report["model"], yvalue_name,report[title_value_name] ))
    plot.xticks(data_size)
    plot.xlabel(xvalue_name)
    plot.ylabel(xvalue_name)
    plot.legend()
    plot.show()


def summary(result):
    '''Plot and print the summary of the result.'''
    report = {}
    train_reports = []
    validate_reports = []
    train_reports_detail = []
    validate_reports_detail = []
    report["noisy_size"] = []
    for i in range(len(result)):
        if i == 0:
            report["model"] = result[i]["model"]
            report["index"] = result[i]["index"]
            report["title"] = "## %s %s-Noisy %s" % (report["index"], report["model"], report["noisy_size"])

        report["noisy_size"].append(result[i]["noisy"])
        train_report = get_report_summary(result[i]["train_report"])
        validate_report = get_report_summary(result[i]["validate_report"])
        train_report["TrainSize"] = result[i]["train"]
        validate_report["TestSize"] = result[i]["test"]
        train_reports.append(train_report)
        validate_reports.append(validate_report)

        train_reports_detail.append(result[i]["train_report"])
        validate_reports_detail.append(result[i]["validate_report"])

    train_reports.sort(key=lambda r: r["TrainSize"]*100000+ r["Accuracy"])
    validate_reports.sort(key=lambda r: r["TrainSize"]*100000 + r["Accuracy"])
    report['train_report'] = train_reports
    report['validate_report'] = validate_reports

    train_reports_detail.sort(key=lambda r: -r["accuracy"])
    validate_reports_detail.sort(key=lambda r: -r["accuracy"])
    report['train_report_detail'] = train_reports_detail
    report['validate_report_detail'] = validate_reports_detail

    noises = np.unique(report["noisy_size"])
    report["noisy_size"] = '/'.join(['%s' % v for v in noises])

    train_reports.extend(validate_reports)
    print(pd.DataFrame(train_reports))

    outputs = [
        ['TrainSize', 'Accuracy', 'noisy_size'], 
        ['TrainSize', 'Macro-F1', 'noisy_size'], 
        ['TrainSize', 'Macro-precision', 'noisy_size'],
        ['TrainSize', 'Loss', 'noisy_size']
    ]
    for s in outputs:
        plot_item(report, s[0], s[1], s[2])

    return report


def predict(filename, configs):
    '''Predict the data from data file with the parameters from the config of the group.'''

    result = []
    for i in range(len(configs)):
        config = configs[i]
        if (config['enabled'] != 1):
            continue

        model, train_sizes, validate_size, test_size, noisy_sizes, params = config["model"], config["train"], config['validate'], config['test'], config['noisy'], config['params']
        for i in range(len(train_sizes)):
            name = '%s. %s-%s ' % (i+1, model, params["kernel"])
            X_train, y_train, X_validate, y_validate, X_test, y_test, X_noisy, y_noisy = get_data(filename, train_sizes[i], validate_size, test_size, noisy_sizes[i], name)

            if(model == "SVM"):
                # SVM kernels
                for j in range(len(params["kernel"])):
                    name = '%s. %s-%s ' % (i+1, model, params["kernel"][j])
                    report = svm_fit(X_train, y_train, X_test, y_test, params["kernel"][j], "train" , params)
                    report_validate = svm_fit(X_validate, y_validate, X_test, y_test, params["kernel"][j], "validate", params)
                    result.append({"index": i+1, "model": model, "train": train_sizes[i], "validate": validate_size, "test": test_size, "noisy": noisy_sizes[i], "train_report": report, "validate_report": report_validate})

            else:
                name = '%s. %s-%s ' % (i+1, model, "")
                report = bert_fit(X_train, y_train, X_test, y_test, "train", params)
                report_validate = bert_fit(X_validate, y_validate, X_test, y_test, "validate", params)
                result.append({"index": i+1, "model": model, "train": train_sizes[i], "validate": validate_size, "test": test_size, "noisy": noisy_sizes[i], "train_report": report, "validate_report": report_validate})
        # Show the result in figures and tables
        summary(result)
        result = []  # Empty the result after print the summary

    return result


if __name__ == '__main__':

    file_name = './twitter_sentiment_data.csv'

    groups = 7
    configs = [
        {  # svm -debug
            "enabled": 0,
            "model": "SVM",
            "train": [100,200,300],
            "validate": 100,
            "test": 110,
            "noisy":[80,90,100],
            "params":{"kernel": ['linear', 'poly', 'rbf', 'sigmoid'], "tokenizer":True,"one_hot_encoding":True }  # Not support 'precomputed'
        },
        {  # bert -debug
            "enabled": 0,
            "model": "Bert",
            "train": [10,20,30],
            "validate": 10,
            "test": 10,
            "noisy":[0,0,0],
            "params":{"kernel": ''}
        },

        
        {  # svm with no tokenizer , one_hot_encoding，无噪音情况下tokenizer和one_hot_encoding会降低精度
            "enabled": 0,
            "model": "SVM",
            "train": [2000, 4000, 5000, 8000, 10000, 15000, 20000],
            "validate": 1000,
            "test": 4000,
            "noisy":[0, 0, 0, 0, 0, 0, 0],
            "params":{"kernel": ['linear', 'poly', 'rbf', 'sigmoid'], "tokenizer":False, "one_hot_encoding":False}
        },
        {  # svm with tokenizer, one_hot_encoding
            "enabled": 0,
            "model": "SVM",
            "train": [2000, 4000, 5000, 8000, 10000, 15000, 20000],
            "validate": 1000,
            "test": 4000,
            "noisy":[0, 0, 0, 0, 0, 0, 0],
            "params":{"kernel": ['linear'], "tokenizer":True, "one_hot_encoding":True}
        },
        {  # svm - noisy with no tokenizer , one_hot_encoding
            "enabled": 1,
            "model": "SVM",
            "train": [4000,  8000, 15000],
            "validate": 1000,
            "test": 4000,
            "noisy":[1000, 2000, 5000],
            "params":{"kernel": ['linear'], "tokenizer":False, "one_hot_encoding":False} 
        },
        {  # svm - noisy with tokenizer , one_hot_encoding
            "enabled": 1,
            "model": "SVM",
            "train": [ 4000,  8000, 15000],
            "validate": 1000,
            "test": 4000,
            "noisy":[1000, 2000, 5000],
            "params":{"kernel": ['linear'], "tokenizer":True, "one_hot_encoding":True} 
        },
        {  # bert
            "enabled": 0,
            "model": "BERT",
            "train": [2000, 4000, 5000, 8000, 10000, 15000, 20000],
            "validate": 1000,
            "test": 4000,
            "noisy":[0, 0, 0, 0, 0, 0, 0],
            "params":{"kernel": ''}
        },
        {  # bert - noisy
            "enabled": 0,
            "model": "BERT",
            "train": [2000, 4000, 5000, 8000, 10000, 15000, 20000],
            "validate": 1000,
            "test": 4000,
            "noisy":[0, 1000, 0, 2000, 0, 5000, 0],
            "params":{"kernel": ''}
        },
    ]

    predict(file_name, configs)
