from gensim.test.utils import common_dictionary, common_corpus
from gensim.models import LsiModel
from collections import defaultdict
from gensim import corpora
import pandas as pd
import numpy as np

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer

class LSI:
    def __init__(self):
        self.mDict = None
        self.mCorpus = None
        self.mLsiModel = None
        self.mNumTopics = 20
        pass
    
    def BuildModel(self, documents, num_topics = 20):
        '''
        build the lsi model by documents
        '''
        self.mDict, self.mCorpus = self.Document2Corpus(documents)
        self.mNumTopics = num_topics
        self.mLsiModel = LsiModel(corpus=self.mCorpus, id2word=self.mDict, num_topics=num_topics)
    
    def Query2LatentSpace(self, queryStr):
        '''
        find the vector in latent space for a query
        '''
        query = self.mDict.doc2bow(queryStr.lower().split())
        queryLsi = self.mLsiModel[query]  # convert the query to LSI space
        df = pd.DataFrame(queryLsi, columns=['index','vec'])
        queryVec = df['vec'].tolist()
        return queryVec
    
    def QueryList2LatentSpace(self, queryList):
        resultArray = None
        for queryStr in queryList:
            queryVec = self.Query2LatentSpace(queryStr)
            row = np.array(queryVec)            
            if resultArray is not None:
                if row.shape[0] == 0 :
                    # LSI can't derive vector from the query,so I append 0s to 
                    dummyVec = np.zeros( self.mNumTopics)
                    testVec = self.Query2LatentSpace(queryStr)
                    resultArray = np.vstack([resultArray, dummyVec])
                else:                    
                    resultArray = np.vstack([resultArray, row])
            else:
                resultArray = row
            
        return resultArray
    
    # 将文档中的所有单词转换为小写，移除掉stop words，并按空格分割
    # remove common words and tokenize
    def Document2Corpus(self,documents):    
        stoplist = set('for a of the and to in'.split())
        texts = [
            [word for word in document.lower().split() if word not in stoplist]
            for document in documents
        ]
        # 这里删除罕见词，将只出现过1次的词删掉
        # remove words that appear only once
        
        
        '''
        frequency = defaultdict(int)
        for text in texts:
            for token in text:
                frequency[token] += 1
        
        texts = [
            [token for token in text if frequency[token] > 1]
            for text in texts
        ]
        '''

        # 将文档库的所有单词存入字典，内部自动赋予ID
        # build the dictionary
        dictionary = corpora.Dictionary(texts)
        # 重新将文档库进行doc2bow，每篇文章由单词数组，转换为了[(id1, number),(id2, number), ... ]的形式，每个独一无二的单词对应一个id
        # transform the documents to bag of words, the format is [(id1, number),(id2, number), ... ]
        corpus = [dictionary.doc2bow(text) for text in texts]
        return (dictionary, corpus)
    
class SKLearnLSA:
    def __init__(self):
        # doc-term matrix
        self.mDTM = None
        self.mLSA = None
        self.U = None
        self.V = None
        self.S = None
        self.mVectorizer = None
        pass
    
    def BuildModel(self, docs, topics = 2):
        '''
        Build LSA model
        '''
        # build doc term matrix
        vectorizer = CountVectorizer(min_df = 1, stop_words = 'english')
        self.mVectorizer = vectorizer
        self.mDTM = vectorizer.fit_transform(docs)
        dtm = self.mDTM
        # build svd from doc term matrix
        self.mLSA = TruncatedSVD(topics, algorithm = 'arpack')
        lsa = self.mLSA
        dtmfloat = dtm.astype(float)
        dtm_lsa = self.mLSA.fit_transform(dtmfloat)
        dtm_lsa = Normalizer(copy=False).fit_transform(dtm_lsa)
        
        # component names
        compNames = []
        for k in range(topics):
            compNames.append("component_" + str(k+1))
        
        UT = pd.DataFrame(lsa.components_,index = compNames,columns = vectorizer.get_feature_names_out())
        self.U = UT.T
        self.V = pd.DataFrame(dtm_lsa, index = docs, columns = compNames)
        self.S = np.diag(lsa.singular_values_)
        self.SInv = np.diag(1 / lsa.singular_values_)
        
    def Query2LatentSpace(self, strList):
        q = self.mVectorizer.transform(strList)
        query = q@self.U@self.S
        return query
    def Query2LatentSpaceUnscaled(self, strList):
        q = self.mVectorizer.transform(strList)
        query = q@self.U@self.SInv
        return query