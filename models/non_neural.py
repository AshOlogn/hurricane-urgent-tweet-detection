import pandas as pd
from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
from nltk.tokenize import TweetTokenizer
from nltk import pos_tag

from .neg_detection import *

class NgramsClassifier(ClassifierMixin, BaseEstimator):
    
    def __init__(self, base_estimator, n=[1,1], stop_words=['a', 'an', 'the']):
        self.base_estimator = base_estimator
        self.n = n
        self.stop_words = stop_words
        self.vectorizer = self.vectorizer = CountVectorizer(ngram_range=self.n, preprocessor=lambda x: x, 
                                     tokenizer=lambda x: self.process_tweet(x, stop_words=self.stop_words))
        self.pos_tags = None
    
    def process_tweet(self, tweet, stop_words):
        tknzr = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
        tokens_raw = tknzr.tokenize(tweet)
        
        def keep_token(word):
            #get rid of URLs (handles already gone)
            if word in stop_words or word == 'rt' or word[:4] == 'http':
                return False
            
            #now make sure word has some letters or numbers (so not just punctuation)
            for c in word:
                if c.isalnum():
                    return True
            return False
        
        return list(filter(keep_token, tokens_raw))
    
    def _process_training_data(self, df):
        """
        Use fit_transform to remember the corpus for testing
        """
        return pd.DataFrame(self.vectorizer.fit_transform(df).toarray())
    
    def _process_test_data(self, df):
        """
        Use transform to use the corpus seen during training
        """
        return pd.DataFrame(self.vectorizer.transform(df).toarray())
    
    def fit(self, X, y=None):
        X_ngram = self._process_training_data(X)
        self.base_estimator.fit(X_ngram, y)
        return self
    
    def predict(self, X, y=None):
        X_ngram = self._process_test_data(X)
        return self.base_estimator.predict(X_ngram)
    
    def decision_function(self, X, y=None):
        X_ngram = self._process_test_data(X)
        return self.base_estimator.decision_function(X_ngram)
    
    def predict_proba(self, X, y=None):
        X_ngram = self._process_test_data(X)
        return self.base_estimator.predict_proba(X_ngram)
    
    def score(self, X, y, sample_weight=None):
        X_ngram = self._process_test_data(X)
        return self.base_estimator.score(X_ngram, y, sample_weight)


class NgramsPOSClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, base_estimator, tagset=load('help/tagsets/upenn_tagset.pickle'), 
                 n=(1,2), stop_words=['a', 'an', 'the']):
        
        self.base_estimator = base_estimator
        self.tagset = tagset
        self.n = n
        self.stop_words = stop_words
        self.vectorizer = self.vectorizer = CountVectorizer(ngram_range=self.n, preprocessor=lambda x: x, 
                                     tokenizer=lambda x: self.process_tweet(x, stop_words=self.stop_words))
    
    
    def process_tweet(self, tweet, stop_words):
        tknzr = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
        tokens_raw = tknzr.tokenize(tweet)
        
        def keep_token(word):
            #get rid of URLs (handles already gone)
            if word in stop_words or word == 'rt' or word[:4] == 'http':
                return False
            
            #now make sure word has some letters or numbers (so not just punctuation)
            for c in word:
                if c.isalnum():
                    return True
            return False
        return list(filter(keep_token, tokens_raw))
    
    def pos_tag_data(self, data):
        counts = {}
        for tag in self.tagset:
            counts[tag] = []
        
        for tweet in data:
            #don't omit stop words since we want complete POS counts
            tokens = self.process_tweet(tweet, [])
            count = Counter([j for i,j in pos_tag(tokens)])
            for tag in counts:
                counts[tag].append(count[tag])
        
        df_pos = pd.DataFrame(counts)
        return df_pos
    
    def _process_training_data(self, data, y):
        """
        Use fit_transform to remember the corpus for testing
        """
        df_ngram = pd.DataFrame(self.vectorizer.fit_transform(data).toarray())
        
        #now determine the most important n-gram features using chi2
        feature_names = self.vectorizer.get_feature_names()
        name_to_index = {}
        for index,feature in enumerate(feature_names):
            name_to_index[feature] = index
        
        #rank features with chi2
        scored_features = sorted(list(zip(feature_names, chi2(df_ngram, y)[0])), 
                                 key=lambda x: x[1], reverse=True)[:1000]
        
        
        self.ngram_cols = [name_to_index[f[0]] for f in scored_features] 
        df_ngram = df_ngram[self.ngram_cols]
        df_pos = self.pos_tag_data(data)
        return pd.concat([df_pos, df_ngram], axis=1)
    
    def _process_test_data(self, data):
        """
        Use transform to use the corpus seen during training
        """
        df_ngram = pd.DataFrame(self.vectorizer.transform(data).toarray())[self.ngram_cols]
        df_pos = self.pos_tag_data(data)
        return pd.concat([df_pos, df_ngram], axis=1)
    
    def fit(self, X, y=None):
        X_ngram = self._process_training_data(X, y)
        self.base_estimator.fit(X_ngram, y)
        return self
    
    def predict(self, X, y=None):
        X_ngram = self._process_test_data(X)
        return self.base_estimator.predict(X_ngram)
    
    def decision_function(self, X, y=None):
        X_ngram = self._process_test_data(X)
        return self.base_estimator.decision_function(X_ngram)
    
    def predict_proba(self, X, y=None):
        X_ngram = self._process_test_data(X)
        return self.base_estimator.predict_proba(X_ngram)
    
    def score(self, X, y, sample_weight=None):
        X_ngram = self._process_test_data(X)
        return self.base_estimator.score(X_ngram, y, sample_weight)


class NgramsNegDetectionClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, base_estimator, n_features=1000, n=[1], negation_words=['no', 'not']):
        self.base_estimator = base_estimator
        self.n_features = n_features
        self.n = n
        self.negation_words = negation_words
        self.ngrams = None
    
    def process_tweet(self, tweet, stop_words=['a', 'an', 'the']):
        tknzr = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
        tokens_raw = tknzr.tokenize(tweet)

        def keep_token(word):
            #get rid of URLs (handles already gone)
            if word in stop_words or word == 'rt' or word[:4] == 'http':
                return False
            
            #now make sure word has some letters or numbers (so not just punctuation)
            for c in word:
                if c.isalnum():
                    return True
            return False
        
        return list(filter(keep_token, tokens_raw))
    
    def ngrams_vectorizer(self, X, y, n_features=1000, n=[1,2], negation_words=['no', 'not']):
        tokens = list(map(lambda x: self.process_tweet(x), X))
        columns = {}
        ngrams_total = []
        n = sorted(n)
        
        for n2 in n:
            #determine n2-grams
            ngrams = ngrams_nd(tokens=tokens, n=n2, negation_words=negation_words)
            ngrams_total.append(ngrams)
            binary_features = np.zeros((len(X), len(ngrams)), dtype=np.int8)
            
            #create binary feature vector for each input text
            for i in range(len(X)):
                binary_features[i] = ngrams_vectorizer_nd(tokens[i], ngrams, n=n2, negation_words=negation_words)
            
            #create a dataframe with the binary features
            binary_features = np.transpose(binary_features)
            for i in range(len(binary_features)):
                columns[ngrams[i]] = binary_features[i]
        
        df_ngram = pd.DataFrame(columns)
        df_ngram = df_feature_select(df_ngram, y, k=n_features)
        
        #only return the ngram features that are selected
        return (df_ngram, list(df_ngram.columns))
    
    def ngrams_test_vectorizer(self, X, ngrams, n, negation_words=['no', 'not']):
        tokens = list(map(lambda x: self.process_tweet(x), X))
        columns = {}
        n = sorted(n)
        
        for n2 in n:
            #first filter out the indices ngrams of length n=n2
            indices_n2 = []
            for i in range(len(ngrams)):
                if(len(ngrams[i].split('@')) == n2):
                    indices_n2.append(i)
            
            if len(indices_n2)==0:
                continue
            
            binary_features = np.zeros((len(X), len(ngrams)), dtype=np.int8)
            
            #create binary feature vector for each input text
            for i in range(len(X)):
                binary_features[i] = ngrams_vectorizer_nd(tokens[i], ngrams, n=n2, negation_words=negation_words)
            
            #create a dataframe with the binary features, only at indices where the ngram has n=n2
            binary_features = np.transpose(binary_features)
            for i in indices_n2:
                columns['ngram:' + ngrams[i]] = binary_features[i]
        
        df_ngram = pd.DataFrame(columns)
        return df_ngram
    
    def fit(self, X, y=None):
        (X_ngram, ngrams) = self.ngrams_vectorizer(X, y, n_features=self.n_features, n=self.n, 
                                               negation_words=self.negation_words)
        self.ngrams = ngrams
        self.base_estimator.fit(X_ngram, y)
        return self
    
    def predict(self, X, y=None):
        #first transform input into ngram binary features
        X_ngram = self.ngrams_test_vectorizer(X, self.ngrams, self.n, self.negation_words)
        return self.base_estimator.predict(X_ngram)
    
    def decision_function(self, X, y=None):
        X_ngram = self.ngrams_test_vectorizer(X, self.ngrams, self.n, self.negation_words)
        return self.base_estimator.decision_function(X_ngram)
    
    def predict_proba(self, X, y=None):
        X_ngram = ngrams_test_vectorizer(X, self.ngrams, self.n, self.negation_words)
        return self.base_estimator.predict_proba(X_ngram)
    
    def score(self, X, y, sample_weight=None):
        X_ngram = self.ngrams_test_vectorizer(X, self.ngrams, self.n, self.negation_words)
        return self.base_estimator.score(X_ngram, y, sample_weight)