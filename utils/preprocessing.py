import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

import gensim
from gensim.utils import simple_preprocess
from gensim import corpora, models

from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')

from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

from symspellpy.symspellpy import SymSpell, Verbosity
from sklearn.base import BaseEstimator, TransformerMixin

#create spell checker/word splitter
def create_symspell(max_edit_distance, prefix_length, freq_file_path):
    
    # create object
    sym_spell = SymSpell(max_edit_distance, prefix_length)
    
    # create dictionary using corpus.txt
    if not sym_spell.create_dictionary(freq_file_path):
        print("Corpus file not found")
        return None
    
    return sym_spell

def is_valid_token(w):
    special = ['<url>','<number>', '<user>']
    return w.isalpha() or w in special

def process_tweet(tweet, idxr, tknzr, stop_words=stop_words, sym_spell=None, lemmatize=True, advanced=False):
    st_1 = []
    for w in tknzr.tokenize(tweet):
        #remove retweet annotation if present:
        if w == 'RT':
            if advanced:
                st_1.append('rt')
        
        elif w[0] == '@':
            if advanced:
                st_1.append('<user>')
        
        #remove hashtag symbol
        elif w[0] == '#':
            st_1.append(w[1:])
        
        #replace link with <url> embedding token
        elif w[:4] == 'http':
            st_1.append('<url>')
        
        elif w.isnumeric():
            if advanced:
                st_1.append('<number>')
        
        elif w not in idxr and sym_spell != None:
            split = sym_spell.word_segmentation(w.lower()).corrected_string
            for w_seg in tknzr.tokenize(split):
                st_1.append(w_seg)
        
        else:
            st_1.append(w)
    
    st_2 = []
    
    #remove stop words and punctuation, make everything lowercase
    if advanced:
        st_2 = [w.lower() for w in st_1 if is_valid_token(w) and 
                    not w.lower() in stop_words]
    else:
        st_2 = [w.lower() for w in st_1 if w.isalpha() and
                not w.lower() in stop_words]
    
    #lemmatization (converts all words to root form for standardization)
    lem = WordNetLemmatizer()
    st_3 = list(map(lambda x: lem.lemmatize(x.lower(), pos='v') if lemmatize else x, st_2))

    #now do word segmentation/spell check
    return ' '.join(st_3)

def read_embeddings(vec_length=100, custom_embeddings=None):
    
    num_custom = 0 if custom_embeddings is None else len(custom_embeddings)
    
    #an array of word embeddings as they appear in the file, plus custom embeddings
    embeddings = np.zeros((1193514+num_custom, vec_length))
    
    #two-way map, index->word and word->index
    glove = {}
    
    #first insert custom embeddings
    if custom_embeddings is not None:
        for i,key in enumerate(custom_embeddings.keys()):
            embeddings[i] = custom_embeddings[key]
            glove[i] = key
            glove[key] = i
    
    #now insert the embeddings
    index = num_custom
    with open('data/glove.twitter.27B/glove.twitter.27B.%dd.txt' % vec_length) as f:
        for l in f:
            line = []
            try:
                line = l.split()
                if len(line) != vec_length+1:
                    print('empty line')
                    continue

                word = line[0]
                embeddings[index] = np.array(line[1:]).astype(np.float)
                glove[index] = word
                glove[word] = index
                index += 1
            except:
                break
    
    return (embeddings, glove)

#takes dataframe with processed tweets and returns dataframe with word embeddings
def tweets_to_df(df, labels, embeddings, glove, vec_length=100):
    
    weights = []
    index_omit = []
    index = -1
    tweets = df['text']
    
    #a column for each entry in the embedding vector
    for i in range(vec_length+1):
        weights.append([])
    
    for i in range(len(tweets)):
        index += 1
        cur_embed = []
        cur_tweet = tweets[i]
        cur_label = labels[i]
        for i in cur_tweet.split():
            if i in glove:
                cur_embed.append(embeddings[glove[i]])
        
        if len(cur_embed) == 0:
            #make sure we drop this row from the input dataframe
            index_omit.append(index)
            continue
        
        x = np.asarray(np.mean(cur_embed, axis=0))
        
        for j in range(vec_length):
            weights[j].append(x[j])
        
        #weights[vec_length].append(0 if cur_label == 0 else 1)
        weights[vec_length].append(cur_label)
        
    df_pruned = df.drop(index_omit)
    
    #convert to dataframe
    cols = {}
    for i in range(vec_length):
       cols['v' + str(i)] = weights[i]
    
    cols['class'] = weights[vec_length]
    return pd.DataFrame(data=cols)

def apply_embeddings(df, labels, glove):
    
    X = []
    max_len = 0;
    
    for tweet in df['text']:
        indices = []
        words = tweet.split()
        max_len = max(max_len, len(words))
        
        for word in words:
            if word in glove:
                indices.append(glove[word])
            else:
                indices.append(glove['UNK'])
        
        X.append(indices)
    
    # add padding to make every tweet the same length
    for i in range(len(X)):
        tweet = X[i]
        if len(tweet) < max_len:
            tweet = np.append(tweet, np.ones(max_len - len(tweet)))
        X[i] = tweet
    
    X = np.asarray(X, dtype=np.int64)
    y = np.array(labels, dtype=np.int64)
    return (X, y)

class ProcessTweetTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, glove, symspell=None, lemmatize=True, advanced=True):
        self.glove = glove
        self.symspell = symspell
        self.lemmatize = lemmatize
        self.advanced = advanced
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        tknzr = TweetTokenizer(strip_handles=False, reduce_len=True)
        X['text'] = X['text'].map(lambda x: process_tweet(x, self.glove, tknzr, self.symspell, self.lemmatize, self.advanced))
        return X

class AverageEmbeddingsTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, embeddings, glove, class_name='class'):
        self.embeddings = embeddings
        self.glove = glove
        self.class_name = class_name
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return tweets_to_df(X, X[self.class_name], self.embeddings, self.glove)

def tweets_to_average_embeddings(df_tweets, embeddings=None, embeddings_dict=None):
    symspell = create_symspell(3,7,'data/frequency_dictionary_en_82_765.txt')

    if embeddings is None != embeddings_dict is None:
        raise Exception("Both embeddings and associated dictionary should be provided together.")
    elif embeddings is None:
        (embeddings, embeddings_dict) = read_embeddings(vec_length=100)
    
    #now preprocess the tweets using word embeddings
    tweet_process = Pipeline([
                    ('process tweet', ProcessTweetTransformer(embeddings_dict, symspell, False, True))
                    ('generate embeddings', AverageEmbeddingsTransformer(embeddings, embeddings_dict, 'class'))
                ])
    
    dfv = tweet_process.fit_transform(df_tweets)
    labels = dfv.pop('class')
    return (dfv, labels)
