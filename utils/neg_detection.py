import numpy as np
import pandas as pd

def build_window(tokens, start, n, negation_words):
    window = []
    count = 0
    index = start
    while count < n and index < len(tokens):
        if (index+1 < len(tokens) and (tokens[index] in negation_words or 
                                      tokens[index+1] in negation_words)):
            window.append(tokens[index]+'+'+tokens[index+1])
            index += 2
        else:
            window.append(tokens[index])
            index += 1
        count += 1
    if count < n:
        return (None, -1, -1)
    else:
        return (window, start, index-1)

def update_window(tokens, window, start, end, n, negation_words):
    if end >= len(tokens)-1:
        return (None, -1, -1)
    elif '+' in window[0] and window[0].split('+')[1] in negation_words:
        return build_window(tokens, start+1, n, negation_words)
    else:
        if end+2 < len(tokens) and tokens[end+2] in negation_words:
            #fuse negator at end+2 to the next word added
            window.append(tokens[end+1]+'+'+tokens[end+2])
            start += 2 if '+' in window[0] else 1
            del window[0]
            return (window, start, end+2)
        else:
            window.append(tokens[end+1])
            start += 2 if '+' in window[0] else 1
            del window[0]
            return (window, start, end+1)

def ngrams_nd(tokens, n=1, negation_words=['no', 'not']):
    '''
    tokens: list or Series of words, or list of word lists
    negation_words: words that should only be used as a modifier for previous and next word
                    (e.g. not+good or better+not)
    returns a list of ngrams found in the input word list
    
    Assumes no two negation words are consecutive in the dataset (a BIG assumption, admittedly)
    '''
    def ngrams_nd_single_list(tokens, n=1, negation_words=['no', 'not']):
        ngrams = set()
        (window,start,end) = build_window(tokens, 0, n, negation_words)
        
        if window is None:
            return ngrams
        
        while window is not None:
            ngrams.add('@'.join(window))
            (window, start, end) = update_window(tokens, window, start, end, n, negation_words)
        return ngrams
    
    if isinstance(tokens[0], pd.Series) or isinstance(tokens[0], list):
        ngrams = set()
        for t in tokens:
            ngrams |= ngrams_nd_single_list(t, n, negation_words)
        return list(ngrams)
    else:
        return list(ngrams_nd_single_list(tokens, n, negation_words))

def ngrams_vectorizer_nd(tokens, ngrams, n=1, negation_words=['no', 'not']):
    ngram_to_index = {}
    for i in range(len(ngrams)):
        ngram_to_index[ngrams[i]] = i
    
    binary_features = np.zeros((len(ngrams),), dtype=np.int8)
    (window,start,end) = build_window(tokens, 0, n, negation_words)
    
    if window is None:
        return binary_features
    
    while window is not None:
        if '@'.join(window) in ngram_to_index:
            binary_features[ngram_to_index['@'.join(window)]] = 1
        (window, start, end) = update_window(tokens, window, start, end, n, negation_words)    
    return binary_features

def df_feature_select(df, labels, k=1000, feature_selector=SelectKBest(score_func=chi2)):
    feature_selector.k = k
    feature_selector.fit(df, labels)
    selected_columns = [df.columns[i] for i in feature_selector.get_support(indices=True)]
    df_selected = pd.DataFrame(data=feature_selector.transform(df), columns=selected_columns)
    return df_selected