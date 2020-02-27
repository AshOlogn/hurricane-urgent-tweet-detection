from .preprocessing import tweets_to_average_embeddings
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

#########################
##  Non-neural Models  ##
#########################

def get_stats(model, X, y, cv=10, verbose=False):
    cv_results = cross_validate(model, X, y, scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'], 
                                cv=cv, return_train_score=False)
    
    if verbose:
        print(cv_results)
    
    return cv_results

def get_formatted_stats(models, method, dfv, labels, cv=10):
    f1 = []
    precision = []
    recall = []
    accuracy = []
    auc = []
    
    for k,v in models.items():
        stats = get_stats(v, dfv, labels, cv)
        accuracy_avg = np.average(stats['test_accuracy'])
        accuracy_std = np.std(stats['test_accuracy'])
        precision_avg = np.average(stats['test_precision'])
        precision_std = np.std(stats['test_precision'])
        recall_avg = np.average(stats['test_recall'])
        recall_std = np.std(stats['test_recall'])
        f1_avg = np.average(stats['test_f1'])
        f1_std = np.std(stats['test_f1'])
        auc_avg = np.average(stats['test_roc_auc'])

        f1.append('%.2f ± %.2f' % (f1_avg, f1_std))
        precision.append('%.2f ± %.2f' % (precision_avg, precision_std))
        recall.append('%.2f ± %.2f' % (recall_avg, recall_std))
        accuracy.append('%.2f ± %.2f' % (accuracy_avg, accuracy_std))
        auc.append('%.2f' % auc_avg)
    
    df_view = pd.DataFrame(data={'Method': method, 'f1': f1, 
                                 'precision':precision, 'recall':recall,
                                 'accuracy':accuracy, 'auc':auc})
    return df_view

def train_non_neural_average_embeddings(model, df_tweets):
    (dfv, labels) = tweets_to_average_embeddings(df_tweets)
    model.fit(dfv, labels)
    return model

def test_non_neural_average_embeddings(model, df_tweets):
    (dfv, labels) = tweets_to_average_embeddings(df_tweets)
    y_pred = model.predict(dfv)
    return {'accuracy': accuracy_score(labels, y_pred),
            'precision': precision_score(labels, y_pred),
            'recall': recall_score(labels, y_pred),
            'f1': f1_score(labels, y_pred)}

###########
##  CNN  ##
###########

