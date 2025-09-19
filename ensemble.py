from typing import Iterable
from pandas.core.frame import DataFrame
import numpy as np
from joblib import load
from sklearn.metrics import classification_report, f1_score, confusion_matrix, roc_auc_score, brier_score_loss
from utils import DebuggablePipeLine
from sklearn.calibration import CalibratedClassifierCV
from itertools import chain, repeat, count, islice
from collections import Counter
def build_ensemble_path(models, path):
    ensemble = []
    for m in models:
        ensemble.append((m, load(path+f"{m}.joblib")))
    
    return ensemble

def build_ensemble(models):
    ensemble = []
    for name,model in models:
        ensemble.append((name, model))
    
    return ensemble


def predict_ensemble(ensemble, X, y, threshold=0.5):
    y_proba = []
    for m in ensemble:
        # Do a cast only if you want to see your data transformed
        #m = DebuggablePipeLine.cast(m)
        y_proba.append(m.predict_proba(X))
    y_proba = np.mean(y_proba, axis=0)
    y_pred = y_proba[:, 1] > threshold
    return y_proba, y_pred


def evaluate_ensemble(ensemble, X, y, threshold=0.5, verbose=True):
    y_proba, y_pred = predict_ensemble(ensemble, X, y)
    if verbose:
        print(classification_report(y, y_pred, digits=3))
        print(f"auroc {roc_auc_score(y, y_proba[:, 1]):.3f}")
        print(f"brier {brier_score_loss(y, y_proba[:, 1]):.3f}")
        print(confusion_matrix(y, y_pred))
  
    return (roc_auc_score(y, y_proba[:, 1]),f1_score(y, y_pred, average="macro"), brier_score_loss(y, y_proba[:, 1]))



def find_best_ensemble(models_list, path, X_training, y_training ,  X_valid, y_valid, verbose = False):
    results = []
    for key in range(2,len(models_list)):
        combinations = list(unique_combinations(models_list,key))
        for combine in combinations:
            combine_name = list(name for (name,model) in combine)
            #print(list(name for (name,model) in combine))
            #print(combine)
            ensemble = build_ensemble(combine)

            # tmp = ensemble
            tmp = [_m for _, _m in ensemble]
            acc = evaluate_ensemble(tmp, X_valid, y_valid, verbose= verbose)
            results.append((combine_name, tmp, acc))
    results.sort(key=lambda item:item[2][2])
    results = results[:5]
    copy = results.copy()
    filter = [(x[0],x[2]) for x in results]
    copy.sort(key=lambda item:-item[2][1])
    for names, model, score  in copy[:5]:
        if (names, score) not in filter:
            results.append((names,model,score))
    results.sort(key=lambda item:item[2][2])
    index = 1
    for ensemble in results[:10]:
        names, model, score = ensemble
        print("##############################################")
        print (f" Rank: #{index} Names: {names}, Score: {score}")
        index +=1
    #return model ensemble
    return results
    

def repeat_chain(values, counts):
    return chain.from_iterable(map(repeat, values, counts))


def unique_combinations_from_value_counts(values, counts, r):
    n = len(counts)
    indices = list(islice(repeat_chain(count(), counts), r))
    if len(indices) < r:
        return
    while True:
        yield tuple(values[i] for i in indices)
        for i, j in zip(reversed(range(r)), repeat_chain(reversed(range(n)), reversed(counts))):
            if indices[i] != j:
                break
        else:
            return
        j = indices[i] + 1
        for i, j in zip(range(i, r), repeat_chain(count(j), counts[j:])):
            indices[i] = j


def unique_combinations(iterable, r):
    values, counts = zip(*Counter(iterable).items())
    return unique_combinations_from_value_counts(values, counts, r)