import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay
from contextlib import redirect_stdout
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
import joblib
import matplotlib.pyplot as plt 

def report(results, n_top=3):
    """Utility function to report the best scores."""

    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates[:1]:
            print("Model rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})"
                  .format(results['mean_test_score'][candidate],
                          results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
            

def evaluate(pipe, X, y, plot=False):
    """Evaluate models."""
    y_pred = pipe.predict(X)
    print(classification_report(y, y_pred, digits=3))
    print(f"auc macro {roc_auc_score(y, pipe.predict_proba(X)[:, 1]):.3f}")

    if plot:
        ConfusionMatrixDisplay.from_estimator(pipe, X, y, normalize=None, values_format = '')
        plt.grid(False)
    else:
        print("confusion matrix")
        print(confusion_matrix(y, y_pred))


def train_and_evaluate(
    preprocess, 
    model, 
    hyperparams, 
    X_train, 
    y_train, 
    X_valid, 
    y_valid, 
    scoring="f1_macro", 
    iter=5000, 
    save=True, 
    savename="",
    path_models ="",
    output_models = "",
    suffix = ""
):
    rand = train(
        preprocess=preprocess,
        model=model,
        hyperparams=hyperparams,
        X_train=X_train,
        y_train=y_train,
        scoring=scoring,
        iter=iter
    )
    
    print("Testing on training set:")
    evaluate(rand.best_estimator_, X_train, y_train)
    print("Testing on validation set:")
    evaluate(rand.best_estimator_, X_valid, y_valid)
    report(rand.cv_results_, n_top=5)
    file_name = output_models.replace("models_output/", '').replace("/","") + suffix + ".txt"

    if save:
         # Dump results as log
        with open(f"{output_models}{file_name}", 'a+') as f:
            with redirect_stdout(f):
                print (f"####################   {savename}    #########################")
                print("Testing on training set:")
                evaluate(rand.best_estimator_, X_train, y_train)
                print("Testing on validation set:")
                evaluate(rand.best_estimator_, X_valid, y_valid)
                report(rand.cv_results_, n_top=1)
                print (f"####################   {savename}  END   #########################")
        joblib.dump(rand.best_estimator_, f"{path_models}{savename}.joblib")
    
    return rand.best_estimator_

def train(
    preprocess, 
    model, 
    hyperparams, 
    X_train, 
    y_train, 
    scoring="f1_macro", 
    iter=5000,
):
    """Train and evaluation pipeline."""
    pipe = Pipeline(steps=[
        ('preprocess', preprocess), 
        ('model', model)
    ])

    rand = RandomizedSearchCV(estimator= pipe,
                              param_distributions=hyperparams,
                              n_iter=iter,
                              scoring=scoring,
                              cv=2,
                              n_jobs=-1,    # use all processors
                              refit=True,   # refit the best model at the end
                              return_train_score=True,
                              verbose=0).fit(X_train, y_train)
    
    return rand