import sys
sys.dont_write_bytecode = True

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import inspect



warnings.filterwarnings('ignore')


def accuracy_f1_scores(X_train:pd.DataFrame , y_train:np.array , model):
    """Getting accuracy and f1 scores using cross validation values

    Args:
        X_train (pd.DataFrame): indenpendent values that model will train using it
        y_train (anp.array): depenendent values that model will train using it
        model   (Any): model that you will train

    Returns:
        Accuracy_F1_array (np.array): accuracy is the first array in the list and f1 the second array
                
    """
    from sklearn.model_selection import cross_val_score , StratifiedKFold
    import pandas as pd
    # Create a reproducible cross-validation splitter
    cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    accuracy = cross_val_score(model,X_train , y_train , scoring='accuracy' , cv=cv_splitter) # Getting accuracy using cross validation to the high performance of the model
    f1 = cross_val_score(model ,X_train , y_train , scoring='f1' , cv=cv_splitter) # also the same thing but this to check there are outliers effected on the model by checking if there aren't any significant 
    print('CV=5 , random_state=42 , shuffel=True')
    print('accuracy: ' , accuracy.mean())
    print('f1: ' , f1.mean())
    print(pd.DataFrame({
        'Accuracy score': accuracy.flatten(),
        'F1 score':f1.flatten()
    }))
    return accuracy , f1

def drop_percentages(X_test:pd.DataFrame ,y_test:np.array , model:any , accuracy:np.array , f1:np.array):
    """Get accuracy and f1 drop percentage to check overfitting or underfitting

    Args:
        X_test (pd.DataFrame): indenpendent test sample
        y_test (np.array): denpendent test sample
        model (Any): model you fitted
        accuracy (np.array): accuracy scores that got from **accuracy_f1_scores** func
        f1 (np.array): f1 scores that got from **accuracy_f1_scores** func
    Returns:
        Drop_percentags (dict): Accuracy drop percentage , f1 drop percentage , y_pred
    """
    # See accuracy and f1 drop percentage
    from sklearn.metrics import accuracy_score, f1_score
    y_pred = model.predict(X_test)

    accuracy_unseen = accuracy_score(y_test , y_pred)
    f1_unseen = f1_score(y_test , y_pred)
    
    accuracy_drop_percentage = f'{round(
        (accuracy.mean() - accuracy_unseen) * 100 
        , 1)}%'
    f1_drop_percentage = str(round(
        (f1.mean() - f1_unseen) * 100 ,1)
                            ) + '%'
    
    return {
        'Accuracy drop percentage' : accuracy_drop_percentage,
        'f1 drop percentage': f1_drop_percentage,
        'y_pred' : y_pred
    }
    

def model_results_imports():
    """Import all needed functions to get the accuracy of the classification models
    and inject them into the caller's global namespace.
    """
    import sklearn.metrics
    caller_globals = inspect.currentframe().f_back.f_globals
    caller_globals['accuracy_score'] = sklearn.metrics.accuracy_score
    caller_globals['classification_report'] = sklearn.metrics.classification_report
    caller_globals['confusion_matrix'] = sklearn.metrics.confusion_matrix
    caller_globals['f1_score'] = sklearn.metrics.f1_score