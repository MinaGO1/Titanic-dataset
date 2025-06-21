import sys
sys.dont_write_bytecode = True

import inspect
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.typing import NDArray
import warnings 
warnings.filterwarnings('ignore')
def auto_imports():
    """Import main libraries like pandas,seaborn,...""" 
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import warnings
    warnings.filterwarnings('ignore')
    global_ = inspect.currentframe().f_back.f_globals
    global_['pd'] = pd
    global_['np'] = np
    global_['sns'] = sns
    global_['plt'] = plt

def accuracy_f1_scores(X_train:pd.DataFrame , y_train:np.ndarray , model):
    """Getting accuracy and f1 scores using cross validation values

    Args:
        X_train (pd.DataFrame): indenpendent values that model will train using it
        y_train (anp.array): depenendent values that model will train using it
        model   (Any): model that you will train

    Returns:
        Accuracy_F1_array (np.ndarray): accuracy is the first array in the list and f1 the second array
                
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

def drop_percentages(X_test:pd.DataFrame ,y_test:np.ndarray , model:any , accuracy:np.ndarray , f1:np.ndarray):
    """Get accuracy and f1 drop percentage to check overfitting or underfitting

    Args:
        X_test (pd.DataFrame): indenpendent test sample
        y_test (np.ndarray): denpendent test sample
        model (Any): model you fitted
        accuracy (np.ndarray): accuracy scores that got from **accuracy_f1_scores** func
        f1 (np.ndarray): f1 scores that got from **accuracy_f1_scores** func
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
    


# def drop_percentages(y_pred:NDArray[np.int64] ,y_test:NDArray[np.int64] , model:any ,accuracy:float,print: bool = False):
#     """Get accuracy and f1 drop percentage to check overfitting or underfitting

#     Args:
#         y_pred (NDArray[np.int64]): predicted trager
#         y_test (NDArray[np.float64]): denpendent test sample
#         model (Any): model you fitted
#         accuracy (float): accuracy score of the seen data
#     Returns:
#         Drop_percentags (dict): Accuracy drop percentage , y_pred
#     """
#     # See accuracy and f1 drop percentage
#     from sklearn.metrics import accuracy_score, f1_score

#     accuracy_unseen = accuracy_score(y_test , y_pred)
#     f1_unseen = f1_score(y_test , y_pred)
    
#     accuracy_drop_percentage =round(
#         (accuracy - accuracy_unseen) * 100 
#         , 1)

#     if print:
#         print('accuracy of the seen data: ' , accuracy)
#         print('accuracy of the unseen data: ' , accuracy_unseen) 
#         print('f1 of the unseen data: ' , f1_unseen) 
#     return {
#         'Accuracy drop percentage' : accuracy_drop_percentage,
#         'unseen data':{
#             'accuracy':accuracy_unseen,
#             'f1':f1_unseen
#         }
#     }
    

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
    
    
def save_model_predictions(y_pred:NDArray[np.float64] , modelname:str)-> pd.DataFrame:
    """Save model predictions to a csv file 

    Args:
        y_pred (NDArray[np.float64]): your predict y from the model
        modelname (str): model name to save csv file

    Returns:
        pd.DataFrame: returns a pandas dataframe as a csv file
    """
    import os
    predict_df = pd.read_csv(r'E:\Data science\Titanic dataset\data\Processed data\Data Analysis\processed_data.csv') # loading predict df to get passengers ID
    predict_df = predict_df[predict_df['ind'] == 'test']
    pd.DataFrame({
    'PassengerId': predict_df['PassengerId'],
    
    'Survived': y_pred # Predicting predict data
    }).to_csv(os.path.join('E:\Data science\Titanic dataset\data\Processed data\Data Modeling' ,
                           modelname+'_predictions.csv'),
              index=False)
    
# getting best random  state of train_test_split
# def get_best_seed(x:pd.DataFrame , y:NDArray[np.float64] , model:any , iter:int=50 , max_drop: float = 10.0) -> int:
#     """Get best seed for random_state param of the train test split

#     Args:
#         x (pd.DataFrame): independent values (feature)
#         y (NDArray[np.float64]): depenedent values (target)
#         model (any): classification model like (svm , rf , logit , knn)
#         iter (int, optional): number of seeds you want (from 1 to iter). Defaults to 50.
#         max_drop (float ,optional): max drop precentage for accuracy and f1. Defaults to 10.0.

#     Returns:
#         best_seed: return best seed for train test split random state
#     """
#     from sklearn.model_selection import train_test_split
#     best_seed = None
#     best_score = -np.inf
#     best_precision_0 = best_recall_0 = 0
#     best_precision_1 = best_recall_1 = 0
#     lowest_accuracy_drop = np.inf

#     for seed in range(iter):
#         X_train , X_test , y_train , y_test = train_test_split(
#             x , y , test_size=0.2 , random_state=seed
#         )
#         # make model
#         model.fit(X_train , y_train) # fit model

#         from sklearn.metrics import classification_report       
#         model_probs = model.predict_proba(X_test)[:,1]
#         report = get_best_thershold(model_probs , y_test) # get best results using best threshold value
#         class_0 = report['0'] #class 0 scores
#         class_1 = report['1'] #class 1 scores
#         best_threshold = report['best_threshold'] # best threshold value
#         report_drop = drop_percentages(model.predict(X_test) , y_test , model , model.score(X_train , y_train))
#         accuracy_drop = report_drop['Accuracy drop percentage'] # extract drop percentage from dict

#         precision_0_threshold , recall_0_threshold = class_0['precision'] , class_0['recall']
#         precision_1_threshold , recall_1_threshold = class_1['precision'] , class_1['recall']

#         # Score: prioritize lowest accuracy drop, then sum of precision/recall
#         score = -(accuracy_drop) + (precision_0_threshold + recall_0_threshold + precision_1_threshold + recall_1_threshold)

#         if accuracy_drop <= max_drop and score > best_score:
#             best_score = score
#             best_seed = seed
#             best_precision_0 , best_recall_0 = precision_0_threshold , recall_0_threshold 
#             best_precision_1 , best_recall_1 = precision_1_threshold , recall_1_threshold
#             lowest_accuracy_drop = accuracy_drop

#     print('best precision for class 0: ', best_precision_0)            
#     print('best recall for class 0: ', best_recall_0)            
#     print('best precision for class 1: ', best_precision_1)            
#     print('best recall for class 1: ', best_recall_1)            
#     print('best seed: ', best_seed)
#     return best_seed

