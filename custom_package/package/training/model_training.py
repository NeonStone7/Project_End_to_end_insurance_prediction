import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.model_selection import cross_val_score,cross_val_predict, StratifiedKFold
from sklearn.metrics import  confusion_matrix, classification_report, precision_recall_curve, roc_curve
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay

multi_roc_scorer = make_scorer(lambda y_in, y_p_in: roc_auc_score(y_in, y_p_in, multi_class='ovr'), needs_proba=True)

def split_data_by_type(data:pd.DataFrame, target:str):
    
    """This function splits the data into independent and dependent variables, then it further extracts the 
    categorical and numerical features into lists.
    """
    
    xtrain, ytrain = data.drop(target, axis = 1).copy(), data[target].copy()
    
    categorical = xtrain.select_dtypes('O').columns.tolist()
    numerical = xtrain.select_dtypes('number').columns.tolist()
    
    return categorical, numerical, xtrain, ytrain


def performance_plots(cm,fpr,tpr,roc_auc,estimator_name, precision,recall):
    
    """This functions takes in parameters need to display the roc_curve, precision-recall curve and confusion_matrix
    cm: confusion matrix
    Inputs:
    
    cm: confusion matrix
    fpr: False Positive Rate
    tpr: True Positive Rate
    roc_auc: ROC AUC score
    estimator_name: Model name
    precision: Precisions from precision-recall curve
    recall: Recalls from precision-recall curve
    
    """
    
    ConfusionMatrixDisplay(cm).plot()
    plt.title(f'Confusion matrix for {estimator_name}')
    plt.show()
    
    
    RocCurveDisplay(fpr = fpr,tpr = tpr,roc_auc = roc_auc,estimator_name = estimator_name).plot()
    plt.title(f'RocCurve for {estimator_name}')
    plt.show()
    
    PrecisionRecallDisplay(precision = precision,recall = recall,estimator_name = estimator_name).plot()
    plt.title(f'Precision Recall for {estimator_name}')
    plt.show()


def model_evaluation(models, xtrain, ytrain, experiment_number):

    print('Starting evaluation.....')
    
    """This function evaluates the performance of several models. It displays performance plots like confusion_matrix,
    roc_curve, precision-recall curve
    
    Input:
    models: list of models and names
    xtrain: independent data
    ytrain: dependent data
    experiment_number: number of training experiment
    
    Outputs: 
    DataFrame with models and metrics
    
    """
    cv = StratifiedKFold(n_splits = 3, shuffle = True)
    
    names, accuracy, precision, recall, f1_scores, auc_scores = [],[],[],[],[],[]
    
    for name, model in models:
        
        names.append(f'{name}_{experiment_number}')
        
        acc = cross_val_score(model, xtrain, ytrain, cv = cv, scoring = 'accuracy').mean()
        accuracy.append(acc)
        
        prec = cross_val_score(model, xtrain, ytrain, cv = cv, scoring = 'precision').mean()
        precision.append(prec)
        
        rec = cross_val_score(model, xtrain, ytrain, cv = cv, scoring = 'recall').mean()
        recall.append(rec)
        
        f1_ = cross_val_score(model, xtrain, ytrain, cv = cv, scoring = 'f1').mean()
        f1_scores.append(f1_)
        
        auc = cross_val_score(model, xtrain, ytrain, cv = cv, scoring = multi_roc_scorer, error_score="raise").mean()
        auc_scores.append(auc)
        
        yproba = cross_val_predict(model, xtrain, ytrain, cv = cv, method = 'predict_proba')[:, 1]
        ypred = cross_val_predict(model, xtrain, ytrain, cv = cv)
        
        fpr, tpr, _ = roc_curve(ytrain, yproba)
        
        precisions, recalls, _ =  precision_recall_curve(ytrain, yproba)
        
        cm = confusion_matrix(ytrain, ypred)
        
        performance_plots(cm,fpr,tpr,auc,name, precisions,recalls)
        
        print(classification_report(ytrain, ypred))
        
        
    return pd.DataFrame({
    'Model': names,
    'AUC': auc_scores,
    'F1': f1_scores,
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall
    })

def evaluate_sets(ytrue, ypred, yproba, set_type, name):

    print('Starting evaluation.....')
    
    """This function evaluates the performance of a fitted model. It displays performance plots like confusion_matrix,
    roc_curve, precision-recall curve
    
    Input:
    ytrue: true values of y
    ypred: predicted values of y
    yproba: predicted probabilities of y
    
    Outputs: 
    DataFrame with models and metrics, and plots
    
    """
    
    accuracy, precision, recall, f1_scores, auc_scores = [],[],[],[],[]
            
    acc = accuracy_score(ytrue, ypred)
    accuracy.append(acc)

    prec =  precision_score(ytrue, ypred)
    precision.append(prec)

    rec =  recall_score(ytrue, ypred)
    recall.append(rec)

    f1_ =  f1_score(ytrue, ypred)
    f1_scores.append(f1_)

    auc =  roc_auc_score(ytrue, ypred)
    auc_scores.append(auc)


    fpr, tpr, _ = roc_curve(ytrue, yproba)

    precisions, recalls, _ =  precision_recall_curve(ytrue, yproba)

    cm = confusion_matrix(ytrue, ypred)

    performance_plots(cm,fpr,tpr,auc,name, precisions,recalls)

    print(classification_report(ytrue, ypred))
        
        
    return pd.DataFrame({
    'Set type': set_type,
    'AUC': auc_scores,
    'F1': f1_scores,
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall
    }, index = [0])