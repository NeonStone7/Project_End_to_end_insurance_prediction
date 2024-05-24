import pandas as pd

# hyperparameter tuning
from hyperopt import hp
from hyperopt import tpe, hp, fmin, STATUS_OK,Trials, partial

import xgboost as xgb

# data settings
pd.pandas.set_option('display.max_rows',None)
pd.pandas.set_option('display.max_columns',None)
pd.set_option('display.max_colwidth', None)


from sklearn import set_config
set_config(transform_output = 'pandas')
from sklearn.model_selection import cross_val_score

import warnings
warnings.simplefilter(action="ignore")

########## xgboost #############
xgb_search_space = {
    'n_estimators': hp.quniform('n_estimators', 100, 1000, 1),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
    'max_depth': hp.quniform('max_depth', 3, 15, 1),
    'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
    'gamma': hp.uniform('gamma', 0, 5),
    'reg_alpha': hp.uniform('reg_alpha', 0, 1),
    'reg_lambda': hp.uniform('reg_lambda', 0, 1),
    'scale_pos_weight': hp.quniform('scale_pos_weight', 1, 10, 1),
}

def objective(params, n_folds, x, y):
    
    params['scale_pos_weight'] = int(params['scale_pos_weight'])
    params['max_depth'] = int(params['max_depth'])
    params['n_estimators'] = int(params['n_estimators'])
    params['min_child_weight'] = int(params['min_child_weight'])
    
    model = xgb.XGBClassifier(**params, random_state = 0)
    
    scores = cross_val_score(model, x, y, cv = n_folds, scoring = 'f1', error_score='raise')
    
    max_score = max(scores)
    
    # to minimize
    loss = 1 - max_score
    
    return {'loss':loss,
           'params':params,
           'status': STATUS_OK}


def run_hyperparameter_tuning(xtrain_exp1, ytrain_exp1):
    
    #  optimize with the TPE algorithm
    trials_x = Trials()
    n_folds = 4

    best = fmin(fn = partial(objective, n_folds = n_folds,
                            x = xtrain_exp1, y = ytrain_exp1),
            space = xgb_search_space, algo = tpe.suggest, max_evals = 8, trials = trials_x,
            )
    int_hyperparams = ['max_depth', 'min_child_samples', 'n_estimators', 'num_leaves', 'min_samples_split',
                  'max_leaf_nodes', 'scale_pos_weight', 'min_child_weight']

    for hyperparam in int_hyperparams:
          
        if hyperparam in best:
            
            best[hyperparam] = int(best[hyperparam])
    return trials_x.best_trial['result']['loss'], best