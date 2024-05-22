#import preprocessor
from preprocessor import split_and_load
# miscellaneous
import matplotlib.pyplot as plt
from data_retrieval import load_data

from glob import glob #library that helps us search for files
import scipy
import random
from datetime import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
from scipy.special import inv_boxcox
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

# hyperparameter tuning
from hyperopt import hp
from hyperopt import tpe, hp, fmin, STATUS_OK,Trials, partial

# preprocessing
from category_encoders import OneHotEncoder,TargetEncoder,OrdinalEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer, PowerTransformer,LabelEncoder, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV, StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score,cross_val_predict, KFold, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer, PowerTransformer,LabelEncoder, MaxAbsScaler, RobustScaler

# models
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression,BayesianRidge, ElasticNet, Lasso
from sklearn.dummy import DummyClassifier
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier, VotingClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import clone
import lightgbm as lgb

# metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
from sklearn.metrics import log_loss, confusion_matrix, classification_report, precision_recall_curve
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.utils.validation import check_is_fitted

# feature selection / data sampling
from sklearn.feature_selection import RFE, SelectKBest, f_classif, SelectFromModel, VarianceThreshold
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.inspection import permutation_importance

# saving model
import pickle,joblib
import boto3

# data settings
pd.pandas.set_option('display.max_rows',None)
pd.pandas.set_option('display.max_columns',None)
pd.set_option('display.max_colwidth', None)


from sklearn import set_config
set_config(transform_output = 'pandas')

import warnings
warnings.simplefilter(action="ignore")


import json

# custom package
import importlib
import package
importlib.reload(package)

from package.data_retrieval.data_retrieval import s3_retrieval, write_to_s3
from package.eda.data_exploration import correlation
from package.training.model_training import model_evaluation, split_data_by_type, performance_plots
from package.preprocessing.data_preprocessing import create_pipeline, winsorize, percentile_imputer, random_sample_imputer, count_encoder 
from mlflow_utils import set_or_create_mlflow_experiment
import mlflow
from mlflow import MlflowClient
###################################################################################44

# xtrain, ytrain, xval, yval, xtest, ytest = split_and_load()
# preprocessor = load_pipeline()

# open config
config_path = "./config.json"

# Opening JSON file
with open(config_path) as file:
    
    config = json.load(file)


experiment_name = config['Develop']['Preprocessor']['experiment_name'] 
run_name = config['Develop']['Preprocessor']['run_name'] 
preprocessor_model_name = config['Develop']['Preprocessor']['preprocessor_model_name'] 
artifact_path = config['Develop']['Preprocessor']['artifact_path'] 
run_id = config['Develop']['Preprocessor']['run_id'] 

# set or create the experiment and retrieve the experiment_id
experiment_id = set_or_create_mlflow_experiment(experiment_name, artifact_path)

xtrain, ytrain, xval, yval, xtest, ytest = split_and_load()


# load prepreprocessor
preprocessor_uri = f"runs:/{run_id}/{artifact_path}"
preprocessor_model = mlflow.pyfunc.load_model(preprocessor_uri)

xtrain_processed = preprocessor_model.transform(xtrain)
print(xtrain_processed[:1])


