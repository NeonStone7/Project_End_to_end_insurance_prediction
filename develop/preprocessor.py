from data_retrieval import load_data
from package.training.model_training import split_data_by_type
from package.preprocessing.data_preprocessing import create_pipeline, winsorize
import pandas as pd
import numpy as np
from category_encoders import OneHotEncoder
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import mlflow
from mlflow import MlflowClient
# data settings
pd.pandas.set_option('display.max_rows',None)
pd.pandas.set_option('display.max_columns',None)
pd.set_option('display.max_colwidth', None)

from sklearn import set_config
set_config(transform_output = 'pandas')

import warnings
warnings.simplefilter(action="ignore")
import json
from mlflow_utils import set_or_create_mlflow_experiment

##################################################
config_path = "./config.json"

# Opening JSON file
with open(config_path) as file:
    
    config = json.load(file)


def split_and_load():
    
    full_train_df, validation_df, test_df = load_data()

    # split data into dependent and independent and by type
    _, _, xtrain, ytrain = split_data_by_type(full_train_df, config['EDA']['target'])
    _, _, xval, yval = split_data_by_type(validation_df, config['EDA']['target'])
    _, _, xtest, ytest = split_data_by_type(test_df, config['EDA']['target'])

    return xtrain, ytrain, xval, yval, xtest, ytest

def load_pipeline():

    """Create the pipeline for preprocessing"""

    categorical_col_input = config['Model Training']['Preprocessor']['Categorical_input_columns'] 
    numerical_col_input = config['Model Training']['Preprocessor']['Numerical_input_columns'] 
    
    num_pipeline = Pipeline([
        ('transformer', FunctionTransformer(np.log1p)),
        ('imputer', SimpleImputer(strategy = 'median')),
        ('winsorize', FunctionTransformer(winsorize)),
        ('scaler', StandardScaler())])
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy = 'most_frequent')),
        ('encoder', OneHotEncoder())])
    
    main_pipeline = create_pipeline(num_pipeline, numerical_col_input, cat_pipeline, categorical_col_input)

    return main_pipeline

xtrain, ytrain, xval, yval, xtest, ytest = split_and_load()
preprocessor = load_pipeline()

# set variables from mlflow and save to config
config['Develop'] = {}
config['Develop']['Preprocessor'] = {}

experiment_name = config['Develop']['Preprocessor']['experiment_name'] = 'Porto_Seguro_Insurance_prediction'
run_name = config['Develop']['Preprocessor']['run_name'] = 'creating_preprocessor'
preprocessor_model_name = config['Develop']['Preprocessor']['preprocessor_model_name'] = 'preprocessor'
artifact_path = config['Develop']['Preprocessor']['artifact_path'] = 'preprocessor_artifacts'

# set or create the experiment and retrieve the experiment_id
experiment_id = set_or_create_mlflow_experiment(experiment_name, artifact_path)

# create a wrapper so we can log it with mlflow
class PreprocessorWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor

    def predict(self, context, model_input):
        return self.preprocessor.transform(model_input)


with mlflow.start_run(run_name = run_name, experiment_id = experiment_id) as preprocessor_run:

    client = MlflowClient()
    preprocessor.fit(xtrain)

    preprocessor_wrapper = PreprocessorWrapper(preprocessor)

    model_info = mlflow.pyfunc.log_model(
        artifact_path = artifact_path,
        python_model = preprocessor_wrapper)
    
    print('Preprocessor logged successfully')

    config['Develop']['Preprocessor']['run_id'] = preprocessor_run.info.run_id

    # register a new model version preprocessor
    model_version_info = client.create_model_version(
        name = preprocessor_model_name,
        source = f'{model_info.artifact_path}/{artifact_path}',
        run_id = preprocessor_run.info.run_id
    )

    # stage model to production
    client.transition_model_version_stage(name = preprocessor_model_name,
                                          version = model_version_info.version,
                                         stage = 'Production')
    
    print(f'Staged to production, preprocessor version: {model_version_info.version}')
    
    config['Develop']['Preprocessor']['version'] = model_version_info.version

    # archive previous version
    if config['Develop']['Preprocessor']['version']:
        
        client.transition_model_version_stage(name = preprocessor_model_name,
                                              version = config['Develop']['Preprocessor']['version'] - 1,
                                              stage = 'Archived')
        
        print(f"Archived preprocessor version: {config['Develop']['Preprocessor']['version']-1}")


# save config
json_object = json.dumps(config, indent=4)
 
with open("./config.json", "w") as outfile:
    outfile.write(json_object)



