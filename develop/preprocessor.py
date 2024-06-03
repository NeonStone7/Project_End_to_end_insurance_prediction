"""To train preprocessor and register to mlflow"""
# Import necessary modules and functions
import json
import warnings
warnings.simplefilter(action="ignore")
from sklearn import set_config
set_config(transform_output='pandas')
from data_retrieval import load_data
from package.training.model_training import split_data_by_type
from package.preprocessing.data_preprocessing import create_pipeline, winsorize, percentile_imputer
import pandas as pd
import numpy as np
from category_encoders import  OrdinalEncoder
from sklearn.preprocessing import FunctionTransformer, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import mlflow
from mlflow import MlflowClient
from mlflow_utils import set_or_create_mlflow_experiment

# Data settings
pd.pandas.set_option('display.max_rows', None)
pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

# Load configuration
CONFIG_PATH = "./config.json"
with open(CONFIG_PATH, 'r') as file:
    config = json.load(file)

# Define functions
def split_and_load(full_train_df=None, validation_df=None, test_df=None):

    """This takes in the sets retrieved from s3 and splits it into dependent and independent variables"""
    if full_train_df is None or validation_df is None or test_df is None:
        full_train_df, validation_df, test_df = load_data()

    _, _, xtrain, ytrain = split_data_by_type(full_train_df, config['EDA']['target'])
    _, _, xval, yval = split_data_by_type(validation_df, config['EDA']['target'])
    _, _, xtest, ytest = split_data_by_type(test_df, config['EDA']['target'])

    return xtrain, ytrain, xval, yval, xtest, ytest

def load_pipeline():

    """This creates the preprocessor pipeline so we can preprocess the data"""

    categorical_col_input = config['Model Training']['Preprocessor']['Categorical_input_columns']
    numerical_col_input = config['Model Training']['Preprocessor']['Numerical_input_columns']
    
    num_pipeline = Pipeline([
        ('transformer', FunctionTransformer(np.sqrt)),
        ('imputer', FunctionTransformer(percentile_imputer)),
        ('winsorize', FunctionTransformer(winsorize)),
        ('scaler', RobustScaler()),  
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy = 'most_frequent')),
        ('encoder', OrdinalEncoder()),
        
    ])
    
    main_pipeline = create_pipeline(num_pipeline, numerical_col_input, cat_pipeline, categorical_col_input)
    return main_pipeline

class PreprocessorWrapper(mlflow.pyfunc.PythonModel):
    """Creates a custom wrapper so we can log the preprocessor model with mlflow"""
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor

    def predict(self, context, model_input):
        return self.preprocessor.transform(model_input)

# Main logic for preprocessor.py
def main():
    """We create this so when we import this script in another, it doesn't run thw whole script"""
    xtrain, _, _, _, _, _ = split_and_load()
    preprocessor = load_pipeline()

    config['Develop'] = {}
    config['Develop']['Preprocessor'] = {}

    experiment_name = config['Develop']['Preprocessor']['experiment_name'] = 'Porto_Seguro_Insurance_prediction'
    run_name = config['Develop']['Preprocessor']['run_name'] = 'creating_preprocessor'
    preprocessor_model_name = config['Develop']['Preprocessor']['preprocessor_model_name'] = 'preprocessor'
    artifact_path = config['Develop']['Preprocessor']['artifact_path'] = 'preprocessor_artifacts'

    experiment_id = set_or_create_mlflow_experiment(experiment_name, artifact_path)

    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as preprocessor_run:

        client = MlflowClient()

        # transform training set
        preprocessor.fit(xtrain)

        # pass it into the wrapper
        preprocessor_wrapper = PreprocessorWrapper(preprocessor)

        # log the preprocessor model to mlflow
        model_info = mlflow.pyfunc.log_model(
            artifact_path=artifact_path,
            python_model=preprocessor_wrapper
        )
        
        print('Preprocessor logged successfully')
        config['Develop']['Preprocessor']['Preprocessor_uri'] = model_info.model_uri

        config['Develop']['Preprocessor']['run_id'] = preprocessor_run.info.run_id

        # register the model
        model_version_info = client.create_model_version(
            name=preprocessor_model_name,
            source=f'{model_info.artifact_path}/{artifact_path}',
            run_id=preprocessor_run.info.run_id
        )

        # transition it to production
        client.transition_model_version_stage(name=preprocessor_model_name,
                                              version=model_version_info.version,
                                              stage='Production')
        
        print(f'Staged to production, preprocessor version: {model_version_info.version}')
        
        config['Develop']['Preprocessor']['version'] = model_version_info.version

        # stage the previous preprocessor model
        if config['Develop']['Preprocessor']['version']:
            client.transition_model_version_stage(name=preprocessor_model_name,
                                                  version=config['Develop']['Preprocessor']['version'] - 1,
                                                  stage='Staging')
            
            print(f"Archived preprocessor version: {config['Develop']['Preprocessor']['version']-1}")

    # Save updated config
    json_object = json.dumps(config, indent=4)
    with open(CONFIG_PATH, "w") as outfile:
        outfile.write(json_object)

# Ensure that main() runs only when this script is executed directly
if __name__ == "__main__":
    main()
