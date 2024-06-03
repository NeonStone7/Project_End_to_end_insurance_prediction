"""To generate test sample and test model"""
import json
import pandas as pd
from flask import Flask, render_template, request, send_file
import mlflow
from sklearn import set_config
set_config(transform_output = 'pandas')
from deploy.mlflow_utils import set_or_create_mlflow_experiment
app = Flask(__name__)

# open config
CONFIG_PATH = './config.json'
with open(CONFIG_PATH, 'r') as config_file:
    config = json.load(config_file)

# extract required variables
experiment_name = config['Develop']['Preprocessor']['experiment_name']
run_name = config['Develop']['Model']['run_name']
model_name = config['Develop']['Model']['model_name']
model_artifact_path = config['Develop']['Model']['artifact_path']
model_run_id = config['Develop']['Model']['run_id']
preprocessor_run_id = config['Develop']['Preprocessor']['run_id']
preprocessor_artifact_path = config['Develop']['Preprocessor']['artifact_path']
selected_columns = config['Develop']['Model']['selected_columns']
preprocessor_version = config['Develop']['Preprocessor']['version']
preprocessor_model_name = config['Develop']['Preprocessor']['preprocessor_model_name'] 
model_version = config['Develop']['Model']['model_version'] 
model_uri = config['Develop']['Model']['Model_uri']
preprocessor_uri = config['Develop']['Preprocessor']['Preprocessor_uri'] 

# preprocessor_uri = f"models:/{preprocessor_model_name}/{preprocessor_version}"
# preprocessor_uri = mlflow.get_artifact_uri(preprocessor_artifact_path)

print(f"Preprocessor URI: {preprocessor_uri}")
preprocessor_model = mlflow.pyfunc.load_model(preprocessor_uri)

# model_uri = f'models:/{model_name}/{model_version}'
model = mlflow.sklearn.load_model(model_uri)

input_df = pd.read_csv("C:/Users/Oamen/OneDrive/Documents/DATASETS/porto_seguro_insurance_data/porto-seguro-safe-driver-prediction/test.csv")

# preprocess the data
input_df_processed = preprocessor_model.predict(input_df)
print(input_df_processed[:2])
input_df_processed = input_df_processed[selected_columns]

# get predictions and add to file
input_df['Prediction'] = model.predict(input_df_processed)
print(input_df[:2])

# take asample
#input_df.sample(2000).to_csv('test_data.csv', index = False)
# model = mlflow.sklearn.load_model(model_uri)
