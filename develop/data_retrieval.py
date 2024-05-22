from package.data_retrieval.data_retrieval import s3_retrieval, write_to_s3
from package.training.model_training import model_evaluation, split_data_by_type, performance_plots
import pandas as pd
import numpy as np
import boto3
import os
import json
from dotenv import load_dotenv

# get the environment variable
ENV = os.getenv('ENV', 'local')

# if it's running locally, load the local environment variables
if ENV=='local':
    
    load_dotenv()

access_key = os.getenv('AWS_ACCESS_KEY')
access_key_id = os.getenv('AWS_ACCESS_KEY')

# create S3 client object 
s3_client = boto3.client('s3',aws_access_key_id=access_key_id,aws_secret_access_key=access_key)

config_path = "./config.json"

# Opening config file
with open(config_path) as file:
    
    config = json.load(file)

bucket_name = config['EDA']['bucket_name'] 

def load_data():
    """Retrieve data from the s3 bucket"""

    full_train_df = s3_retrieval(s3_client, bucket_name, config['Model_Selection']['Input_data']['full_train'])
    validation_df = s3_retrieval(s3_client, bucket_name, config['Model_Selection']['Input_data']['validation'])
    test_df = s3_retrieval(s3_client, bucket_name, config['Model_Selection']['Input_data']['test'])

    return full_train_df, validation_df, test_df