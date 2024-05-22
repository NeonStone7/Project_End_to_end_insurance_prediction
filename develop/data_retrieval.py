from package.data_retrieval.data_retrieval import s3_retrieval, write_to_s3
from package.training.model_training import model_evaluation, split_data_by_type, performance_plots
import pandas as pd
import numpy as np
import boto3
import os

print(os.getenv('AWS_SECRET_ACCESS_KEY'))