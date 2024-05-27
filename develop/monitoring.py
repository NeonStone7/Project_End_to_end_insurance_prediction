"""Module to set up monitoring for model. This script will use EvidentlyAI
to check for drift"""

# Import necessary modules and functions

import json
import warnings
warnings.simplefilter(action="ignore")
from sklearn import set_config
set_config(transform_output='pandas')
from data_retrieval import load_data
import pandas as pd
import evidently
from evidently.dashboard import Dashboard
from evidently.tabs import DataDriftTab,CatTargetDriftTab
 
# Data settings
pd.pandas.set_option('display.max_rows', None)
pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

# Load configuration
CONFIG_PATH = "./config.json"
with open(CONFIG_PATH, 'r') as file:
    config = json.load(file)

# read raw data
full_train_df, _, test_df = load_data()

# since we don't have oot sample, we will take a random set from test
oot = test_df.sample(2000)

config['Monitoring'] = {}

monitor_dashboard_path = config['Monitoring']['dashboard_path'] = './deploy/data_and_target_drift_dashboard.html'

# Create instances of the tab classes
data_drift_tab = DataDriftTab()
cat_target_drift_tab = CatTargetDriftTab()

# Pass the instances to the Dashboard
drift_dashboard = Dashboard(tabs=[data_drift_tab, cat_target_drift_tab])
drift_dashboard.calculate(full_train_df, oot, column_mapping = None)

drift_dashboard.save(monitor_dashboard_path)


