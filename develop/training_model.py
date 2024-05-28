"""To train model and register to mlflow"""
#import preprocessor
from sklearn import set_config
set_config(transform_output = 'pandas')
import warnings
warnings.simplefilter(action="ignore")
import json
from package.training.model_training import evaluate_sets 
from mlflow_utils import set_or_create_mlflow_experiment
import mlflow
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from preprocessor import split_and_load
from hyperparameter_tuning import run_hyperparameter_tuning
import pandas as pd
import xgboost as xgb
from imblearn.under_sampling import RandomUnderSampler
pd.pandas.set_option('display.max_rows',None)
pd.pandas.set_option('display.max_columns',None)
pd.set_option('display.max_colwidth', None)
###################################################################################44

# xtrain, ytrain, xval, yval, xtest, ytest = split_and_load()
# preprocessor = load_pipeline()

# open config
CONFIG_PATH = "./config.json"

# Opening JSON file
with open(CONFIG_PATH, 'r') as file:
    
    config = json.load(file)

config['Develop']['Model'] = {}

experiment_name = config['Develop']['Preprocessor']['experiment_name'] 
run_name = config['Develop']['Model']['run_name'] = 'creating_insurance_model'
model_name = config['Develop']['Model']['model_name'] = 'insurance_model'
artifact_path = config['Develop']['Model']['artifact_path'] = 'model_artifacts'
preprocessor_run_id = config['Develop']['Preprocessor']['run_id']
preprocessor_artifact_path = config['Develop']['Preprocessor']['artifact_path']

# set or create the experiment and retrieve the experiment_id
experiment_id = set_or_create_mlflow_experiment(experiment_name, artifact_path)

# retrieve split data
xtrain, ytrain, xval, yval, xtest, ytest = split_and_load()

# load prepreprocessor
preprocessor_uri = f"runs:/{preprocessor_run_id}/{preprocessor_artifact_path}"
preprocessor_model = mlflow.pyfunc.load_model(preprocessor_uri)

xtrain_processed = preprocessor_model.predict(xtrain)
xval_processed = preprocessor_model.predict(xval)
xtest_processed = preprocessor_model.predict(xtest)
config['Develop']['Model']['all_columns'] = xtrain_processed

# resample train data
under_sampler = RandomUnderSampler()
xtrain_samp, ytrain_samp = under_sampler.fit_resample(xtrain_processed, ytrain)

# feature selection
rf = RandomForestClassifier()
selector = SelectFromModel(rf, threshold = 0.001).fit(xtrain_samp, ytrain_samp)
# select features and save them to config
selected_columns = xtrain_samp.columns[selector.get_support()]
config['Develop']['Model']['selected_columns'] = selected_columns.tolist()

# filter data down to selected columns
xtrain_samp = xtrain_samp[selected_columns]
xval_processed = xval_processed[selected_columns]
xtest_processed = xtest_processed[selected_columns]

experiment_id = set_or_create_mlflow_experiment(experiment_name, artifact_path)

config['Develop']['Hyperparameter_Tuning'] = {}
hyp_run_name = config['Develop']['Hyperparameter_Tuning']['run_name'] = 'hyperparameter_tuning'

with mlflow.start_run(run_name=run_name, experiment_id=experiment_id, nested=True) as model_run:

    client = MlflowClient()
    
    # with mlflow.start_run(run_name = hyp_run_name, experiment_id=experiment_id, nested=True) as tuning_run:
        
    #     # perform hyperparameter tuning and save best hyperparameters and loss to config
    #     loss, hyperparams = run_hyperparameter_tuning(xval_processed, yval)
    #     mlflow.log_params(hyperparams)

    #     config['Develop']['Hyperparameter_Tuning']['Best hyperparameters'] = hyperparams
    #     config['Develop']['Hyperparameter_Tuning']['Best hyperparameters'] = hyperparams
    #     config['Develop']['Hyperparameter_Tuning']['loss'] = loss
        
    #     mlflow.end_run()
    # instantiate model with best hyperparameters
    hyperparams = config['Hyperparameter Tuning']['best_hyperparameters']['Xgboost']
    model = xgb.XGBClassifier(**hyperparams)

    # automatic logging of artifacts
    mlflow.autolog()

    model.fit(xtrain_samp, ytrain_samp)

    y_train_pred = model.predict(xtrain_samp)

    y_train_proba = model.predict_proba(xtrain_samp)[:, 1]

    # evaluate model performance on test set
    train_results = evaluate_sets(ytrain_samp, y_train_pred, y_train_proba, 
                              'Train', 'Train')
    
    # log training metrics
    train_results_dict = train_results[['AUC', 'F1', 'Accuracy', 'Precision', 'Recall']].to_dict()
    # return only metrics and values
    train_results_dict_re = {metric:list(value.values())[0] for metric, value in train_results_dict.items()}

    mlflow.log_metrics(train_results_dict_re)
    config['Develop']['Model']['Training metrics'] = train_results_dict_re


    # evaluate model performance on test set
    ytest_pred = model.predict(xtest_processed)
    ytest_proba = model.predict_proba(xtest_processed)[:, 1]

    test_results = evaluate_sets(ytest, ytest_pred, ytest_proba, 
                              'Test', 'Test')
    
    # log test metrics

    test_results_res = test_results[['AUC', 'F1', 'Accuracy', 'Precision', 'Recall']].to_dict()
    test_results_res_dict = {metric:list(value.values())[0] for metric, value in test_results_res.items()}

    mlflow.log_metrics(test_results_res_dict)
    config['Develop']['Model']['Test metrics'] = test_results_res_dict

    # log model manually
    print('logging model')
    model_info = mlflow.sklearn.log_model(sk_model = model, artifact_path = artifact_path, registered_model_name = model_name)
    print('Model Logged successfully')  

    # save model run
    model_run_id = config['Develop']['Model']['run_id'] = model_run.info.run_id

    # register model 
    model_version_info = client.create_model_version(
        name = model_name,
        source = f'{model_info.artifact_path}/{artifact_path}',
        run_id = model_run_id
    )

    # transition to production
    client.transition_model_version_stage(
        name = model_name,
        version = model_version_info.version,
        stage = 'Production'
    )

    model_version = config['Develop']['Model']['model_version'] = model_version_info.version

    print(f'Transitioned to prod: Model Version {model_version}')

    # transition previous model to staging
    if config['Develop']['Model']['model_version']:
        client.transition_model_version_stage(
            name = model_name,
            version = model_version_info.version - 1,
            stage = 'Staging'
        )
        print(f'Staged previous model version:{model_version_info.version - 1}')

# save to config
json_object = json.dumps(config, indent=4)
with open(CONFIG_PATH, "w") as outfile:
    outfile.write(json_object)
