"""Flask app to expose model to users"""
import json
import pandas as pd
from flask import Flask, render_template, request, send_file
import mlflow

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
model_run_id =  config['Develop']['Model']['run_id'] 
preprocessor_run_id = config['Develop']['Preprocessor']['run_id']
preprocessor_artifact_path = config['Develop']['Preprocessor']['artifact_path']
selected_columns = config['Develop']['Model']['selected_columns'] 

# load model and preprocessor
preprocessor_uri = f"runs:/{preprocessor_run_id}/{preprocessor_artifact_path}"
preprocessor_model = mlflow.pyfunc.load_model(preprocessor_uri)

model_uri = f'runs:/{model_run_id}/{model_artifact_path}'
model = mlflow.sklearn.load_model(model_uri = model_uri)

@app.route('/')
def home():

    return render_template('./home.html')

@app.route('/predict', methods = ['GET', 'POST'])
def predict():
    
    if request.method == 'POST':
        file = request.file['file']
        if file:

            # read the csv file
            input_df = pd.read_csv(file)

            # preprocess the data
            input_df_processed = preprocessor_model.predict(input_df)

            input_df_processed = input_df_processed[selected_columns]

            # get predictions and add to file
            input_df['Prediction'] = model.predict(input_df_processed)

            # save the csv
            output_path = 'output_data.csv'
            input_df.to_csv(output_path, index = False)

            # return the csv
            return send_file(output_path, as_attachment = True)

if __name__ == '__main__':
    app.run(debug = True)