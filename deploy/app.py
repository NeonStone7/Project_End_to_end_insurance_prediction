"""This is a flask web app to perform batch prediction using the
model registered to mlflow"""
import json
import pandas as pd
from flask import Flask, render_template, request, send_file
import mlflow
from sklearn import set_config

set_config(transform_output='pandas')

app = Flask(__name__)

# Open config
CONFIG_PATH = './config.json'
with open(CONFIG_PATH, 'r') as config_file:
    config = json.load(config_file)

# Extract required variables
experiment_name = config['Develop']['Preprocessor']['experiment_name']
run_name = config['Develop']['Model']['run_name']
model_name = config['Develop']['Model']['model_name']
model_artifact_path = config['Develop']['Model']['artifact_path']
model_run_id = config['Develop']['Model']['run_id']
preprocessor_run_id = config['Develop']['Preprocessor']['run_id']
preprocessor_artifact_path = config['Develop']['Preprocessor']['artifact_path']
selected_columns = config['Develop']['Model']['selected_columns']

# get preprocessor uri
preprocessor_uri = f'./model_artifacts/{preprocessor_run_id}/artifacts/preprocessor_artifacts'
preprocessor_model = mlflow.pyfunc.load_model(preprocessor_uri)
print(f"Preprocessor URI: {preprocessor_uri}")

#model_uri = f'runs:/{model_run_id}/{model_artifact_path}'
model_uri = f'./model_artifacts/{model_run_id}/artifacts/model_artifacts'
model = mlflow.sklearn.load_model(model_uri)
print(f"Model URI: {model_uri}")


@app.route('/')
def home():
    """This is the home page. It contains links to the documentation,
      github repo, predict page etd"""
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """This is the predict function. It takes in a csv file,
      makes predictions on all samples and returns an output 
      csv file with the predicition column"""
    set_config(transform_output='pandas')

    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            input_df = pd.read_csv(file)

            # Preprocess the data
            input_df_processed = preprocessor_model.predict(input_df)
            input_df_processed = input_df_processed[selected_columns]

            # Get predictions and add to file
            input_df['Prediction'] = model.predict(input_df_processed)

            save_output_path = './deploy/output_data.csv'
            input_df.to_csv(save_output_path, index=False)

            file_name = 'output_data.csv'
            return send_file(file_name, as_attachment=True)
        
        return render_template('predict.html', message="No file provided")
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
