from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from datetime import timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    dag_id = 'retrain_model_dag',
    default_args=default_args,
    description='A simple DAG to retrain model',
    schedule_interval=timedelta(days=30),  # Adjust the schedule as needed
    start_date=days_ago(1),
    tags=['example'],
)

t1 = BashOperator(
    task_id='retrain_model_task',
    bash_command='python ./Insurance_claim_prediction_porto/develop/training_model.py',  # Update the path to your script
    dag=dag,
)

t1
