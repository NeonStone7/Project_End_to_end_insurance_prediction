import mlflow

def set_or_create_mlflow_experiment(experiment_name, artifact_location, tags = {'env':'dev','version':'1.0.0'}):

    """Create the experiment if it does not exist else set the experiment
    Returns: experiment _id for the experiment"""

    try:
        # if the experiment doesn't exist, create it
        exp_id = mlflow.create_experiment(name = experiment_name, 
                                          artifact_location = artifact_location,
                                          tags = tags)
        print('Created Experiment')
    except:
        # if it exists, get the experiment id
        exp_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        print('Extracted experiment')

    finally:
        # set the experiment
        mlflow.set_experiment(experiment_id=exp_id)

    return exp_id

