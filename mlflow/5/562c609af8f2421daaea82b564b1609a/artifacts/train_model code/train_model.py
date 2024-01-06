import pandas as pd
from catboost import CatBoostRegressor
import pickle
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://0.0.0.0:5000")
mlflow.set_experiment("train_data")
with mlflow.start_run():

    X_train = pd.read_csv('/home/data-srv/project/datasets/X_train.csv')
    y_train = pd.read_csv('/home/data-srv/project/datasets/y_train.csv')

    cat = CatBoostRegressor(iterations=100, random_seed=42, loss_function='RMSE')
    cat.fit(X_train, y_train)
    mlflow.log_artifact(local_path="/home/data-srv/project/scripts/train_model.py",artifact_path="train_model code")
    mlflow.end_run()
    # Save the model using pickle
    with open('/home/data-srv/project/models/ada1.pickle', 'wb') as model_file:
        pickle.dump(cat, model_file)
