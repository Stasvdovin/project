import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
import requests

mlflow.set_tracking_uri("http://0.0.0.0:5000")
mlflow.set_experiment("get_model")
with mlflow.start_run():
    url = "https://raw.githubusercontent.com/Stasvdovin/mlops_2/main/cars_moldova.csv"
    df = pd.read_csv(url)
    df.to_csv("/home/data-srv/project/datasets/df.csv", index=False)
    mlflow.log_artifact(local_path="/home/data-srv/project/scripts/get_data.py", artifact_path="get_data code")
    mlflow.end_run()
