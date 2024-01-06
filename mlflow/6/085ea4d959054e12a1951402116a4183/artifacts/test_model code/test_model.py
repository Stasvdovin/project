import pandas as pd
from sklearn.metrics import accuracy_score
import pickle
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://0.0.0.0:5000")
mlflow.set_experiment("test_model")

with mlflow.start_run():
    X_test = pd.read_csv('/home/data-srv/project/datasets/X_test.csv')
    y_test = pd.read_csv('/home/data-srv/project/datasets/y_test.csv')
    with open('/home/data-srv/project/models/ada1.pickle', 'rb') as model_file:
        clf = pickle.load(model_file)

        predicted_label_y = clf.predict(X_test)
        predicted_label_y = predicted_label_y.astype(int)
        score = accuracy_score(predicted_label_y, y_test)
        print("score=",score)
        mlflow.log_artifact(local_path='/home/data-srv/project/scripts/test_model.py',artifact_path="test_model code")
        mlflow.log_metric("score", score)
        mlflow.end_run()
