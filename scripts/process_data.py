import pandas as pd
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient


mlflow.set_tracking_uri("http://0.0.0.0:5000")
mlflow.set_experiment("process_data")
with mlflow.start_run():
    df = pd.read_csv('/home/data-srv/project/datasets/df.csv')
    # Почистим данные
    question_dist = df[(df.Year <2021) & (df.Distance < 1100)]
    df = df.drop(question_dist.index)
    question_dist = df[(df.Distance > 1e6)]
    df = df.drop(question_dist.index)
    question_engine = df[df["Engine_capacity(cm3)"] < 200]
    df = df.drop(question_engine.index)
    question_engine = df[df["Engine_capacity(cm3)"] > 5000]
    df = df.drop(question_engine.index)
    question_price = df[(df["Price(euro)"] < 101)]
    df = df.drop(question_price.index)
    question_price = df[df["Price(euro)"] > 1e5]
    df = df.drop(question_price.index)
    question_year = df[df.Year < 1971]
    df = df.drop(question_year.index)
    df = df.reset_index(drop=True)

    # Список имён колонок с числовыми и категориальными значениями
    num_columns = list(df.select_dtypes(include='number').columns)
    cat_columns = list(df.select_dtypes(exclude='number').columns)

    df[num_columns] = df[num_columns].fillna(0)
    df[cat_columns] = df[cat_columns].fillna("")

    # добавляем новый признак
    df['Age'] = 2022 - df.Year
    df['km_year'] = df.Distance/df.Age
    question_km_year = df[df.km_year > 50e3]
    df = df.drop(question_km_year.index)
    question_km_year = df[df.km_year < 100]
    df = df.drop(question_km_year.index)
    df = df.reset_index(drop=True)

    df.drop(['Make',	'Model',	'Style',	'Fuel_type',	'Transmission'],
    axis=1, inplace=True)
    df.to_csv("/home/data-srv/project/datasets/df.csv", index=False)
    mlflow.log_artifact(local_path="/home/data-srv/project/scripts/process_data.py",artifact_path="process_data code")
    mlflow.end_run()
