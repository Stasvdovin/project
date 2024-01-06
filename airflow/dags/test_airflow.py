from airflow import DAG
from airflow.operators.bash import BashOperator
import pendulum
import datetime as dt

# Определяем аргументы по умолчанию для DAG
args = {
    "owner": "admin",  # Владелец DAG
    "start_date": dt.datetime(2019, 1, 1),  # Начальная дата выполнения DAG
    "retries": 1,  # Количество попыток выполнения для каждой задачи
    "retry_delays": dt.timedelta(minutes=1),  # Время задержки между повторными попытками
    "depends_on_past": False  # Зависят ли задачи от успешного выполнения предыдущих задач
}

# Создаем новый DAG в Airflow с указанным идентификатором, аргументами по умолчанию и без интервала выполнения
with DAG(dag_id='test_score', default_args=args, schedule=None, tags=['test_cars']) as dag:
    # Определяем каждую задачу как BashOperator, предоставляя идентификатор задачи и команду Bash для выполнения

    # Задача для получения данных (Запускает скрипт Python get_data.py)
    get_data = BashOperator(task_id='get_data',
        bash_command='python /home/data-srv/project/scripts/get_data.py',
        dag=dag)

    # Задача для подготовки данных (Запускает скрипт Python preprocess_data.py)
    prepare_data = BashOperator(task_id='prepare_data',
        bash_command='python /home/data-srv/project/scripts/process_data.py',
        dag=dag)

    # Задача для разделения данных на обучающую и тестовую выборки (Запускает скрипт Python train_test_split.py)
    train_test_split = BashOperator(task_id='train_test_split',
        bash_command='python /home/data-srv/project/scripts/train_test_split.py',
        dag=dag)

    # Задача для обучения модели машинного обучения (Запускает скрипт Python train_model.py)
    train_model = BashOperator(task_id='train_model',
        bash_command='python /home/data-srv/project/scripts/train_model.py',
        dag=dag)

    # Задача для тестирования модели машинного обучения (Запускает скрипт Python test_model.py)
    test_model = BashOperator(task_id='test_model',
        bash_command='python /home/data-srv/project/scripts/test_model.py',
        dag=dag)

    # Определяем зависимости задач, указывая порядок их выполнения, чтобы обеспечить выполнение в нужной последовательности
    get_data >> prepare_data >> train_test_split >> train_model >> test_model
