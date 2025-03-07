"""NYT Sentiment Forecast DAG with New Data"""

import os
from airflow import DAG
from airflow.models import Variable
from airflow.utils.dates import days_ago
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
import pandas as pd
import mlflow
from prophet import Prophet
import psycopg2

NEONDB_URI = Variable.get("NeonDBName")
MLFLOW_URI = Variable.get("MLFLOW_TRACKING_URI")
AWS_ACCESS_KEY_ID = Variable.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = Variable.get("AWS_SECRET_ACCESS_KEY")

# Default Arguments
default_args = {
    "owner": "airflow",
    "start_date": days_ago(1),
    "catchup": False,
}

# DAG Definition
dag = DAG(
    dag_id="nyt_sentiment_forecast_newdata",
    default_args=default_args,
    schedule_interval="@daily",
    catchup=False,
)


def fetch_data_from_s3(**context):
    """Fetch data from S3"""
    s3_hook = S3Hook(aws_conn_id="aws_s3")
    file_path = s3_hook.download_file(
        key="training/df_ny_times.json", bucket_name="nytimes-etl", local_path="/tmp"
    )
    context["task_instance"].xcom_push(key="s3_file", value=file_path)


fetch_s3_task = PythonOperator(
    task_id="fetch_data_from_s3", python_callable=fetch_data_from_s3, dag=dag
)


def load_recent_data_from_neondb(**context):
    """Load recent data from NeonDB"""
    conn = psycopg2.connect(Variable.get("NeonDBName"))
    query = "SELECT id, title, abstract, published_date FROM nyt_business_articles"
    df = pd.read_sql(query, conn)
    conn.close()

    recent_data_path = "/tmp/recent_neondb.csv"
    df.to_csv(recent_data_path, index=False)

    context["task_instance"].xcom_push(key="neon_file", value=recent_data_path)


load_neondb_task = PythonOperator(
    task_id="load_recent_data_from_neondb",
    python_callable=load_recent_data_from_neondb,
    dag=dag,
)


def preprocess_and_sentiment(**context):
    """Preprocess data and perform sentiment analysis"""
    os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface_cache"
    os.environ["HF_HOME"] = "/tmp/huggingface_cache"
    os.makedirs("/tmp/huggingface_cache", exist_ok=True)

    from transformers import pipeline

    s3_file = context["task_instance"].xcom_pull(
        key="s3_file", task_ids="fetch_data_from_s3"
    )
    recent_file = context["task_instance"].xcom_pull(
        key="neon_file", task_ids="load_recent_data_from_neondb"
    )

    if not recent_file:
        raise FileNotFoundError("Loading from NeonDB failed.")

    df_s3 = pd.read_json(s3_file)
    df_neon = pd.read_csv(recent_file)

    df_neon = df_neon.rename(
        columns={
            "title": "snippet",
            "abstract": "lead_paragraph",
            "published_date": "pub_date",
        }
    )
    df_neon["section_name"] = "Business Day"

    df_combined = pd.concat([df_s3, df_neon], ignore_index=True)
    df_business = df_combined[df_combined["section_name"] == "Business Day"]

    df_business["pub_date"] = pd.to_datetime(
        df_business["pub_date"], errors="coerce", utc=True
    )

    invalid_dates = df_business[df_business["pub_date"].isna()]
    if not invalid_dates.empty:
        print(
            "Invalid values for pub_date :",
            invalid_dates[["snippet", "pub_date"]].head(),
        )

    df_business["pub_date"] = df_business["pub_date"].dt.tz_localize(None)
    df_business["text"] = df_business["snippet"] + " " + df_business["lead_paragraph"]

    pipe = pipeline(
        "text-classification", model="tabularisai/multilingual-sentiment-analysis"
    )
    sentiments = [pipe(text[:512])[0] for text in df_business["text"]]

    df_result = pd.concat(
        [df_business.reset_index(drop=True), pd.DataFrame(sentiments)], axis=1
    )
    df_result.to_csv("/tmp/sentiments.csv", index=False)


preprocess_task = PythonOperator(
    task_id="preprocess_and_sentiment",
    python_callable=preprocess_and_sentiment,
    dag=dag,
)


def prophet_forecast():
    """Forecast sentiment using Prophet"""
    df_sentiments = pd.read_csv("/tmp/sentiments.csv")
    df_sentiments["date"] = pd.to_datetime(df_sentiments["pub_date"]).dt.date
    df_pivot = df_sentiments.pivot_table(
        index="date", columns="label", values="score", aggfunc="mean"
    ).reset_index()

    forecasts = []
    for label in df_pivot.columns[1:]:
        df_label = (
            df_pivot[["date", label]]
            .rename(columns={"date": "ds", label: "y"})
            .dropna()
        )
        model = Prophet()
        model.fit(df_label)
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)
        forecast["label"] = label
        forecasts.append(forecast[["ds", "yhat", "label"]])

    pd.concat(forecasts).to_csv("/tmp/forecast.csv", index=False)


forecast_task = PythonOperator(
    task_id="prophet_forecast", python_callable=prophet_forecast, dag=dag
)


def store_forecasts_neondb():
    """Store forecasts in NeonDB"""
    conn = psycopg2.connect(NEONDB_URI)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS sentiment_forecasts (
            id SERIAL PRIMARY KEY,
            date TIMESTAMP,
            sentiment_label TEXT,
            prediction FLOAT
        );
    """
    )

    df_forecast = pd.read_csv("/tmp/forecast.csv")
    for _, row in df_forecast.iterrows():
        cursor.execute(
            "INSERT INTO sentiment_forecasts (date, sentiment_label, prediction) VALUES (%s,%s,%s)",
            (row["ds"], row["label"], row["yhat"]),
        )
    conn.commit()
    cursor.close()
    conn.close()


store_task = PythonOperator(
    task_id="store_forecasts_neondb", python_callable=store_forecasts_neondb, dag=dag
)


def track_mlflow():
    """Track metrics and artifacts in MLflow"""
    mlflow.set_tracking_uri(Variable.get("MLFLOW_TRACKING_URI"))

    os.environ["AWS_ACCESS_KEY_ID"] = Variable.get("AWS_ACCESS_KEY_ID")
    os.environ["AWS_SECRET_ACCESS_KEY"] = Variable.get("AWS_SECRET_ACCESS_KEY")

    mlflow.set_experiment("nyt-sentiment-forecast")

    with mlflow.start_run():
        df_forecast = pd.read_csv("/tmp/forecast.csv")

        for sentiment in df_forecast["label"].unique():
            avg_pred = df_forecast[df_forecast["label"] == sentiment]["yhat"].mean()
            mlflow.log_metric(f"average_prediction_{sentiment}", avg_pred)

        mlflow.log_artifact("/tmp/forecast.csv", "forecast_data")


mlflow_task = PythonOperator(
    task_id="track_mlflow", python_callable=track_mlflow, dag=dag
)

start = DummyOperator(task_id="start", dag=dag)
end = DummyOperator(task_id="end", dag=dag)

(
    start
    >> [fetch_s3_task, load_neondb_task]
    >> preprocess_task
    >> forecast_task
    >> store_task
    >> mlflow_task
    >> end
)
