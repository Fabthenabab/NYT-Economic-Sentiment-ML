""" 
ETL DAG for fetching, transforming, and storing NYT Business data in S3 and NeonDB.
This DAG performs the following tasks:
1. Fetch raw data from the NYT API and store it in an S3 bucket.
2. Transform the raw data and store it in NeonDB (PostgreSQL).
3. Ensure the table exists before inserting data.
"""

import json
import logging
from datetime import datetime
import pandas as pd
import requests
from airflow import DAG
from airflow.models import Variable
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python_operator import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
import psycopg2

# Environment Variables (from Airflow UI)
NYT_API_KEY = Variable.get("NytApiKey")  # API Key for NYT API
NEONDB_URI = Variable.get("NeonDBName")  # Database connection URI for NeonDB
S3_BUCKET = "nytimes-etl"  # Name of the S3 bucket to store raw data

# DAG Definition
default_args = {
    "owner": "airflow",
    "start_date": datetime(2025, 3, 5),
    "catchup": False,  # Prevents running past executions
}

dag = DAG(
    "nyt_etl_s3_pipeline",
    default_args=default_args,
    schedule_interval="@hourly",  # Runs every hour
)


# Fetch and Store Raw Data in S3
def fetch_and_store_raw_data(**context):
    """Fetches data from the NYT API and stores it in an S3 bucket."""

    url = f"https://api.nytimes.com/svc/news/v3/content/all/business.json?api-key={NYT_API_KEY}"
    response = requests.get(url)

    if response.status_code == 200:
        articles = response.json()["results"]
        raw_data = json.dumps(articles)

        # Generate filename with timestamp for uniqueness
        raw_filename = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_nyt_raw.json"
        local_path = f"/tmp/{raw_filename}"

        # Save the raw JSON file locally before uploading to S3
        with open(local_path, "w") as f:
            f.write(raw_data)

        # Upload file to S3
        s3_hook = S3Hook(aws_conn_id="aws_s3")
        s3_hook.load_file(
            filename=local_path, key=f"raw_data/{raw_filename}", bucket_name=S3_BUCKET
        )

        # Push filename to XCom for later use
        context["task_instance"].xcom_push(
            key="raw_s3_filename", value=f"raw_data/{raw_filename}"
        )
        logging.info(f"Raw data stored in S3: {raw_filename}")


fetch_task = PythonOperator(
    task_id="fetch_and_store_raw_data",
    python_callable=fetch_and_store_raw_data,
    dag=dag,
)


# Transform and Store Processed Data in NeonDB
def transform_and_store(**context):
    """Downloads raw data from S3, transforms it, and stores it in NeonDB."""
    
    # Retrieve the raw file name from XCom
    s3_key = context["task_instance"].xcom_pull(key="raw_s3_filename")

    # Extract the filename from the S3 key
    local_filename = s3_key.split("/")[-1]
    local_path = f"/tmp/{local_filename}"

    # Download file from S3 into `/tmp/`
    s3_hook = S3Hook(aws_conn_id="aws_s3")
    s3_hook.download_file(key=s3_key, bucket_name=S3_BUCKET, local_path="/tmp")

    # Load JSON from the downloaded file
    with open(local_path, "r") as f:
        articles = json.load(f)
    df = pd.DataFrame(articles)

    # Connect to NeonDB
    conn = psycopg2.connect(NEONDB_URI)
    cursor = conn.cursor()

    # Ensure the target table exists before inserting data
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS nyt_business_articles (
            id SERIAL PRIMARY KEY,
            title TEXT NOT NULL,
            abstract TEXT,
            published_date TIMESTAMP NOT NULL,
            UNIQUE (title, published_date)
        );
    """)

    # ✅ Insert with "ON CONFLICT DO NOTHING" to avoid duplicate key errors
    for _, row in df.iterrows():
        cursor.execute(
            """
            INSERT INTO nyt_business_articles (title, abstract, published_date)
            VALUES (%s, %s, %s)
            ON CONFLICT (title, published_date) DO NOTHING;
            """,
            (row["title"], row["abstract"], row["published_date"]),
        )

    # Commit transaction and close connection
    conn.commit()
    cursor.close()
    conn.close()
    logging.info("Transformed data stored in NeonDB without duplicates.")



transform_task = PythonOperator(
    task_id="transform_and_store",
    python_callable=transform_and_store,
    dag=dag,
)

# Ensure Table Exists Before Inserting Data
create_table = PostgresOperator(
    task_id="create_nyt_table",
    sql="""
        CREATE TABLE IF NOT EXISTS nyt_business_articles (
            id SERIAL PRIMARY KEY,
            title TEXT NOT NULL,
            abstract TEXT,
            published_date TIMESTAMP NOT NULL,
            UNIQUE (title, published_date)
        );
    """,
    postgres_conn_id="postgres_default",
    dag=dag,
)

# DAG Flow
start = DummyOperator(task_id="start", dag=dag)
end = DummyOperator(task_id="end", dag=dag)

start >> fetch_task >> create_table >> transform_task >> end
